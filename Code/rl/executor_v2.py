"""Executor v2 (Stage 2b): the Stage 2 transformer with (1) a mixture of K Gaussian trajectories per player instead of one Gaussian,
(2) a pass-target head (which teammate receives the first pass in the next 2 s, given that the holder passes), and (3) the same
decision head. Same inputs as v1 (executor_data tokens + context), permutation-equivariant in the player slots.

Per player the movement head outputs K modes, each with a mixture logit, a mean 1-second trajectory (5 steps x 2) and its log-variance:
    y_pred['mix'] : (B, 10, K, 21) = [logit, mu (10), logvar (10)]
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import keras                                   # noqa: E402
from keras import layers, ops                  # noqa: E402

from executor_model import HolderPool, FirstTokens, HOLDER_CH    # noqa: E402

N_STEPS = 5


@keras.saving.register_keras_serializable(package='rl')
class TargetLogits(layers.Layer):
    """Logits over the five offensive slots for 'who receives the pass'; the holder's own slot is masked out."""

    def __init__(self, hidden=64, **kw):
        super().__init__(**kw)
        self.hidden = hidden
        self.d1 = layers.Dense(hidden, activation='gelu')
        self.d2 = layers.Dense(1)

    def call(self, enc, pooled, tok):
        off = enc[:, :5, :]
        h = ops.broadcast_to(pooled[:, None, :], ops.shape(off))
        z = self.d2(self.d1(ops.concatenate([off, h], axis=-1)))[..., 0]
        return z - 1e9 * tok[:, :5, HOLDER_CH]

    def get_config(self):
        return {**super().get_config(), 'hidden': self.hidden}


def build_model(f_dim, g_dim, k_modes=3, d=96, n_layers=3, heads=4, ff=192, dropout=0.05):
    tok = keras.Input((10, f_dim), name='tokens')
    ctx = keras.Input((g_dim,), name='ctx')
    x = layers.Dense(d)(tok)
    c = layers.Reshape((1, d))(layers.Dense(d)(ctx))
    h = layers.Concatenate(axis=1)([x, c])
    for _ in range(n_layers):
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads)(h, h)
        h = layers.LayerNormalization()(h + layers.Dropout(dropout)(a))
        f = layers.Dense(d)(layers.Dense(ff, activation='gelu')(h))
        h = layers.LayerNormalization()(h + layers.Dropout(dropout)(f))
    players = FirstTokens(10)(h)
    m = layers.Dense(ff, activation='gelu')(players)
    m = layers.Dense(k_modes * (1 + 4 * N_STEPS))(m)
    mix = layers.Reshape((10, k_modes, 1 + 4 * N_STEPS), name='mix')(m)
    pooled = HolderPool()(players, tok)
    dec = layers.Dense(3, activation='softmax', name='dec')(layers.Dense(64, activation='gelu')(pooled))
    tgt = layers.Softmax(name='tgt')(TargetLogits()(players, pooled, tok))
    return keras.Model([tok, ctx], {'mix': mix, 'dec': dec, 'tgt': tgt})


def split_mix(mix):
    """-> logit (B,10,K), mu (B,10,K,5,2), logvar (B,10,K,5,2)."""
    shp = ops.shape(mix)
    logit = mix[..., 0]
    mu = ops.reshape(mix[..., 1:1 + 2 * N_STEPS], (shp[0], shp[1], shp[2], N_STEPS, 2))
    lv = ops.clip(ops.reshape(mix[..., 1 + 2 * N_STEPS:], (shp[0], shp[1], shp[2], N_STEPS, 2)), -6.0, 3.0)
    return logit, mu, lv


def mix_nll(y_true, y_pred):
    """Per-sample mixture NLL, summed over the 10 steps*dims of a player's second, averaged over players and divided by 10 so it is
    comparable per dimension with the single-Gaussian loss of v1 (the constant 0.5 log 2 pi is omitted in both)."""
    logit, mu, lv = split_mix(y_pred)
    y = y_true[:, :, None, :, :]
    comp = -0.5 * ops.sum(lv + ops.square(y - mu) * ops.exp(-lv), axis=(-2, -1))      # (B,10,K)
    logp = ops.log_softmax(logit, axis=-1)
    ll = ops.logsumexp(logp + comp, axis=-1)                                           # (B,10)
    return -ops.mean(ll, axis=1) / (2.0 * N_STEPS)


def tgt_loss(y_true, y_pred):
    """Cross-entropy over teammate slots; y_true < 0 means 'no label' (weight it to zero via sample_weight)."""
    idx = ops.cast(ops.maximum(y_true, 0), 'int32')
    p = ops.take_along_axis(y_pred, ops.reshape(idx, (-1, 1)), axis=1)[:, 0]
    return -ops.log(ops.clip(p, 1e-9, 1.0))


CUSTOM_OBJECTS = {'HolderPool': HolderPool, 'FirstTokens': FirstTokens, 'TargetLogits': TargetLogits, 'mix_nll': mix_nll, 'tgt_loss': tgt_loss}
