"""Behaviour model ("executor"): a transformer over the ten players that predicts, for EVERY player, where he will be over the next
1 s (five steps of 0.2 s, Gaussian with learned uncertainty) and, for the ball handler, his decision over the next 2 s
(shoot / pass / keep). Offence and defence use the same network; a role flag in each token says which side a player is on, and the
player's profile (profiles.py) is part of his token. Permutation-equivariant in the player slots (tested)."""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import keras                                   # noqa: E402
from keras import layers, ops                  # noqa: E402

HOLDER_CH = 13          # channel of the holder flag in the token (see executor_data.make_tokens)


@keras.saving.register_keras_serializable(package='rl')
class HolderPool(layers.Layer):
    """Sum of the offensive players' encodings weighted by the holder flag (zeros when no offensive player holds the ball)."""

    def call(self, enc, tok):
        w = tok[:, :5, HOLDER_CH:HOLDER_CH + 1]
        return ops.sum(w * enc[:, :5, :], axis=1)


@keras.saving.register_keras_serializable(package='rl')
class FirstTokens(layers.Layer):
    def __init__(self, n=10, **kw):
        super().__init__(**kw)
        self.n = n

    def call(self, x):
        return x[:, :self.n, :]

    def get_config(self):
        return {**super().get_config(), 'n': self.n}


@keras.saving.register_keras_serializable(package='rl')
class SplitClip(layers.Layer):
    def call(self, x):
        mu, lv = x[..., :2], ops.clip(x[..., 2:], -6.0, 3.0)
        return ops.concatenate([mu, lv], axis=-1)


def build_model(f_dim, g_dim, d=96, n_layers=3, heads=4, ff=192, n_fut=5, dropout=0.05):
    tok = keras.Input((10, f_dim), name='tokens')
    ctx = keras.Input((g_dim,), name='ctx')
    x = layers.Dense(d)(tok)
    c = layers.Reshape((1, d))(layers.Dense(d)(ctx))
    h = layers.Concatenate(axis=1)([x, c])                       # 10 player tokens + 1 context token
    for _ in range(n_layers):
        a = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads)(h, h)
        h = layers.LayerNormalization()(h + layers.Dropout(dropout)(a))
        f = layers.Dense(d)(layers.Dense(ff, activation='gelu')(h))
        h = layers.LayerNormalization()(h + layers.Dropout(dropout)(f))
    players = FirstTokens(10)(h)
    m = layers.Dense(ff, activation='gelu')(players)
    m = layers.Dense(n_fut * 4)(m)
    m = layers.Reshape((10, n_fut, 4))(m)
    move = SplitClip(name='move')(m)
    pooled = HolderPool()(players, tok)
    dec = layers.Dense(3, activation='softmax', name='dec')(layers.Dense(64, activation='gelu')(pooled))
    return keras.Model([tok, ctx], {'move': move, 'dec': dec})


def move_nll(y_true, y_pred):
    mu, lv = y_pred[..., :2], y_pred[..., 2:]
    return 0.5 * ops.mean(lv + ops.square(y_true - mu) * ops.exp(-lv), axis=(1, 2, 3))


CUSTOM_OBJECTS = {'HolderPool': HolderPool, 'SplitClip': SplitClip, 'FirstTokens': FirstTokens, 'move_nll': move_nll}
