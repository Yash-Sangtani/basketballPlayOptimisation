import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'difficulty'))
import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'rl')
pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason='needs data/rl')


@pytest.fixture(scope='module')
def stack():
    import league as L
    from load import load_games
    from profiles import build_profiles
    from splits import games as split_games
    from train_executor import fit_val_games
    from engine_validate import quarter_starts
    st = L.load_stack()
    fit_g, _ = fit_val_games()
    table = build_profiles(load_games(DATA, fit_g[:20]))
    G = load_games(DATA, split_games('holdout')[:4])
    qs = quarter_starts(G)[:6]
    return st, G, table, [(g, k) for g, k, _ in qs]


def test_defence_agent_sets_schemes_and_returns(stack):
    import league as L
    st, G, table, starts = stack
    o, d = L.make_agents(st, learn=True, seed=0)
    eng = L.run_match(st, G, table, starts, 1, 160, A=(None, d), B=(None, d))
    assert (eng.scheme >= -1).all() and (eng.scheme < st['lib'].K).all()
    assert len(eng.poss_log) > 0 and d.buf
    out = d.update_from(eng)
    assert out and np.isfinite(out['loss'])


def test_offence_agent_acts_and_learns(stack):
    import league as L
    st, G, table, starts = stack
    o, d = L.make_agents(st, learn=True, seed=0)
    eng = L.run_match(st, G, table, starts, 2, 160, A=(o, None), B=(None, None))
    out = o.update_from(eng)
    assert out == {} or np.isfinite(out['loss'])
    m = L.match_metrics(eng)
    assert m['n_poss'] > 0


def test_zero_residual_agents_are_valid_policies(stack):
    import league as L
    st, G, table, starts = stack
    o, d = L.make_agents(st, learn=False, seed=0)
    F = np.random.default_rng(0).normal(size=(5, d.D)).astype(np.float32)
    p = d.probs(F)
    assert np.allclose(p.sum(1), 1) and np.allclose(p, np.exp(d.prior_logp(F)), atol=1e-5)
