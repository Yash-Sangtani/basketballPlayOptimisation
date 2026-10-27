"""Fast tests on tiny synthetic data. Run from Code/:  ../.venv/bin/python -m pytest -q"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

import evaluation as ev
import synthetic as syn

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = sys.executable


@pytest.fixture(scope='module')
def arrays():
    return syn.make_arrays(n_games=30, n_rows=150, seed=1)


# ---------------------------------------------------------------- 1. no game on both sides of any split
def test_every_recorded_split_is_game_disjoint(arrays):
    X, y, g, feats = arrays
    splits = ev.outer_folds(g, 5, seed=3)
    res = ev.evaluate_splits(X, y, g, splits, ['simple', 'logistic'], feats, seed=3)
    assert len(res.split_log) == 5 * 2                         #one outer + one calibration record per fold
    for r in res.split_log:                                    #re-checked here, independently of verify_split_log
        assert not (r.train & r.held), f"fold {r.fold} {r.kind}: shared games {sorted(r.train & r.held)[:3]}"
    for k in range(5):
        outer = next(r for r in res.split_log if r.fold == k and r.kind == 'outer')
        for r in res.split_log:
            if r.fold == k and r.kind != 'outer':
                assert not ((r.train | r.held) & outer.held), "an inner split touched the outer-test games"
    #every game is scored exactly once, and only by the model that did not train on it
    assert len(res.scored) == len(y) and len(np.unique(res.scored)) == len(y)
    assert set(np.unique(g[res.scored])) == set(np.unique(g))


def test_row_level_split_is_rejected(arrays):
    """A plain row-level KFold puts the same game on both sides; the evaluation must refuse it."""
    X, y, g, feats = arrays
    bad = list(KFold(5, shuffle=True, random_state=0).split(X))
    with pytest.raises(AssertionError, match='both sides'):
        ev.evaluate_splits(X, y, g, bad, ['logistic'], feats)


def test_verify_split_log_catches_tampering():
    ok = [ev.SplitRecord(0, 'outer', frozenset('abcd'), frozenset('ef')),
          ev.SplitRecord(0, 'cal', frozenset('abc'), frozenset('d'))]
    ev.verify_split_log(ok)
    leak_test_into_train = [ok[0]._replace(train=frozenset('abcde')), ok[1]]
    with pytest.raises(AssertionError):
        ev.verify_split_log(leak_test_into_train)
    cal_uses_test_game = [ok[0], ok[1]._replace(held=frozenset('de'))]
    with pytest.raises(AssertionError, match='outer-test'):
        ev.verify_split_log(cal_uses_test_game)
    val_overlaps_cal = ok + [ev.SplitRecord(0, 'val', frozenset('ab'), frozenset('c')),
                             ev.SplitRecord(0, 'val', frozenset('a'), frozenset('d'))]
    with pytest.raises(AssertionError, match='calibration'):
        ev.verify_split_log(val_overlaps_cal)


def test_split_by_game_properties(arrays):
    _, _, g, _ = arrays
    idx = np.arange(0, len(g), 2)
    a, b = ev.split_by_game(idx, g, 0.25, seed=5)
    assert set(a) | set(b) == set(idx) and not (set(a) & set(b))
    assert not (set(g[a]) & set(g[b]))
    n_games = len(np.unique(g[idx]))
    assert len(np.unique(g[b])) == round(0.25 * n_games)
    a2, b2 = ev.split_by_game(idx, g, 0.25, seed=5)
    assert np.array_equal(b, b2)                                #deterministic


# ---------------------------------------------------------------- 2. bootstrap resamples whole games
def test_bootstrap_resamples_whole_games(arrays):
    _, y, g, _ = arrays
    boot = ev.GameBootstrap(g, n_boot=25, seed=7)
    rows_per_game = pd.Series(g).value_counts()
    for b in range(boot.n_boot):
        rows = boot.resample_rows(b)
        counts = pd.Series(g[rows]).value_counts()
        n_drawn = boot.game_counts(b)
        assert n_drawn.sum() == len(boot.games)                  #as many games as the original
        for game, c in counts.items():                           #rows of a game appear k * (its size) times
            k = n_drawn[np.searchsorted(boot.games, game)]
            assert c == k * rows_per_game[game]
        w = boot.row_weights(b)
        for j, game in enumerate(boot.games):                    #weight constant within each game
            assert len(np.unique(w[boot.codes == j])) == 1
        assert w.sum() == len(rows)
    assert any((boot.game_counts(b) == 0).any() for b in range(boot.n_boot))     #some games are left out
    assert np.array_equal(ev.GameBootstrap(g, 25, 7).draws, boot.draws)           #seeded


def test_weighted_metrics_match_sklearn_on_explicit_resample(arrays):
    _, y, g, _ = arrays
    rng = np.random.default_rng(0)
    s = rng.normal(size=len(y)) + 1.5 * y
    s_ties = np.round(s, 0)                                      #heavy ties, like an isotonic output
    boot = ev.GameBootstrap(g, n_boot=6, seed=2)
    for score in (s, s_ties):
        rs = ev.RankedScores(y, score)
        auc, ap = rs.curves()
        assert auc == pytest.approx(roc_auc_score(y, score), abs=1e-12)
        assert ap == pytest.approx(average_precision_score(y, score), abs=1e-12)
        for b in range(boot.n_boot):
            rows = boot.resample_rows(b)                         #explicit concatenated resample
            auc_w, ap_w = rs.curves(boot.row_weights(b))
            assert auc_w == pytest.approx(roc_auc_score(y[rows], score[rows]), abs=1e-10)
            assert ap_w == pytest.approx(average_precision_score(y[rows], score[rows]), abs=1e-10)
    assert ev.weighted_brier(y, np.full(len(y), y.mean())) == pytest.approx(y.mean() * (1 - y.mean()))


def test_bootstrap_ci_covers_estimate_and_noskill_is_prevalence(arrays):
    _, y, g, _ = arrays
    rng = np.random.default_rng(1)
    s = rng.normal(size=len(y)) + 1.0 * y
    boot = ev.GameBootstrap(g, n_boot=200, seed=4)
    bm = ev.bootstrap_metrics(y, {'m': s, 'const': np.zeros(len(y))}, boot)
    lo, hi = ev.ci(bm['m']['roc'])
    assert lo < roc_auc_score(y, s) < hi
    assert np.nanmean(bm['const']['roc']) == pytest.approx(0.5)
    assert ev.RankedScores(y, np.zeros(len(y))).curves()[1] == pytest.approx(y.mean())


# ---------------------------------------------------------------- 3. --split test refuses without confirmation
def _run_cli(*args):
    return subprocess.run([PY, os.path.join(CODE_DIR, 'pass_model.py'), *args], capture_output=True, text=True,
                          cwd=CODE_DIR, timeout=300)


def test_split_test_refuses_without_confirmation(tmp_path):
    missing = str(tmp_path / 'does_not_exist')
    r = _run_cli('--split', 'test', '--pickles', missing)
    assert r.returncode != 0
    assert 'REFUSING' in r.stderr and 'FROZEN' in r.stderr
    assert 'loading' not in r.stdout and 'No such file' not in r.stderr    #it stopped before touching any data


def test_split_test_with_confirmation_warns_and_scores_test_games_only(tmp_path):
    dev, test = syn.game_names(12, '02'), syn.game_names(4, '03')
    syn.write_pickle_dir(str(tmp_path / 'pk'), dev + test, seed=5, n_rows=150)
    (tmp_path / 'dev.txt').write_text('\n'.join(dev) + '\n'); (tmp_path / 'test.txt').write_text('\n'.join(test) + '\n')
    out = tmp_path / 'out'
    r = _run_cli('--split', 'test', '--confirm-frozen-test', '--pickles', str(tmp_path / 'pk'),
                 '--dev-list', str(tmp_path / 'dev.txt'), '--test-list', str(tmp_path / 'test.txt'),
                 '--models', 'logistic', '--n-boot', '20', '--out', str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'WARNING' in r.stderr and 'FROZEN TEST GAMES' in r.stderr
    scored_games = set(np.load(out / 'oof.npz')['game'])
    assert scored_games == set(test)                                       #only test games are scored


def test_dev_mode_never_opens_test_pickles(tmp_path):
    dev, test = syn.game_names(10, '02'), syn.game_names(3, '03')
    syn.write_pickle_dir(str(tmp_path / 'pk'), dev + test, seed=6, n_rows=150)
    for gname in test:                                                      #corrupt every test pickle: any read would crash
        for sub in ('.', 'meta'):
            (tmp_path / 'pk' / sub / ev.game_to_pickle_name(gname)).write_bytes(b'not a pickle')
    ds = ev.load_games(str(tmp_path / 'pk'), dev)
    assert set(ds.games) == set(dev)


# ---------------------------------------------------------------- 4. isotonic calibration uses calibration games only
def test_calibrator_never_sees_test_labels(arrays):
    X, y, g, feats = arrays
    tr, te = next(iter(ev.outer_folds(g, 5, seed=0)))
    run = lambda yy: ev.evaluate_splits(X, yy, g, [(tr, te)], ['logistic', 'gboost'], feats, seed=0)
    base = run(y.astype(float))
    #1. Test labels destroyed (NaN) or flipped: predictions must be bit-identical.
    y_nan = y.astype(float); y_nan[te] = np.nan
    y_flip = y.astype(float); y_flip[te] = 1 - y_flip[te]
    for alt in (run(y_nan), run(y_flip)):
        for m in ('logistic', 'gboost'):
            assert np.array_equal(alt.raw[m], base.raw[m])
            assert np.array_equal(alt.cal[m], base.cal[m])
    #2. Not vacuous: flipping labels of the calibration games only leaves the raw model alone but moves the calibrated output.
    fit_idx, cal_idx = ev.split_by_game(tr, g, 0.25, ev.inner_seed(0, 0))
    y_cal = y.astype(float); y_cal[cal_idx] = 1 - y_cal[cal_idx]
    alt = run(y_cal)
    for m in ('logistic', 'gboost'):
        assert np.array_equal(alt.raw[m], base.raw[m])
        assert not np.allclose(alt.cal[m], base.cal[m])
    #3. The fit rows (outer-train minus calibration games) do change the raw model.
    y_fit = y.astype(float); y_fit[fit_idx] = 1 - y_fit[fit_idx]
    assert not np.allclose(run(y_fit).raw['logistic'], base.raw['logistic'])


def test_calibration_improves_balanced_logistic_brier(arrays):
    X, y, g, feats = arrays
    res = ev.evaluate_splits(X, y, g, ev.outer_folds(g, 5, 0), ['logistic'], feats, seed=0)
    yy = y[res.scored].astype(float)
    assert ev.weighted_brier(yy, res.cal['logistic']) < ev.weighted_brier(yy, res.raw['logistic'])   #balanced weights inflate raw P
    assert res.raw['logistic'].mean() > 2 * yy.mean()
    assert res.cal['logistic'].min() >= ev.PROB_EPS and res.cal['logistic'].max() <= 1 - ev.PROB_EPS


def test_reliability_table_counts_and_rates():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = ev.reliability_table(y, p)
    assert t['n'].sum() == 1000 and len(t) == 10 and t['failures'].sum() == y.sum()
    assert (t['observed'] - t['mean_pred']).abs().max() < 0.12
    ties = ev.reliability_table(y, np.round(p * 4) / 4)          #5 distinct scores: fewer bins, never an error
    assert ties['n'].sum() == 1000 and len(ties) < 10


# ---------------------------------------------------------------- loader
def test_loader_whitelist_skips_and_checks(tmp_path):
    games = syn.game_names(6, '04')
    d = str(tmp_path / 'pk')
    syn.write_pickle_dir(d, games, seed=2)
    os.remove(os.path.join(d, 'meta', ev.game_to_pickle_name(games[1])))       #no meta
    os.remove(os.path.join(d, ev.game_to_pickle_name(games[2])))               #no pickle
    st = open(os.path.join(d, '_status.csv')).read().replace(f'{games[3]},ok', f'{games[3]},failed')
    open(os.path.join(d, '_status.csv'), 'w').write(st)
    ds = ev.load_games(d, games)
    assert set(ds.games) == {games[0], games[4], games[5]}
    assert ds.info['skipped'] == {'no_pickle': [games[2]], 'no_meta': [games[1]], 'status_not_ok': [games[3]]}
    assert ds.info['versions'] == {'abc123': 3} and ds.info['label_outcome_mismatch'] == 0
    assert ds.features == ev.FEATURES and ds.X.shape[1] == 15                  #whitelist only
    assert 'outcome' not in ds.features and 'target_inferred' not in ds.features and ev.LABEL not in ds.features
    with pytest.raises(FileNotFoundError):
        ev.load_games(str(tmp_path / 'nowhere'), games)


def test_loader_rejects_misaligned_meta(tmp_path):
    games = syn.game_names(2, '05')
    d = str(tmp_path / 'pk'); syn.write_pickle_dir(d, games)
    mp = os.path.join(d, 'meta', ev.game_to_pickle_name(games[0]))
    meta = pd.read_pickle(mp); meta['game'] = games[1]
    meta.to_pickle(mp)
    with pytest.raises(ValueError, match='meta.game'):
        ev.load_games(d, games)


def test_lists_must_be_disjoint():
    ev.check_lists_disjoint(['a'], ['b'])
    with pytest.raises(AssertionError):
        ev.check_lists_disjoint(['a', 'b'], ['b'])


def test_real_split_files_are_disjoint_and_complete():
    dev = ev.read_game_list(os.path.join(ev.SPLIT_DIR, 'dev_games.txt'))
    test = ev.read_game_list(os.path.join(ev.SPLIT_DIR, 'test_games.txt'))
    ev.check_lists_disjoint(dev, test)
    assert len(dev) == 501 and len(test) == 130 and len(set(dev)) == 501


def test_population_frontcourt_drops_constant_column(tmp_path):
    games = syn.game_names(4, '06')
    d = str(tmp_path / 'pk'); syn.write_pickle_dir(d, games)
    ds = ev.load_games(d, games)
    fc = ev.restrict_population(ds, 'frontcourt')
    assert 'backcourt' not in fc.features and fc.X.shape[1] == 14 and len(fc.y) < len(ds.y)


# ---------------------------------------------------------------- MLP (skipped if TensorFlow/Keras is unavailable)
def test_mlp_is_seeded_and_validates_by_game(arrays):
    pytest.importorskip('keras')
    X, y, g, feats = arrays
    tr, te = next(iter(ev.outer_folds(g, 5, seed=0)))
    runs = [ev.evaluate_splits(X, y, g, [(tr, te)], ['mlp'], feats, seed=11) for _ in range(2)]
    a, b = runs
    assert np.allclose(a.raw['mlp'], b.raw['mlp'], atol=1e-6), "same seed must give the same MLP predictions"
    val = [r for r in a.split_log if r.kind == 'val']
    assert len(val) == 1 and val[0].train and val[0].held
    assert a.notes['mlp'][0]['best_epoch'] >= 1
    assert ((a.raw['mlp'] > 0) & (a.raw['mlp'] < 1)).all()


# ---------------------------------------------------------------- date-blocked folds and failure-type table
def test_date_block_folds_are_game_disjoint_and_ordered():
    games = [f"{m:02d}.{d:02d}.2015.AAA.at.B{m}{d}" for m in (11, 12) for d in (3, 9, 20)] + ["01.02.2016.AAA.at.CCC"]
    g = np.repeat(np.array(games), 5)
    folds = ev.date_block_folds(g, 3)
    assert len(folds) == 3
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(g)))                        #every row tested exactly once
    last_end = None
    for tr, te in folds:
        ev.assert_game_disjoint(g, tr, te)
        dates = sorted(ev.game_date(x) for x in set(g[te]))
        if last_end is not None:
            assert dates[0] >= last_end                                   #blocks follow each other in time
        last_end = dates[-1]
    X, y, gg, feats = syn.make_arrays(n_games=30, n_rows=150, seed=2)
    res = ev.evaluate_splits(X, y, gg, ev.date_block_folds(gg, 5), ['logistic'], feats, seed=0)
    assert len(res.scored) == 30 * 150


def test_by_outcome_table():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    outcome = np.array(['completed'] * 4 + ['intercepted', 'intercepted', 'out_of_bounds', 'out_of_bounds'])
    score = np.array([.1, .2, .3, .4, .9, .8, .35, .15])
    t = ev.by_outcome_table(y, outcome, {'m': score})
    assert list(t['failure_type']) == ['intercepted', 'out_of_bounds'] and list(t['n']) == [2, 2]
    assert t.loc[0, 'm ROC'] == 1.0
    assert t.loc[1, 'm ROC'] == pytest.approx(roc_auc_score([0, 0, 0, 0, 1, 1], [.1, .2, .3, .4, .35, .15]))


def test_restrict_features_passer_only(tmp_path):
    games = syn.game_names(3, '07')
    d = str(tmp_path / 'pk'); syn.write_pickle_dir(d, games)
    ds = ev.load_games(d, games)
    po = ev.restrict_features(ds, ev.PASSER_FEATURES)
    assert po.features == ev.PASSER_FEATURES and po.X.shape == (len(ds.y), 5)
    assert np.array_equal(po.X[:, 0], ds.X[:, ds.features.index('shot_clock')], equal_nan=True)
    assert not set(ev.PASSER_FEATURES) & {'dist', 'ball_end_basket_dist', 'target_speed', 'lane_n_def_5ft',
                                          'closest_def_dist_ball_end', 'closest_def_trajectory'}
