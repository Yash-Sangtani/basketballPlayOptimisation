"""Importing a Shots module must not do anything: no extraction, no training, no file reads or
writes, no TensorFlow, no forced matplotlib backend. Each module is imported in a fresh
subprocess so a module imported by an earlier test cannot hide a side effect."""
import subprocess
import sys
import textwrap

import pytest

from conftest import MODULES, SHOTS_DIR

PROBE = textwrap.dedent("""
    import os, sys
    before = set(os.listdir('.'))
    try:
        import {module}
    except ModuleNotFoundError as exc:
        print('MISSING_DEPENDENCY', exc.name)
        sys.exit(3)
    import matplotlib
    after = set(os.listdir('.'))
    print('new_files_in_cwd=', sorted(after - before))
    print('tensorflow_imported=', 'tensorflow' in sys.modules)
    print('seaborn_imported=', 'seaborn' in sys.modules)
    print('backend=', matplotlib.get_backend().lower())
""")


@pytest.mark.parametrize('module', MODULES)
def test_import_has_no_side_effects(module, tmp_path, shots_env):
    scratch = tmp_path / 'cwd'
    scratch.mkdir()
    result = subprocess.run([sys.executable, '-c', PROBE.format(module=module)], cwd=scratch,
                            env=shots_env, capture_output=True, text=True, timeout=300)
    if result.returncode == 3:
        pytest.skip(f'{module}: third-party dependency not installed: {result.stdout.strip()}')
    assert result.returncode == 0, f'import {module} failed:\n{result.stderr[-1500:]}'
    out = result.stdout
    assert 'new_files_in_cwd= []' in out, out          # no temp/ folder, labels.txt, ...
    assert 'tensorflow_imported= False' in out, out
    assert 'seaborn_imported= False' in out, out
    assert 'backend= agg' in out, out                    # MPLBACKEND honoured, not forced to TkAgg
    # nothing written to the (nonexistent) output/model folders either
    assert not (tmp_path / 'out').exists()
    assert not (tmp_path / 'models').exists()


def test_no_windows_paths_or_forced_backend_in_source():
    offenders = []
    for path in sorted(SHOTS_DIR.glob('*.py')):
        text = path.read_text()
        for needle in ('C:/Users', 'C:\\\\Users', 'matplotlib.use('):
            if needle in text:
                offenders.append(f'{path.name}: {needle}')
    assert not offenders, offenders


def test_extraction_only_starts_from_main():
    """The extraction call is behind a __main__ guard (it used to run on import)."""
    source = (SHOTS_DIR / 'shot_tracking.py').read_text()
    last = source.rstrip().splitlines()[-2:]
    assert last == ["if __name__ == '__main__':", '    write_features(extract_games())'], last


def test_missing_models_raise_a_clear_error_before_importing_tensorflow(tmp_path, shots_env):
    code = textwrap.dedent("""
        import sys
        import shot_tracking_plot
        for fn in (shot_tracking_plot.load_model_and_predict, shot_tracking_plot.predict_output):
            try:
                fn([0.0] * 18)
            except FileNotFoundError as exc:
                print('ERROR', str(exc).replace(chr(10), ' | '))
        print('tensorflow_imported=', 'tensorflow' in sys.modules)
    """)
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=shots_env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-1500:]
    out = result.stdout
    assert 'tensorflow_imported= False' in out, out
    for name in ('shot_model_v2.keras', 'shot_scaler_mean.npy', 'shot_scaler_scale.npy',
                 'zone_model_v2.keras', 'zone_scaler_mean.npy', 'encoder_shot_zone.pkl',
                 'encoder_shot_area.pkl', 'encoder_shot_range.pkl'):
        assert name in out, (name, out)
    assert 'SHOTS_MODEL_DIR' in out
