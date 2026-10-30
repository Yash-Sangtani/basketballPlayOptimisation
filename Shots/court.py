"""
Court geometry and the one coordinate convention used for shot features.

RAW court coordinates (the SportVU convention, also used by ``Code/``): the court is
94 x 50 ft, x in [0, 94], y in [0, 50], baskets at (5.35, 25) and (88.65, 25).

``Game._get_moment_details`` returns y shifted by -50 (``PLOT_Y_OFFSET``) for matplotlib, whose
court is drawn below the x-axis. That shifted frame is for plotting only. Shot features must be
computed from raw coordinates, so every feature function asks for ``raw=True`` and takes its
baskets and court centre from here.

Why this lives in one file: in commit 5e8b9b4 the features were computed from plot-frame y while
the basket was at y=+25, which made every shot ~50 ft from the rim (the committed
``shot_features/01.02.2016-DEN-GSW.csv`` still has those numbers). Commit 3b4302b flipped the
basket to y=-25 in two files by hand. Do not repeat constants elsewhere.

Importing this module has no side effects.
"""

COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
LEFT_BASKET = (5.35, 25.0)
RIGHT_BASKET = (88.65, 25.0)
COURT_CENTER = (47.0, 25.0)
PLOT_Y_OFFSET = 50.0


def basket_for_x(x):
    """The basket on the same half of the court as ``x`` (list, raw coordinates)."""
    return list(LEFT_BASKET) if x < COURT_CENTER[0] else list(RIGHT_BASKET)


def plot_to_raw_y(y):
    """Undo the plot shift (works on floats and numpy arrays)."""
    return y + PLOT_Y_OFFSET


def assert_raw_y(y_values, where):
    """
    Guard against plot-frame y reaching code that assumes raw coordinates. In raw coordinates
    players are on the court (y in [0, 50]); in the plot frame they are in [-50, 0]. The median
    is used so one stray tracking point does not trip it.
    """
    values = [float(y) for y in y_values]
    values.sort()
    median = values[len(values) // 2] if values else 25.0
    if median < 0:
        raise ValueError(f"{where}: y coordinates look like the plot frame (median {median:.1f}, "
                         "expected raw court coordinates in [0, 50]). Get the moment with "
                         "Game._get_moment_details(frame, raw=True).")
