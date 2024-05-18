# MIT License
#
# Copyright (c) 2024 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Union, Iterable, Generator
from pathlib import Path
import re as _re

import numpy as _np
import pandas as _pd
from tqdm import tqdm as _tqdm

from . import (
    defaults as _defaults,
    ellipse as _ellipse,
    fitting as _fitting,
)


PathLike = Union[str, Path]


def fit_hdf(
    hdffile: PathLike,
    likelihood_threshold: float = _defaults.LIKELIHOOD_THRESHOLD,
    min_valid_points: int = _defaults.MIN_VALID_POINTS,
    column_name_pattern: str = 'pupiledge',
    desc: str = 'fitting',
    verbose: bool = True,
) -> _pd.DataFrame:
    """frame-by-frame Ellipse fitting.

    Arguments
    ---------
    hdiffle: the DeepLabCut output HDF file path.
    likelihood_threshold: the threshold for the likelihood of each point to be used in fitting.
    column_name_pattern: the pattern of the column name to be used as the perimeter points of the ellipse.
    desc: the title of the progress bar. only used when `verbose` is set to True.
    verbose: allows verbose output of the processing.

    Returns
    --------
    table: each row corresponding to the Ellipse object, plus the number of points used to fit.
    """
    pts = _pd.read_hdf(str(hdffile), key='df_with_missing').droplevel(0, axis=1)  # remove the 'scorer' header

    # extract estimations of perimeter points as xp / yp / pval
    point_names = _extract_columns(pts.columns, column_name_pattern)
    T = pts.shape[0]
    N = len(point_names)
    xp = _np.empty((T, N), dtype=_defaults.DTYPE)
    yp = _np.empty((T, N), dtype=_defaults.DTYPE)
    pval = _np.empty((T, N), dtype=_defaults.DTYPE)
    for i, name in enumerate(point_names):
        xp[:, i] = pts[name, 'x'].values
        yp[:, i] = pts[name, 'y'].values
        pval[:, i] = pts[name, 'likelihood'].values
    valid = pval > likelihood_threshold
    nums  = _np.count_nonzero(valid, axis=1)
    frac  = nums / N

    # frame-by-frame fitting
    iter_frames = range(T)
    if verbose == True:
        iter_frames = _tqdm(iter_frames, total=T, desc=desc, mininterval=1, smoothing=0.02)
    data = {}
    for fld in _ellipse.Ellipse.data_fields():
        data[fld] = _np.empty((T,), dtype=_defaults.DTYPE)
        data[fld].fill(_np.nan)
    data['N'] = nums
    data['P'] = frac
    for i in iter_frames:
        if nums[i] < min_valid_points:
            continue
        xx = [float(x) for x in xp[i, valid[i]]]
        yy = [float(y) for y in yp[i, valid[i]]]
        el = _fitting.fit(xx, yy)
        for fld, val in el.data_items():
            data[fld][i] = val
    return _pd.DataFrame(data=data)


def _extract_columns(columns_orig, name_pattern: str) -> Iterable[str]:
    pat = _re.compile(name_pattern)
    ret = []
    prev = None
    for (name, axis) in columns_orig:
        if (prev is not None) and (prev == name):
            continue
        matches = pat.search(name)
        if matches:
            ret.append(name)
        prev = name
    return sorted(set(ret))


def iterate_dataframe(tab: _pd.DataFrame) -> Generator[_ellipse.Ellipse, None, None]:
    for _, row in tab.iterrows():
        row = row.to_dict()
        yield _ellipse.Ellipse(**dict((fld, row[fld]) for fld in _ellipse.Ellipse._fields))
