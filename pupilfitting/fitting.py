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

from typing import Iterable
import math as _math

import numpy as _np
import scipy.optimize as _sopts

from . import defaults as _defaults
from .ellipse import (
    EllipseParams as _EllipseParams,
    Ellipse as _Ellipse,
    ellipse_to_circle_matrix as _ellipse_to_circle,
)


Ub = _np.array([[1, 0, 0], [0, 1, 0]])  # projection: homogeneous coords to normal 2-d coords


def fit(xp: Iterable[float], yp: Iterable[float]) -> _Ellipse:
    """xp and yp must not contain NaNs."""
    try:
        res = _sopts.least_squares(compute_loss, estimate_x0(xp=xp, yp=yp), kwargs=dict(xp=xp, yp=yp))
        if res['success'] == True:
            return _Ellipse.from_params(res['x'])
        else:
            return _Ellipse.NA()
    except OverflowError:
        pass
    return _Ellipse.NA()


def estimate_x0(
    xp: Iterable[float] = None,
    yp: Iterable[float] = None,
) -> _EllipseParams:
    """xp and yp must not contain NaNs."""
    rx0 = (max(xp) - min(xp)) / 2
    ry0 = (max(yp) - min(yp)) / 2
    a0 = -_math.log(rx0)
    b0 = -_math.log(ry0)
    cx0 = _np.mean(xp)
    cy0 = _np.mean(yp)
    return (a0, b0, 0, cx0, cy0)


def compute_loss(
    params: _EllipseParams,
    xp: Iterable[float] = None,
    yp: Iterable[float] = None,
):
    """xp and yp must not contain NaNs."""
    ones = _np.ones((len(xp),), dtype=_defaults.DTYPE)
    pts = _np.stack([xp, yp, ones], axis=0)
    M = _ellipse_to_circle(params)
    pts2 = Ub @ M @ pts
    return ((pts2 ** 2).sum(axis=0) - 1)
