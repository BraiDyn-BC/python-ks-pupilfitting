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
from collections import namedtuple as _namedtuple
import math as _math

import numpy as _np
import numpy.typing as _npt

from . import defaults as _defaults
from .points import Points as _Points

EllipseParams = Iterable[float]
AffineTransform = _npt.NDArray


class Ellipse(_namedtuple('Ellipse', ('A', 'B', 'phi', 'cx', 'cy'))):
    @classmethod
    def NA(cls):
        return cls(_math.nan, _math.nan, _math.nan, _math.nan, _math.nan)

    @classmethod
    def from_params(cls, params: EllipseParams):
        A = _math.exp(-float(params[0]))
        B = _math.exp(-float(params[1]))
        phi, cx, cy = params[2:]
        return cls(A, B, float(phi), float(cx), float(cy))

    @classmethod
    def data_fields(cls):
        return ('A', 'B', 'D', 'phi', 'cx', 'cy')

    def data_values(self):
        return (self.A, self.B, max(self.A, self.B) * 2, self.phi, self.cx, self.cy)

    def data_items(self):
        return zip(self.data_fields(), self.data_values())

    def is_success(self):
        return all(~_np.isnan(item) for item in self)

    def to_params(self) -> EllipseParams:
        a = -_math.log(self.A)
        b = -_math.log(self.B)
        return _np.array([a, b, self.phi, self.cx, self.cy], dtype=_defaults.DTYPE)

    def perimeter_points(
        self,
        num_points: int = _defaults.NUM_PERIMETER,
    ) -> _Points:
        return perimeter_points(self.to_params(), num_points=num_points)


def ellipse_to_circle_matrix(params: EllipseParams) -> AffineTransform:
    a = _math.exp(params[0])
    b = _math.exp(params[1])
    phi, cx, cy = params[2:]
    cos = _math.cos(phi)
    sin = _math.sin(phi)
    return _np.array([
        [a * cos, a * sin, -a * (cx * cos + cy * sin)],
        [-b * sin, b * cos, b * (cx * sin - cy * cos)],
        [0, 0, 1]
    ])


def circle_to_ellipse_matrix(params: EllipseParams) -> AffineTransform:
    A = _math.exp(-params[0])
    B = _math.exp(-params[1])
    phi, cx, cy = params[2:]
    cos = _math.cos(phi)
    sin = _math.sin(phi)
    return _np.array([
        [A * cos, -B * sin, cx],
        [A * sin, B * cos, cy],
        [0, 0, 1]
    ])


def perimeter_points(
    params: EllipseParams,
    num_points: int = _defaults.NUM_PERIMETER,
) -> _Points:
    unit = _unit_circle_points(num_points)
    trans = circle_to_ellipse_matrix(params)
    pts = trans @ unit.to_homog_coords()
    return _Points.from_homog_coords(pts)


def _unit_circle_points(num_points: int = _defaults.NUM_PERIMETER) -> _Points:
    angles = _np.linspace(0, _math.pi * 2, num_points, endpoint=False)
    return _Points(_np.cos(angles), _np.sin(angles))
