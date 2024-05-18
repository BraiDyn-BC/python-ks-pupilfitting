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

from collections import namedtuple as _namedtuple

import numpy as _np
import numpy.typing as _npt

from . import defaults as _defaults

HomogCoords = _npt.NDArray


class Points(_namedtuple('Points', ('x', 'y'))):
    @classmethod
    def from_homog_coords(cls, pts: HomogCoords):
        return cls(x=pts[0, :], y=pts[1, :])

    @property
    def size(self):
        return self.x.size

    def to_homog_coords(self) -> HomogCoords:
        pts = _np.empty((3, self.size), dtype=_defaults.DTYPE)
        pts[0, :] = self.x
        pts[1, :] = self.y
        pts[2, :] = 1
        return pts
