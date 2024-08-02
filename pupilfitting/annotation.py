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

from typing import Union, Tuple, Optional
from pathlib import Path
from collections import namedtuple as _namedtuple
import subprocess as _sp
import warnings as _warnings
import json as _json

import numpy as _np
import numpy.typing as _npt
import scipy.ndimage as _ndimage
import matplotlib.colors as _mcolors
import cv2 as _cv2
import pandas as _pd
from tqdm import tqdm as _tqdm

try:
    from skvideo import io as _vio

    # FIXME: a dirty hack to use numpy from skvideo
    _np.float = _np.float32
    _np.int   = _np.int64

except ImportError:
    _warnings.warn("skvideo not detected: install it via `pip install scikit-video` to use annotate_on_video()")
    _vio = None

from . import (
    ellipse as _ellipse,
    dlc as _dlc,
)

PathLike = Union[str, Path]
Number = Union[int, float]

ColorSpec = Union[str, Tuple[Number]]
Color256 = _npt.NDArray[_np.uint8]

ImageLike = _npt.NDArray
ImageMask = _npt.NDArray[bool]
Image256  = _npt.NDArray[_np.uint8]


def as_color256(color: ColorSpec) -> Color256:
    if isinstance(color, str):
        color = _mcolors.to_rgb(color)
    if not hasattr(color, '__iter__') and (color <= 1):
        color = (color, color, color)
    color = _np.array(color, dtype=_np.float32)
    if color.max() <= 1:
        color = color * 255
    return _np.clip(color, 0, 255).astype(_np.uint8)


class Grid(_namedtuple('Grid', ('XYI',))):
    DTYPE = _np.float32

    @property
    def X(self) -> _npt.NDArray[DTYPE]:
        return self.XYI[0]

    @property
    def Y(self) -> _npt.NDArray[DTYPE]:
        return self.XYI[1]

    @property
    def XY(self) -> _npt.NDArray[DTYPE]:
        return self.XYI[:2]

    @classmethod
    def mesh(
        cls,
        height: Optional[int] = None,
        width: Optional[int] = None,
        factor: int = 1,
    ):
        width, height = _validate_size(width=width, height=height)
        X, Y = _np.meshgrid(_np.arange(width * factor, dtype=cls.DTYPE) / factor,
                            _np.arange(height * factor, dtype=cls.DTYPE) / factor,
                            indexing='xy')
        ones = _np.ones((height * factor, width * factor), dtype=cls.DTYPE)
        return cls(XYI=_np.stack([X, Y, ones], axis=0))


def _validate_size(width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int]:
    if width is None:
        if height is None:
            raise ValueError('specify at least one size')
        else:
            height = int(height)
        width = height
    else:
        width = int(width)
        if height is None:
            height = width
        else:
            height = int(height)
    return width, height


def ellipse_to_mask(
    ellipse: _ellipse.Ellipse,
    grid: Optional[Grid] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> ImageMask:
    if grid is None:
        grid = Grid.mesh(width=width, height=height)
    M = _ellipse.ellipse_to_circle_matrix(ellipse.to_params())
    U = _np.tensordot(M, grid.XYI, axes=((1,), (0,)))
    U = U[:2]  # discard the array of ones at the bottom
    dev = (U ** 2).sum(axis=0) - 1
    return (dev <= 0)


def point_to_mask(
    x: float,
    y: float,
    diameter: float = 5.0,
    grid: Optional[Grid] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> ImageMask:
    if grid is None:
        grid = Grid.mesh(width=width, height=height)
    rad = diameter / 2
    ref = _np.array([x, y], dtype=_np.float32).reshape((2, 1, 1))
    dXY = grid.XY - ref
    return ((dXY ** 2).sum(axis=0) <= rad)


def take_border(mask: ImageMask, width: int = 1) -> ImageMask:
    outer = width // 2
    eroded = _ndimage.binary_erosion(mask, iterations=(width - outer))
    if outer > 0:
        dilated = _ndimage.binary_dilation(mask, iterations=outer)
    else:
        dilated = mask
    return _np.logical_xor(eroded, dilated)


def color_mask(
    mask: ImageLike,
    color: ColorSpec,
    alpha: float = 1.0,
    dtype: _npt.DTypeLike = _np.float32,
) -> ImageLike:
    color = as_color256(color)
    H, W = mask.shape
    return mask.reshape((H, W, 1)).astype(dtype) * (alpha * color.reshape((1, 1, 3)).astype(dtype))


def image_as_uint8(img: ImageLike) -> Image256:
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(_np.uint8)


def overlay_images(*imgs) -> Image256:
    out = None
    for img in imgs:
        if out is None:
            out = img.astype(_np.float32)
        else:
            out += img.astype(_np.float32)
    return image_as_uint8(out)


def annotate_frame(
    frame: ImageLike,
    ellipse: _ellipse.Ellipse,
    centersize: Optional[float] = 5,
    centercolor: Optional[ColorSpec] = 'w',
    centeralpha: float = 1.0,
    facecolor: Optional[ColorSpec] = 'c',
    facealpha: float = 0.3,
    borderwidth: Optional[int] = 2,
    bordercolor: Optional[ColorSpec] = 'w',
    borderalpha: float = 1.0,
    grid: Optional[Grid] = None,
    grid_upsample: int = 1,
) -> Image256:
    draw_center = (centersize is not None) and (centersize > 0) and (centercolor is not None) and (centeralpha > 0)
    draw_face = (facecolor is not None) and (facealpha > 0)
    draw_border = (borderwidth is not None) and (borderwidth > 0) and (bordercolor is not None) and (borderalpha > 0)
    if not any([draw_center, draw_face, draw_border]):
        return frame

    H, W = frame.shape[:2]
    if grid is None:
        grid = Grid.mesh(width=W, height=H, factor=grid_upsample)
    mask = ellipse_to_mask(ellipse, grid=grid)

    items = []
    items.append(frame)
    if draw_center:
        center = point_to_mask(ellipse.cx, ellipse.cy, grid=grid, diameter=centersize * grid_upsample)
        center = _cv2.resize(center.astype(_np.float32), dsize=(W, H))
        items.append(color_mask(center, color=centercolor, alpha=centeralpha))
    if draw_border:
        border = take_border(mask, borderwidth * grid_upsample)
        border = _cv2.resize(border.astype(_np.float32), dsize=(W, H))
        items.append(color_mask(border, color=bordercolor, alpha=borderalpha))
    if draw_face:
        face = _cv2.resize(mask.astype(_np.float32), dsize=(W, H))
        items.append(color_mask(face, color=facecolor, alpha=facealpha))
    return overlay_images(*items)


if _vio is None:
    def annotate_on_video(
        srcfile: PathLike,
        dstfile: PathLike,
        annotation: _pd.DataFrame,
        centercolor: Optional[ColorSpec] = 'w',
        facecolor: Optional[ColorSpec] = 'c',
        bordercolor: Optional[ColorSpec] = None,
        framerate: Optional[float] = None,
        verbose: bool = True
    ):
        raise NotImplementedError("install `scikit-video` to enable annotate_on_video()")

else:
    def _infer_frame_rate(videofile: Path) -> str:
        ret = _sp.run(['ffprobe', '-hide_banner', '-v', 'error',
                       '-i', str(videofile.resolve()), '-print_format', 'json',
                       '-show_streams', '-select_streams', 'v:0'],
                      capture_output=True, check=True)
        metadata = _json.loads(ret.stdout.decode())
        return metadata['streams'][0]['avg_frame_rate']

    def annotate_on_video(
        srcfile: PathLike,
        dstfile: PathLike,
        annotation: _pd.DataFrame,
        centercolor: Optional[ColorSpec] = 'w',
        facecolor: Optional[ColorSpec] = 'c',
        bordercolor: Optional[ColorSpec] = None,
        framerate: Optional[float] = None,
        framerange: Optional[Tuple[int]] = None,
        desc: str = 'annotating',
        verbose: bool = True
    ):
        srcfile = Path(srcfile)
        dstfile = Path(dstfile)
        if not srcfile.exists():
            raise FileNotFoundError(str(srcfile))
        if not dstfile.parent.exists():
            dstfile.parent.mkdir(parents=True)
        num_frames = annotation.shape[0]
        if framerange is None:
            framerange = (0, num_frames)
        frame_start = min(framerange)
        frame_stop  = max(framerange)

        if framerate is None:
            framerate = _infer_frame_rate(srcfile)
        indict = {'-r': framerate}
        outdict = {'-r': framerate, '-q:v': '10'}
        grid = None
        factor = 5  # NOTE: the upsampling factor for `grid`. fixed constant for the time being

        with _vio.FFmpegReader(str(srcfile)) as src:
            with _vio.FFmpegWriter(
                str(dstfile),
                inputdict=indict,
                outputdict=outdict
            ) as out:
                iteration = zip(range(num_frames), src.nextFrame(), _dlc.iterate_dataframe(annotation))
                if verbose == True:
                    iteration = _tqdm(iteration, desc=desc, total=annotation.shape[0], mininterval=1, smoothing=0.01)

                for i, frame, el in iteration:
                    if i < frame_start:
                        continue
                    elif i >= frame_stop:
                        continue
                    if grid is None:
                        H, W = frame.shape[:2]
                        grid = Grid.mesh(width=W, height=H, factor=factor)
                    out.writeFrame(annotate_frame(
                        frame,
                        el,
                        grid=grid,
                        grid_upsample=factor,
                        centercolor=centercolor,
                        facecolor=facecolor,
                        bordercolor=bordercolor
                    ))

