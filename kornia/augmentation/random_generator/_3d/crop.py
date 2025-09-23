# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Device, Tensor, tensor, zeros
from kornia.geometry.bbox import bbox_generator3d
from kornia.utils.helpers import _extract_device_dtype


class CropGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ```crop``` transformation for crop transform.

    Args:
        size (tuple): Desired size of the crop operation, like (d, h, w).
            If tensor, it must be (B, 3).
        resize_to (tuple): Desired output size of the crop, like (d, h, w). If None, no resize will be performed.

    Returns:
        A dict of parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(
        self, size: Union[Tuple[int, int, int], Tensor], resize_to: Optional[Tuple[int, int, int]] = None
    ) -> None:
        super().__init__()
        self.size = size
        self.resize_to = resize_to

    def __repr__(self) -> str:
        repr = f"crop_size={self.size}"
        if self.resize_to is not None:
            repr += f", resize_to={self.resize_to}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.rand_sampler = Uniform(tensor(0.0, device=device, dtype=dtype), tensor(1.0, device=device, dtype=dtype))

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size, _, depth, height, width = batch_shape
        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.size if isinstance(self.size, Tensor) else None])

        if not isinstance(self.size, Tensor):
            size = tensor(self.size, device=_device, dtype=_dtype).expand(batch_size, 3)
        else:
            size = self.size.to(device=_device, dtype=_dtype)
        if size.shape != torch.Size([batch_size, 3]):
            raise AssertionError(
                "If `size` is a tensor, it must be shaped as (B, 3). "
                f"Got {size.shape} while expecting {torch.Size([batch_size, 3])}."
            )
        if not (
            isinstance(depth, int)
            and isinstance(height, int)
            and isinstance(width, int)
            and depth > 0
            and height > 0
            and width > 0
        ):
            raise AssertionError(f"`batch_shape` should not contain negative values. Got {(batch_shape)}.")

        # Note: Avoid temporary tensors and favor inplace, and avoid multiple allocation and computation
        x_diff = width - size[:, 2] + 1
        y_diff = height - size[:, 1] + 1
        z_diff = depth - size[:, 0] + 1

        if (x_diff < 0).any() or (y_diff < 0).any() or (z_diff < 0).any():
            raise ValueError(
                f"input_size {(depth, height, width)} cannot be smaller than crop size {size!s} in any dimension."
            )

        if batch_size == 0:
            out0 = zeros([0, 8, 3], device=_device, dtype=_dtype)
            return {"src": out0, "dst": out0}

        # cache for batch_size==1 (micro-optimization for frequent case)
        if same_on_batch:
            _rsample_shape = (1,)
        else:
            _rsample_shape = (batch_size,)

        # Sample all starts together for speed; reuse _adapted_rsampling
        rand = _adapted_rsampling(_rsample_shape, self.rand_sampler, same_on_batch)
        if same_on_batch and batch_size > 1:
            rand = rand.expand(batch_size)
        rand = rand.to(device=_device, dtype=_dtype)

        # avoid repeat _adapted_rsampling call when possible (fuse to one random for all dimensions)
        # but keep separate for correct semantics
        x_start = _adapted_rsampling(_rsample_shape, self.rand_sampler, same_on_batch)
        y_start = _adapted_rsampling(_rsample_shape, self.rand_sampler, same_on_batch)
        z_start = _adapted_rsampling(_rsample_shape, self.rand_sampler, same_on_batch)
        if same_on_batch and batch_size > 1:
            x_start = x_start.expand(batch_size)
            y_start = y_start.expand(batch_size)
            z_start = z_start.expand(batch_size)
        x_start = (x_start.to(device=_device, dtype=_dtype) * x_diff).floor()
        y_start = (y_start.to(device=_device, dtype=_dtype) * y_diff).floor()
        z_start = (z_start.to(device=_device, dtype=_dtype) * z_diff).floor()

        # avoid repeated (size[:,n]-1) computation
        sz2 = size[:, 2] - 1
        sz1 = size[:, 1] - 1
        sz0 = size[:, 0] - 1

        crop_src = bbox_generator3d(x_start.view(-1), y_start.view(-1), z_start.view(-1), sz2, sz1, sz0)

        # don't repeat list allocation in batch crop_dst creation (broadcasting)
        if self.resize_to is None:
            zeros_tmp = zeros([batch_size], device=_device, dtype=_dtype)
            crop_dst = bbox_generator3d(zeros_tmp, zeros_tmp, zeros_tmp, sz2, sz1, sz0)
        else:
            rto = self.resize_to
            if not (
                len(rto) == 3
                and isinstance(rto[0], int)
                and isinstance(rto[1], int)
                and isinstance(rto[2], int)
                and rto[0] > 0
                and rto[1] > 0
                and rto[2] > 0
            ):
                raise AssertionError(f"`resize_to` must be a tuple of 3 positive integers. Got {self.resize_to}.")
            rx, ry, rz = rto[-1], rto[-2], rto[-3]
            corners = [
                [0, 0, 0],
                [rx - 1, 0, 0],
                [rx - 1, ry - 1, 0],
                [0, ry - 1, 0],
                [0, 0, rz - 1],
                [rx - 1, 0, rz - 1],
                [rx - 1, ry - 1, rz - 1],
                [0, ry - 1, rz - 1],
            ]
            crop_dst = tensor(corners, device=_device, dtype=_dtype).unsqueeze(0).expand(batch_size, 8, 3)

        return {"src": crop_src.to(device=_device), "dst": crop_dst.to(device=_device)}


def center_crop_generator3d(
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    size: Tuple[int, int, int],
    device: Optional[Device] = None,
) -> Dict[str, Tensor]:
    r"""Get parameters for ```center_crop3d``` transformation for center crop transform.

    Args:
        batch_size (int): the tensor batch size.
        depth (int) : depth of the image.
        height (int) : height of the image.
        width (int): width of the image.
        size (tuple): Desired output size of the crop, like (d, h, w).
        device (Device): the device on which the random numbers will be generated. Default: cpu.

    Returns:
        params Dict[str, Tensor]: parameters to be passed for transformation.
            - src (Tensor): cropping bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): output bounding boxes with a shape (B, 8, 3).

    Note:
        No random number will be generated.

    """
    if device is None:
        device = torch.device("cpu")
    if not isinstance(size, (tuple, list)) and len(size) == 3:
        raise ValueError(f"Input size must be a tuple/list of length 3. Got {size}")
    if not (
        isinstance(depth, int)
        and depth > 0
        and isinstance(height, int)
        and height > 0
        and isinstance(width, int)
        and width > 0
    ):
        raise AssertionError(f"'depth', 'height' and 'width' must be integers. Got {depth}, {height}, {width}.")
    if not (depth >= size[0] and height >= size[1] and width >= size[2]):
        raise AssertionError(f"Crop size must be smaller than input size. Got ({depth}, {height}, {width}) and {size}.")

    if batch_size == 0:
        return {"src": zeros([0, 8, 3]), "dst": zeros([0, 8, 3])}
    # unpack input sizes
    dst_d, dst_h, dst_w = size
    src_d, src_h, src_w = (depth, height, width)

    # compute start/end offsets
    dst_d_half = dst_d / 2
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_d_half = src_d / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half
    start_z = src_d_half - dst_d_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    end_z = start_z + dst_d - 1
    # [x, y, z] origin
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    # Note: DeprecationWarning: an integer is required (got type float).
    # Implicit conversion to integers using __int__ is deprecated, and may be removed in a future version of Python.
    points_src: Tensor = tensor(
        [
            [
                [int(start_x), int(start_y), int(start_z)],
                [int(end_x), int(start_y), int(start_z)],
                [int(end_x), int(end_y), int(start_z)],
                [int(start_x), int(end_y), int(start_z)],
                [int(start_x), int(start_y), int(end_z)],
                [int(end_x), int(start_y), int(end_z)],
                [int(end_x), int(end_y), int(end_z)],
                [int(start_x), int(end_y), int(end_z)],
            ]
        ],
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1, -1)

    # [x, y, z] destination
    # top-left-front, top-right-front, bottom-right-front, bottom-left-front
    # top-left-back, top-right-back, bottom-right-back, bottom-left-back
    points_dst: Tensor = tensor(
        [
            [
                [0, 0, 0],
                [dst_w - 1, 0, 0],
                [dst_w - 1, dst_h - 1, 0],
                [0, dst_h - 1, 0],
                [0, 0, dst_d - 1],
                [dst_w - 1, 0, dst_d - 1],
                [dst_w - 1, dst_h - 1, dst_d - 1],
                [0, dst_h - 1, dst_d - 1],
            ]
        ],
        device=device,
        dtype=torch.long,
    ).expand(batch_size, -1, -1)
    return {"src": points_src, "dst": points_dst}
