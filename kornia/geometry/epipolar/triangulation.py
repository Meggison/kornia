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

"""Module with the functionalities for triangulation."""

from __future__ import annotations

import torch

from kornia.core import zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.conversions import convert_points_from_homogeneous
from kornia.utils.helpers import _torch_svd_cast

# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp#L68


def triangulate_points(
    P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor, points2: torch.Tensor
) -> torch.Tensor:
    """Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1: The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1: The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2: The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.

    """
    # shape checks (not hotspots except when error is thrown)
    KORNIA_CHECK_SHAPE(P1, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(P2, ["*", "3", "4"])
    KORNIA_CHECK_SHAPE(points1, ["*", "N", "2"])
    KORNIA_CHECK_SHAPE(points2, ["*", "N", "2"])

    # Vectorized computation for the equations matrix X
    # instead of 4 explicit for-loop assignments, batch them
    # Let S = broadcasted shape like (*, N)
    batch_shape = torch.broadcast_shapes(points1.shape[:-1], points2.shape[:-1])
    # N: number of points per batch
    N = batch_shape[-1] if len(batch_shape) > 0 else points1.shape[-2]
    final_shape = batch_shape + (4, 4)
    # Use zeros_like points1[..., :1, :1] to get the dtype/device also right.
    # Optimize zeros allocation.
    X = zeros(final_shape, dtype=points1.dtype, device=points1.device)

    # Expand P1 and P2 to the batch/point shape (broadcast safely)
    # points1, points2 shape: (*, N, 2)
    # P1, P2 shape: (*, 3, 4)

    # Assign the four equations in a vectorized manner.
    x1 = points1[..., 0]
    y1 = points1[..., 1]
    x2 = points2[..., 0]
    y2 = points2[..., 1]
    # The [...] comes from batch dims, N comes from [..., N, 2], and all broadcasting will work

    P1_0 = P1[..., 0:1, :]  # (..., 1, 4)
    P1_1 = P1[..., 1:2, :]  # (..., 1, 4)
    P1_2 = P1[..., 2:3, :]  # (..., 1, 4)
    P2_0 = P2[..., 0:1, :]
    P2_1 = P2[..., 1:2, :]
    P2_2 = P2[..., 2:3, :]

    # All slices' last two dims are (..., N, 4)
    X[..., 0, :] = x1.unsqueeze(-1) * P1_2 - P1_0
    X[..., 1, :] = y1.unsqueeze(-1) * P1_2 - P1_1
    X[..., 2, :] = x2.unsqueeze(-1) * P2_2 - P2_0
    X[..., 3, :] = y2.unsqueeze(-1) * P2_2 - P2_1

    # SVD is inherently a bottleneck
    _, _, V = _torch_svd_cast(X)

    # Efficiently extract (..., N, 4) from (..., N, 4, 4)
    # We want the last column: V[..., -1]
    points3d_h = V[..., -1]
    points3d = convert_points_from_homogeneous(points3d_h)
    return points3d
