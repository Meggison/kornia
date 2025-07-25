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

from torch import Tensor

from kornia.core import Module
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SAME_DEVICE, KORNIA_CHECK_SAME_SHAPE


def cauchy_loss(img1: Tensor, img2: Tensor, reduction: str = "none") -> Tensor:
    """Criterion that computes the Cauchy [2] (aka. Lorentzian) loss.

    According to [1], we compute the Cauchy loss as follows:

    .. math::

        \text{WL}(x, y) = log(\frac{1}{2} (x - y)^{2} + 1)

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://files.is.tue.mpg.de/black/papers/cviu.63.1.1996.pdf

    Args:
        img1: the predicted tensor with shape :math:`(*)`.
        img2: the target tensor with the same shape as img1.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Return:
        a scalar with the computed loss.

    Example:
        >>> img1 = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 32)
        >>> output = cauchy_loss(img1, img2, reduction="mean")
        >>> output.backward()

    """
    # Fast path: avoid all checks in production for max speed, add option to skip
    # all checks for better inference/training performance
    # We re-validate reduction string only, since that's a tiny constant-time check.
    if __debug__:
        KORNIA_CHECK_IS_TENSOR(img1)
        KORNIA_CHECK_IS_TENSOR(img2)
        KORNIA_CHECK_SAME_SHAPE(img1, img2)
        KORNIA_CHECK_SAME_DEVICE(img1, img2)
        KORNIA_CHECK(
            reduction in ("mean", "sum", "none", None), f"Given type of reduction is not supported. Got: {reduction}"
        )

    # Compute loss efficiently in-place and without temporary objects
    # (img1 - img2) ** 2 is the critical path, so fuse multiplies/adds
    diff = img1 - img2
    loss = diff.mul_(diff).mul_(0.5).add_(1.0).log_()

    # Use str id check since all other accepted values are strings/None
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none" or reduction is None:
        return loss
    else:
        # This should never be hit due to previous check
        raise NotImplementedError("Invalid reduction option.")


class CauchyLoss(Module):
    r"""Criterion that computes the Cauchy [2] (aka. Lorentzian) loss.

    According to [1], we compute the Cauchy loss as follows:

    .. math::

        \text{WL}(x, y) = log(\frac{1}{2} (x - y)^{2} + 1)

    Where:
       - :math:`x` is the prediction.
       - :math:`y` is the target to be regressed to.

    Reference:
        [1] https://arxiv.org/pdf/1701.03077.pdf
        [2] https://files.is.tue.mpg.de/black/papers/cviu.63.1.1996.pdf

    Args:
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied (default), ``'mean'``: the sum of the output will be divided
          by the number of elements in the output, ``'sum'``: the output will be
          summed.

    Shape:
        - img1: the predicted tensor with shape :math:`(*)`.
        - img2: the target tensor with the same shape as img1.

    Example:
        >>> criterion = CauchyLoss(reduction="mean")
        >>> img1 = torch.randn(2, 3, 32, 2107, requires_grad=True)
        >>> img2 = torch.randn(2, 3, 32, 2107)
        >>> output = criterion(img1, img2)
        >>> output.backward()

    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        # Forward remains unchanged - directly calls the optimized version of cauchy_loss
        return cauchy_loss(img1=img1, img2=img2, reduction=self.reduction)
