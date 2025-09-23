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

"""Post-processor for the RT-DETR model."""

from __future__ import annotations

from typing import Optional, Union

import torch

from kornia.core import Module, Tensor, tensor
from kornia.models.detection.utils import BoxFiltering

"""Post-processor for the RT-DETR model."""


def mod(a: Tensor, b: int) -> Tensor:
    """Compute the element-wise remainder of tensor `a` divided by integer `b`.

    This function requires `a` to be a `torch.Tensor` and `b` to be an `int`.
    It returns a `torch.Tensor` with the same shape/device as `a`. The
    implementation uses `a % b` (equivalent to `torch.remainder(a, b)`).

    Args:
        a (torch.Tensor): Dividend tensor (any numeric dtype).
        b (int): Divisor (must be non-zero).

    Returns:
        torch.Tensor: Element-wise remainder of `a` divided by `b`.

    Examples:
        >>> mod(torch.tensor(7), 3)
        tensor(1)
        >>> mod(torch.tensor([7, -1, 2]), 3)
        tensor([1, 2, 2])
    """
    return a % b


# TODO: deprecate the confidence threshold and add the num_top_queries as a parameter and num_classes as a parameter
class DETRPostProcessor(Module):
    def __init__(
        self,
        confidence_threshold: Optional[float] = None,
        num_classes: int = 80,
        num_top_queries: int = 300,
        confidence_filtering: bool = True,
        filter_as_zero: bool = False,
    ) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self.confidence_filtering = confidence_filtering
        self.num_top_queries = num_top_queries
        self.box_filtering = BoxFiltering(
            tensor(confidence_threshold) if confidence_threshold is not None else None, filter_as_zero=filter_as_zero
        )

    def forward(self, logits: Tensor, boxes: Tensor, original_sizes: Tensor) -> Union[Tensor, list[Tensor]]:
        """Post-process outputs from DETR.

        Args:
            logits: tensor with shape :math:`(N, Q, K)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries, :math:`K` is the number of classes.
            boxes: tensor with shape :math:`(N, Q, 4)`, where :math:`N` is the batch size, :math:`Q` is the number of
                queries.
            original_sizes: tensor with shape :math:`(N, 2)`, where :math:`N` is the batch size and each element
                represents the image size of (img_height, img_width).

        Returns:
            Processed detections. For each image, the detections have shape (D, 6), where D is the number of detections
            in that image, 6 represent (class_id, confidence_score, x, y, w, h).

        """
        cxcy, wh = boxes[..., :2], boxes[..., 2:]
        # Fuse cxcy - wh*0.5 and concatenate
        half_wh = wh * 0.5
        xy_min = cxcy - half_wh
        boxes_xy = torch.cat([xy_min, wh], dim=-1)

        # Efficient box scaling
        # shapes: boxes_xy (N, Q, 4)
        # Get (img_w, img_h) from original_sizes[0], shape (2,) -> (1, 2)
        img_size = original_sizes[0].flip(0).unsqueeze(0)  # (1, 2)
        # Expand to (1, 1, 4): (img_w, img_h, img_w, img_h)
        scale = img_size.repeat(1, 2)  # (1, 4)
        boxes_xy = boxes_xy * scale  # (N, Q, 4), broadcast (1, 4)

        # Fast sigmoid and flatten
        scores = logits.sigmoid()
        batch_size, num_queries, num_classes = scores.size()
        scores_flat = scores.reshape(batch_size, -1)

        # topk on 2d batched input
        topk_scores, topk_indices = torch.topk(scores_flat, self.num_top_queries, dim=-1)

        # Use divmod vectorized via torch.div and torch.remainder
        labels = torch.remainder(topk_indices, self.num_classes)
        indices = torch.div(topk_indices, self.num_classes, rounding_mode="trunc")

        # More efficient gather by constructing index tensor with correct shape up front
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, boxes_xy.shape[-1])
        selected_boxes = torch.gather(boxes_xy, 1, expanded_indices)

        # Stack outputs efficiently
        all_boxes = torch.cat(
            [labels.unsqueeze(-1).to(selected_boxes.dtype), topk_scores.unsqueeze(-1), selected_boxes], dim=-1
        )

        if not self.confidence_filtering or self.confidence_threshold == 0:
            return all_boxes

        return self.box_filtering(all_boxes, self.confidence_threshold)
