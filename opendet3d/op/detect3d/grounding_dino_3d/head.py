"""3D Grounding DINO ops."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import batched_nms, nms
from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.transformer import FFN, get_clones
from vis4d.op.layer.weight_init import xavier_init

from opendet3d.op.box.box2d import bbox_cxcywh_to_xyxy
from opendet3d.op.detect.grounding_dino.head import (
    convert_grounding_to_cls_scores,
)
from opendet3d.op.geometric.ray import generate_rays, rsh_cart_8
from opendet3d.op.layer.mlp import MLP
from opendet3d.op.util import flat_interpolate

from .coder import GroundingDINO3DCoder


class GroundingDINO3DHead(nn.Module):
    """3D Grounding DINO head with RGRH + GQH"""

    def __init__(
        self,
        embed_dims: int = 256,
        num_decoder_layer: int = 6,
        num_reg_fcs: int = 2,
        as_two_stage: bool = True,
        box_coder: GroundingDINO3DCoder | None = None,
        depth_output_scales: int = 1,
    ) -> None:
        """Initialize the 3D Grounding DINO head."""
        super().__init__()
        self.embed_dims = embed_dims

        self.num_pred_layer = (
            num_decoder_layer + 1 if as_two_stage else num_decoder_layer
        )
        self.as_two_stage = as_two_stage

        self.box_coder = box_coder or GroundingDINO3DCoder()

        # --- Residual Geometry Refinement Head ---
        reg_branch = self._get_reg_branch(num_reg_fcs, self.box_coder.reg_dims)
        self.reg_branches = get_clones(reg_branch, self.num_pred_layer)

        refine_branch = self._get_reg_branch(num_reg_fcs, 4)  # z + dim(3)
        self.refine_branches = get_clones(refine_branch, self.num_pred_layer)
        self.refine_scale = 0.10
        self.gaqs_alpha = 0.3  # GAQS 权重参数，控制几何质量对精修的加权程度

        # --- Geometry Quality Head ---
        geo_branch = self._get_reg_branch(num_reg_fcs, 1)  # single scalar per query
        self.geo_quality_branches = get_clones(geo_branch, self.num_pred_layer)

        # Camera / Depth condition branches
        project_rays, prompt_camera = self._get_condition_branch(
            input_dims=81, expansion=4, embed_dims=embed_dims
        )

        self.project_rays = get_clones(project_rays, self.num_pred_layer)
        self.prompt_camera = get_clones(prompt_camera, self.num_pred_layer)

        depth_embed_dims = embed_dims // 2**depth_output_scales

        project_depth, prompt_depth = self._get_condition_branch(
            depth_embed_dims, expansion=4, embed_dims=embed_dims
        )

        self.project_depth = get_clones(project_depth, self.num_pred_layer)
        self.prompt_depth = get_clones(prompt_depth, self.num_pred_layer)

        self._init_weights()

    def _get_reg_branch(
        self, num_reg_fcs: int, reg_dims: int
    ) -> nn.Sequential:
        """Get the regression branch."""
        layers = []
        for _ in range(num_reg_fcs):
            layers.append(nn.Linear(self.embed_dims, self.embed_dims))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.embed_dims, reg_dims))
        return nn.Sequential(*layers)

    def _get_condition_branch(
        self, input_dims: int, expansion: int, embed_dims: int
    ):
        """Get the condition branch."""
        project_layer = MLP(
            input_dims, expansion=expansion, output_dim=embed_dims
        )

        prompt_layer = Prompt3DQueryLayer(embed_dims)

        return project_layer, prompt_layer

    def _init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            xavier_init(m, distribution="uniform")
        for m in self.refine_branches:
            xavier_init(m, distribution="uniform")
        for m in self.geo_quality_branches:
            xavier_init(m, distribution="uniform")

    def get_camera_embeddings(
        self, intrinsics: Tensor, image_shape: tuple[int, int]
    ) -> Tensor:
        """Get the camera embeddings."""
        rays, _ = generate_rays(intrinsics, image_shape)

        rays = F.normalize(
            flat_interpolate(
                rays,
                old=image_shape,
                new=(image_shape[0] // 16, image_shape[1] // 16),
            ),
            dim=-1,
        )

        return rsh_cart_8(rays)

    def single_forward(
        self,
        layer_id: int,
        hidden_state: Tensor,
        ray_embeddings: Tensor,
        depth_latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Single layer forward pass of the 3D Grounding DINO head."""
        # Camera-aware 3D queries
        ray_embedding = self.project_rays[layer_id](ray_embeddings)

        hidden_state = self.prompt_camera[layer_id](
            hidden_state, ray_embedding, ray_embedding
        )

        # Depth-aware 3D queries
        proj_depth_latents = self.project_depth[layer_id](depth_latents)

        hidden_state = self.prompt_depth[layer_id](
            hidden_state, proj_depth_latents, proj_depth_latents
        )

        # RGRH: Residual Geometry Refinement Head
        reg_output = self.reg_branches[layer_id](hidden_state)

        refine_delta = torch.tanh(self.refine_branches[layer_id](hidden_state))
        refined_output = reg_output.clone()

        # 计算几何质量预测
        geo_quality_pred = torch.sigmoid(
            self.geo_quality_branches[layer_id](hidden_state)
        )

        # 在非训练模式下打印几何质量统计信息
        if not self.training:
            print(
                "[Q_STATS]",
                "mean=", geo_quality_pred.mean().item(),
                "std=", geo_quality_pred.std().item(),
                "min=", geo_quality_pred.min().item(),
                "max=", geo_quality_pred.max().item(),
            )

        # GAQS: Geometry-aware Query Selection
        # 使用几何质量分数动态加权精修步长
        q = geo_quality_pred.detach()
        refine_weight = 1.0 + self.gaqs_alpha * torch.relu(q - 0.5)

        refined_output[..., 2:6] += (
            self.refine_scale * refine_weight * refine_delta
        )

        return refined_output, geo_quality_pred

    def forward(
        self,
        hidden_states: Tensor,
        ray_embeddings: Tensor,
        depth_latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the 3D Grounding DINO head."""
        all_layers_outputs_3d = []
        all_layers_geo_quality_preds = []

        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]

            refined_output, geo_quality_pred = self.single_forward(
                layer_id, hidden_state, ray_embeddings, depth_latents
            )

            all_layers_outputs_3d.append(refined_output)
            all_layers_geo_quality_preds.append(geo_quality_pred)

        return (
            torch.stack(all_layers_outputs_3d),
            torch.stack(all_layers_geo_quality_preds),
        )


class Prompt3DQueryLayer(nn.Module):
    """Prompt 3D object query Layer."""

    def __init__(self, embed_dims: int = 256) -> None:
        """Init."""
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dims=256, num_heads=8, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dims)

        self.cross_attn = MultiheadAttention(
            embed_dims=256, num_heads=1, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = FFN(embed_dims)

        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Forward."""
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
        )
        query = self.norm1(query)

        # cross attention
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
        )
        query = self.norm2(query)

        # FFN
        query = self.ffn(query)
        query = self.norm3(query)

        return query


class RoI2Det3D:
    """Convert RoI to Detection."""

    def __init__(
        self,
        nms: bool = False,
        max_per_img: int = 300,
        class_agnostic_nms: bool = False,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        box_coder: GroundingDINO3DCoder | None = None,
    ) -> None:
        """Create an instance of RoI2Det."""
        self.nms = nms
        self.max_per_img = max_per_img
        self.class_agnostic_nms = class_agnostic_nms
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.box_coder = box_coder or GroundingDINO3DCoder()

    def __call__(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        token_positive_maps: dict[int, list[int]] | None,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        bbox_3d_pred: Tensor,
        geo_quality_pred: Tensor,  # 新增参数
        intrinsics: Tensor,
        padding: list[int] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transform the bbox head output into bbox results."""
        assert len(cls_score) == len(bbox_pred)  # num_queries

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps],
            )[0]

            scores, indexes = cls_score.view(-1).topk(self.max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            det_bboxes = det_bboxes[bbox_index]
            bbox_3d_pred = bbox_3d_pred[bbox_index]
            geo_quality_pred = geo_quality_pred[bbox_index]  # 获取对应的质量分数

            # Remove low scoring boxes
            if self.score_threshold > 0.0:
                mask = scores > self.score_threshold
                det_bboxes = det_bboxes[mask]
                det_labels = det_labels[mask]
                scores = scores[mask]
                bbox_3d_pred = bbox_3d_pred[mask]
                geo_quality_pred = geo_quality_pred[mask]

            # 使用几何质量分数校准分类分数
            alpha = 0.3  # 几何质量权重
            calibrated_scores = (1 - alpha) * scores + alpha * geo_quality_pred.squeeze(-1)

            if self.nms:
                if self.class_agnostic_nms:
                    keep = nms(det_bboxes, calibrated_scores, self.iou_threshold)
                else:
                    keep = batched_nms(
                        det_bboxes, calibrated_scores, det_labels, self.iou_threshold
                    )

                det_bboxes = det_bboxes[keep]
                det_labels = det_labels[keep]
                scores = calibrated_scores[keep]  # 使用校准后的分数
                bbox_3d_pred = bbox_3d_pred[keep]
                geo_quality_pred = geo_quality_pred[keep]  # 保持与筛选后的框对应
            else:
                scores = calibrated_scores  # 使用校准后的分数
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(self.max_per_img)
            det_bboxes = det_bboxes[indexes]
            bbox_3d_pred = bbox_3d_pred[indexes]
            geo_quality_pred = geo_quality_pred[indexes]  # 获取对应的质量分数
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

            # 使用几何质量分数校准分类分数
            alpha = 0.3
            calibrated_scores = (1 - alpha) * scores + alpha * geo_quality_pred.squeeze(-1)
            scores = calibrated_scores  # 使用校准后的分数

        if bbox_3d_pred.numel() == 0:
            return (
                det_bboxes,
                scores,
                det_labels,
                bbox_3d_pred.new_empty((0, 10)),
            )

        det_bboxes3d = self.box_coder.decode(
            det_bboxes, bbox_3d_pred, intrinsics
        )

        # Remove padding when input_hw is affected by padding
        if padding is not None:
            det_bboxes[:, 0] -= padding[0]
            det_bboxes[:, 1] -= padding[2]
            det_bboxes[:, 2] -= padding[0]
            det_bboxes[:, 3] -= padding[2]

            scales = [
                (img_shape[1] - padding[0] - padding[1]) / ori_shape[1],
                (img_shape[0] - padding[2] - padding[3]) / ori_shape[0],
            ]

        else:
            scales = [img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]]

        # Rescale to original shape
        det_bboxes /= det_bboxes.new_tensor(scales).repeat((1, 2))

        # 在最终返回前打印几何质量统计信息
        # 注意：这里假设self.training属性存在，但RoI2Det3D类没有training属性
        # 为了兼容性，我们检查是否有training属性，如果没有，则默认打印
        if not hasattr(self, 'training') or not self.training:
            print(
                "[Q_IOU_PREP]",
                "Q_mean=", geo_quality_pred.mean().item(),
                "Q_std=", geo_quality_pred.std().item(),
                "NumBoxes=", len(scores),
            )

        return det_bboxes, scores, det_labels, det_bboxes3d, geo_quality_pred