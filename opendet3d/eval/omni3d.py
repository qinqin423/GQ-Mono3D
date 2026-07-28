"""Omni3D 3D detection evaluation."""

import contextlib
import copy
import io
import itertools
import os
from collections.abc import Sequence

import numpy as np
from terminaltables import AsciiTable
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import GenericFunc, MetricLogs, NDArrayNumber
from vis4d.eval.base import Evaluator

from opendet3d.data.datasets.omni3d.omni3d_classes import omni3d_class_map
from opendet3d.data.datasets.omni3d.util import get_dataset_det_map

from .detect3d import Detect3Deval, Detect3DEvaluator

omni3d_in = {
    "stationery",
    "sink",
    "table",
    "floor mat",
    "bottle",
    "bookcase",
    "bin",
    "blinds",
    "pillow",
    "bicycle",
    "refrigerator",
    "night stand",
    "chair",
    "sofa",
    "books",
    "oven",
    "towel",
    "cabinet",
    "window",
    "curtain",
    "bathtub",
    "laptop",
    "desk",
    "television",
    "clothes",
    "stove",
    "cup",
    "shelves",
    "box",
    "shoes",
    "mirror",
    "door",
    "picture",
    "lamp",
    "machine",
    "counter",
    "bed",
    "toilet",
}

omni3d_out = {
    "cyclist",
    "pedestrian",
    "trailer",
    "bus",
    "motorcycle",
    "car",
    "barrier",
    "truck",
    "van",
    "traffic cone",
    "bicycle",
}


class Omni3DEvaluator(Evaluator):
    """Simplified Omni3D evaluator (SUNRGBD only, AP25/AP50)."""

    def __init__(
        self,
        data_root: str = "data/omni3d",
        omni3d50: bool = True,
        datasets: Sequence[str] = ("SUNRGBD_test",),
        per_class_eval: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_name = "SUNRGBD_test"
        self.per_class_eval = per_class_eval

        annotation = os.path.join(
            data_root, "annotations", f"{self.dataset_name}.json"
        )

        det_map = get_dataset_det_map(
            dataset_name=self.dataset_name, omni3d50=omni3d50
        )

        self.evaluator = Detect3DEvaluator(
            det_map,
            cat_map=omni3d_class_map,
            annotation=annotation,
            eval_prox=True,
        )

    def __repr__(self) -> str:
        return f"Omni3DEvaluator (SUNRGBD only)"

    @property
    def metrics(self) -> list[str]:
        return ["3D"]

    def reset(self) -> None:
        self.evaluator.reset()

    def gather(self, gather_func: GenericFunc) -> None:
        self.evaluator.gather(gather_func)

    def process_batch(
        self,
        coco_image_id,
        dataset_names,
        pred_boxes,
        pred_scores,
        pred_classes,
        pred_boxes3d=None,
        pred_geo_qualities=None,
    ) -> None:
        for i in range(len(coco_image_id)):
            self.evaluator.process_batch(
                [coco_image_id[i]],
                [pred_boxes[i]],
                [pred_scores[i]],
                [pred_classes[i]],
                pred_boxes3d=[pred_boxes3d[i]] if pred_boxes3d else None,
                pred_geo_qualities=[pred_geo_qualities[i]] if pred_geo_qualities else None,
            )

    def evaluate(self, metric: str):
        import inspect

        print("===== DEBUG =====")
        print("Evaluator type:", type(self.evaluator))
        print("Evaluator file:", inspect.getfile(self.evaluator.__class__))
        print("=================")

        score_dict, log_str = self.evaluator.evaluate(metric)

        new_score_dict = {
            "mAP@0.25": score_dict.get("mAP@0.25", 0.0),
            "mAP@0.50": score_dict.get("mAP@0.5", 0.0),
        }

        return new_score_dict, log_str

    def save(self, metric: str, output_dir: str) -> None:
        if hasattr(self.evaluator, "save"):
            try:
                self.evaluator.save(metric, output_dir, prefix="SUNRGBD")
            except TypeError:
                self.evaluator.save(metric, output_dir)