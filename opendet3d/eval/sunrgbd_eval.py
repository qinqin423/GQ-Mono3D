import numpy as np
from eval.detect3d import Detect3DEvaluator, Detect3Deval
from opendet3d.data.datasets.omni3d.util import get_dataset_det_map
from opendet3d.data.datasets.omni3d.omni3d_classes import omni3d_class_map


# SUNRGBD 10类
SUNRGBD_CLASSES = [
    "bed", "table", "sofa", "chair",
    "toilet", "desk", "dresser",
    "night stand", "bookshelf", "bathtub"
]


class SUNRGBDEvaluator:
    def __init__(self):
        dataset_name = "SUNRGBD_test"

        det_map = get_dataset_det_map(dataset_name=dataset_name)

        self.evaluator = Detect3DEvaluator(
            det_map=det_map,
            cat_map=omni3d_class_map,
            annotation="data/omni3d/annotations/SUNRGBD_test.json",
            eval_prox=True   # ⭐必须
        )

    def process_batch(self, coco_image_id, pred_boxes, pred_scores, pred_classes, pred_boxes3d):
        self.evaluator.process_batch(
            coco_image_id,
            pred_boxes,
            pred_scores,
            pred_classes,
            pred_boxes3d
        )

    def evaluate(self):
        coco_dt = self.evaluator._coco_gt.loadRes(
            self.evaluator._predictions
        )

        evaluator = Detect3Deval(
            self.evaluator._coco_gt,
            coco_dt,
            mode="3D",
            iou_type="bbox",
            eval_prox=True
        )

        # ⭐⭐只用 IoU = 0.25
        evaluator.params.iouThrs = np.array([0.25])

        evaluator.evaluate()
        evaluator.accumulate()

        precisions = evaluator.eval["precision"]
        cat_ids = self.evaluator._coco_gt.getCatIds()

        ap_list = []

        print("\nPer-class AP@0.25:")

        for idx, cat_id in enumerate(cat_ids):
            cat_name = self.evaluator._coco_gt.loadCats(cat_id)[0]["name"]

            if cat_name not in SUNRGBD_CLASSES:
                continue

            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]

            ap = np.mean(precision) if precision.size else float("nan")
            ap_list.append(ap)

            print(f"{cat_name:15s}: {ap:.3f}")

        mAP = np.mean(ap_list)

        print("\nFinal Result:")
        print(f"mAP@0.25 = {mAP:.4f}")

        return mAP