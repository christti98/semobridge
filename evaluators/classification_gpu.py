import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix

from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation import EvaluatorBase


@EVALUATOR_REGISTRY.register()
class ClassificationGPU(EvaluatorBase):
    """Evaluator for classification on GPU."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        mo = mo.to(self.device)
        gt = gt.to(self.device)

        pred = mo.argmax(dim=1)
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.size(0)

        self._y_true.append(gt)
        self._y_pred.append(pred)

        if self._per_class_res is not None:
            for i in range(gt.size(0)):
                label = gt[i].item()
                correct = int(matches[i].item())
                self._per_class_res[label].append(correct)

    def evaluate(self):
        results = OrderedDict()

        # Concatenate GPU tensors and move to CPU once
        y_true = torch.cat(self._y_true).cpu().numpy()
        y_pred = torch.cat(self._y_pred).cpu().numpy()

        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            y_true, y_pred, average="macro", labels=np.unique(y_true)
        )

        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )

        if self._per_class_res is not None:
            labels = sorted(self._per_class_res.keys())
            print("=> per-class result")
            accs = []
            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")
            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(y_true, y_pred, normalize="true")
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(torch.tensor(cmat), save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
