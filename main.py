import numpy as np
from typing import Any
import random
import pandas as pd


class Metrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | list = None):
        if len(y_true) != len(y_pred):
            raise ValueError('y_true and y_pred must have same length')

        self.y_pred = y_pred
        self.y_true = y_true
        self.labels = labels

        if labels is None:
            self.labels = np.unique(self.y_true)

        self.precision: float
        self.precision_by_class: dict
        self.recall: float
        self.recall_by_class: dict
        self.f1: float
        self.f1_by_class: dict
        self.accuracy: float
        self.accuracy_by_class: dict

        self._confusion_matrix_by_class: dict = self._get_confusion_matrix_by_class()

        self.precision, self.precision_by_class = self.get_precision()
        self.recall, self.recall_by_class = self.get_recall()
        self.f1, self.f1_by_class = self.get_f1()
        self.accuracy, self.accuracy_by_class = self.get_accuracy()

    def confusion_matrix(self) -> np.ndarray:
        matrix: np.ndarray = np.zeros(shape=(len(self.labels), len(self.labels)), dtype=np.int32)

        labels_idx: dict = {label: index for index, label in enumerate(self.labels)}

        for true_label, predicted in zip(self.y_true, self.y_pred):
            matrix[labels_idx[true_label], labels_idx[predicted]] += 1

        return matrix

    def _get_confusion_matrix_by_class(self) -> dict:
        matrix_by_class: dict = {}

        for label in self.labels:
            tp = np.sum((self.y_true == self.y_pred) & (self.y_true == label))
            fp = np.sum((self.y_true != self.y_pred) & (self.y_true == label))
            tn = np.sum((self.y_true == self.y_pred) & (self.y_true != label))
            fn = np.sum((self.y_true != self.y_pred) & (self.y_true != label))

            matrix_by_class[label] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

        return matrix_by_class

    def get_precision(self) -> (float, dict):
        precision_by_class: dict[str, float] = {
            str(label): self._confusion_matrix_by_class[label]["tp"] / d
            if (d := self._confusion_matrix_by_class[label]["tp"] + self._confusion_matrix_by_class[label]["fp"]) != 0
            else 0.0
            for label in self.labels
        }

        return np.mean(np.array(list(precision_by_class.values()))), precision_by_class

    def get_recall(self) -> (float, dict):
        recall_by_class: dict[str, float] = {
            str(label): self._confusion_matrix_by_class[label]["tp"] / d
            if (d := self._confusion_matrix_by_class[label]["tp"] + self._confusion_matrix_by_class[label]["fn"]) != 0
            else 0.0
            for label in self.labels
        }
        return np.mean(np.array(list(recall_by_class.values()))), recall_by_class

    def get_f1(self) -> (float, dict):
        f1_by_class: dict[str, float] = {
            str(label): 2 * self.precision_by_class[f'{label}'] * self.recall_by_class[f'{label}'] / d
            if (d := self.precision_by_class[f'{label}'] + self.recall_by_class[f'{label}']) != 0
            else 0.0
            for label in self.labels
        }
        return np.mean(np.array(list(f1_by_class.values()))), f1_by_class

    def get_accuracy(self) -> (float, dict):
        accuracy_by_class: dict[str, float] = {
            str(label): (self._confusion_matrix_by_class[label]["tp"] + self._confusion_matrix_by_class[label]["tn"]) / d
            if (d := len(self.y_pred)) != 0
            else 0.0
            for label in self.labels
        }
        return np.mean(np.array(list(accuracy_by_class.values()))), accuracy_by_class

    def get_metrics(self, verbose: int = 1) -> pd.DataFrame:
        data = {
            "precision": [self.precision],
            "recall": [self.recall],
            "f1": [self.f1],
            "accuracy": [self.accuracy]
        }
        metrics_df = pd.DataFrame(
            data=data,
            columns=("precision", "recall", "f1", "accuracy")
        )
        if verbose > 0:
            print(metrics_df)
            print()
        if verbose > 1:
            print("precision by class")
            for label in self.labels:
                print(f"{label}: {self.precision_by_class[f'{label}']}")
            print()
            print("recall by class")
            for label in self.labels:
                print(f"{label}: {self.recall_by_class[f'{label}']}")
            print()
            print("f1 by class")
            for label in self.labels:
                print(f"{label}: {self.f1_by_class[f'{label}']}")
            print()
            print("accuracy by class")
            for label in self.labels:
                print(f"{label}: {self.accuracy_by_class[f'{label}']}")
            print()

        return metrics_df


if __name__ == "__main__":
    label_num = 2
    num_data = 400

    y_true = np.array([random.randint(0, label_num-1) for _ in range(num_data)])
    y_pred = np.array([random.randint(0, label_num-1) for _ in range(num_data)])

    metrics = Metrics(y_true, y_pred)

    print(metrics.confusion_matrix())
    print('-' * 20)
    metrics.get_metrics(verbose=1)
    print('-' * 20)
    metrics.get_metrics(verbose=2)

