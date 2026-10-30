from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import numpy as np
from scipy import stats
from attack_utils import *
from memory_profiler import profile

from experiment import ExperimentConfiguration

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

@dataclass
class ModelExperiment:
    name: str
    data: np.ndarray
    train: np.ndarray
    train_labels: np.ndarray
    valid: np.ndarray
    valid_labels: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    labels: np.ndarray
    wavelengths: np.ndarray
    layer: int
    groups: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("data must be a two-dimensional array")
        if self.labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional array")
        if self.data.shape[0] != len(self.labels):
            raise ValueError("Each data row must have one label")
        if self.data.shape[1] != len(self.wavelengths):
            raise ValueError("Each data column must have one wavelength")
        if self.train.ndim != 2:
            raise ValueError("Training data must be two-dimensional")
        if self.valid.ndim != 2:
            raise ValueError("Validation data must be two-dimensional")
        if self.test.ndim != 2:
            raise ValueError("Test data must be two-dimensional")        

    @classmethod
    def from_experimentConfiguration(
        cls, 
        exp: ExperimentConfiguration, 
        train_split: float = 0.33, 
        valid_split: float = 0.33, 
        test_split: float = 0.34,
        group_by_print: bool = True,
        layer_wise: bool = True,
        exp_layer: int = 0
    ) -> ModelExperiment:
  
        # Splits an ExperimentConfiguration into train, validation, and test sets. If group_by_print is True, splits 1:1:1 across the physical print sessions/groups.

        if abs(train_split + valid_split + test_split - 1.0) > 1e-5:
            raise ValueError("Splits must sum to 1.0")

        data = exp.data
        wavelengths = exp.wavelengths
        layer = exp_layer

        if layer_wise:
            labels = create_layer_labels(exp.sample_names,exp_layer)
        else:
            labels = exp.sample_names

        if group_by_print:
            train_idx, valid_idx, test_idx = [], [], []
            rows_per_print = exp.measurements_per_print
            total_prints = exp.group_total
            groups = np.zeros(len(data), dtype=int)

            for sample in exp.variation_names:
                mask = exp.sample_names == sample
                sample_row_indices = np.where(mask)[0]
                
                for print_idx in range(total_prints):
                    start = print_idx * rows_per_print
                    end = (print_idx + 1) * rows_per_print
                    print_indices = sample_row_indices[start:end]
                    groups[print_indices] = print_idx

                    # Assign each print session to train, valid, or test based on print index rotation
                    target_bucket = print_idx % 3
                    if target_bucket == 0:
                        train_idx.extend(print_indices)
                    elif target_bucket == 1:
                        valid_idx.extend(print_indices)
                    else:
                        test_idx.extend(print_indices)

            train_idx = np.array(train_idx)
            valid_idx = np.array(valid_idx)
            test_idx = np.array(test_idx)
        else:
            # Random stratified split
            groups = np.zeros(len(data), dtype=int)
            train_val_size = train_split + valid_split
            train_idx, temp_idx = train_test_split(
                np.arange(len(data)), train_size=train_split/train_val_size, stratify=labels, random_state=42
            )
            val_relative = valid_split / (valid_split + test_split)
            valid_rel_idx, test_rel_idx = train_test_split(
                temp_idx, train_size=val_relative, stratify=labels[temp_idx], random_state=42
            )
            valid_idx = valid_rel_idx
            test_idx = test_rel_idx

        return cls(
            name=exp.name,
            data=data,
            train=data[train_idx],
            train_labels=labels[train_idx],
            valid=data[valid_idx],
            valid_labels=labels[valid_idx],
            test=data[test_idx],
            test_labels=labels[test_idx],
            labels=labels,
            wavelengths=wavelengths,
            layer=exp_layer,
            groups=groups
        )

class AttackModel(ABC):

    @abstractmethod
    def attack(self, model: ModelExperiment) -> dict[str, Any]:
        raise NotImplementedError


class KNNClassifier(AttackModel):
    def __init__(self, n_neighbors: int = 3, use_pca: bool = True, n_components: int = 10):
        self.n_neighbors = n_neighbors
        self.use_pca = use_pca
        self.n_components = n_components
        self.pipeline = None

    def attack(self, model: ModelExperiment):

        steps = [('scaler', StandardScaler())]

        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))

        steps.append(('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)))

        self.pipeline = Pipeline(steps)

        self.pipeline.fit(model.train, model.train_labels)

        val_preds = self.pipeline.predict(model.valid)
        val_accuracy = accuracy_score(model.valid_labels, val_preds)

        test_preds = self.pipeline.predict(model.test)
        test_accuracy = accuracy_score(model.test_labels, test_preds)
        report = classification_report(model.test_labels, test_preds, output_dict=True, zero_division=0)
        #conf_matrix = confusion_matrix(model.test_labels, test_preds)

        print(f"Experiment Model : {model.layer}")
        print(f"KNN Attack Results (k={self.n_neighbors}, PCA={self.use_pca}):")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")

        return {
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            #"confusion_matrix": conf_matrix,
            #"predictions": test_preds,
            #"test_labels": model.test_labels
        }

    def cross_validate_attack(self, model: ModelExperiment) -> dict[str, Any]:

        data = model.data
        labels = model.labels
        groups = model.groups

        logo = LeaveOneGroupOut()

        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))
        
        steps.append(('knn', KNeighborsClassifier(n_neighbors=self.n_neighbors)))

        fold_accuracies = []
        for train_idx, test_idx in logo.split(data, labels, groups):
            pipeline = Pipeline(steps)
            pipeline.fit(data[train_idx], labels[train_idx])
            preds = pipeline.predict(data[test_idx])
            fold_accuracies.append(accuracy_score(labels[test_idx], preds))

        max_acc = float(np.max(fold_accuracies)) if fold_accuracies else 0.0
        mean_acc = float(np.mean(fold_accuracies))

        print(f"Experiment Model : {model.layer}")
        print(f"KNN Attack Results (k={self.n_neighbors}, PCA={self.use_pca}):")
        print(f"Max CV Accuracy:  {max_acc:.4f}")
        print(f"Mean CV Accuracy: {mean_acc:.4f}")

        return {
            "fold_accuracies": fold_accuracies,
            "max_accuracy": max_acc,
            "mean_accuracy": mean_acc
        }

class NaiveBayes_Classifier(AttackModel):
    def __init__(self ,use_pca: bool = True, n_components: int = 10):
        self.use_pca = use_pca
        self.n_components = n_components
        self.pipeline = None

    def attack(self, model: ModelExperiment):

        steps = [('scaler', StandardScaler())]

        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))

        steps.append(('naive-bayes', GaussianNB()))

        self.pipeline = Pipeline(steps)

        self.pipeline.fit(model.train, model.train_labels)

        val_preds = self.pipeline.predict(model.valid)
        val_accuracy = accuracy_score(model.valid_labels, val_preds)

        test_preds = self.pipeline.predict(model.test)
        test_accuracy = accuracy_score(model.test_labels, test_preds)
        report = classification_report(model.test_labels, test_preds, output_dict=True,zero_division=0)
        #conf_matrix = confusion_matrix(model.test_labels, test_preds)

        print(f"Experiment Model : {model.layer}")
        print(f"Naive Bayes Attack Results (PCA={self.use_pca}):")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")

        return {
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            #"confusion_matrix": conf_matrix,
            #"predictions": test_preds,
            #"test_labels": model.test_labels
        }

    def cross_validate_attack(self, model: ModelExperiment) -> dict[str, Any]:

        data = model.data
        labels = model.labels
        groups = model.groups

        logo = LeaveOneGroupOut()

        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))
        
        steps.append(('naive-bayes', GaussianNB()))

        fold_accuracies = []
        for train_idx, test_idx in logo.split(data, labels, groups):
            pipeline = Pipeline(steps)
            pipeline.fit(data[train_idx], labels[train_idx])
            preds = pipeline.predict(data[test_idx])
            fold_accuracies.append(accuracy_score(labels[test_idx], preds))

        max_acc = float(np.max(fold_accuracies)) if fold_accuracies else 0.0
        mean_acc = float(np.mean(fold_accuracies))

        print(f"Experiment Model : {model.layer}")
        print(f"Naive Bayes Attack Results (PCA={self.use_pca}):")
        print(f"Max CV Accuracy:  {max_acc:.4f}")
        print(f"Mean CV Accuracy: {mean_acc:.4f}")

        return {
            "fold_accuracies": fold_accuracies,
            "max_accuracy": max_acc,
            "mean_accuracy": mean_acc
        }

class LogisticRegression_classifier(AttackModel):
    def __init__(self, use_pca: bool = True, n_components = 10, max_iter: int = 1000):
        self.use_pca = use_pca
        self.n_components = n_components
        self.max_iter = max_iter
        self.pipeline = None

    def attack(self, model: ModelExperiment):

        steps = [('scaler', StandardScaler())]

        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))

        steps.append(("Logistic_Regression", LogisticRegression(max_iter=self.max_iter)))

        self.pipeline = Pipeline(steps)

        self.pipeline.fit(model.train, model.train_labels)

        val_preds = self.pipeline.predict(model.valid)
        val_accuracy = accuracy_score(model.valid_labels, val_preds)

        test_preds = self.pipeline.predict(model.test)
        test_accuracy = accuracy_score(model.test_labels, test_preds)
        report = classification_report(model.test_labels, test_preds, output_dict=True,zero_division=0)
        #conf_matrix = confusion_matrix(model.test_labels, test_preds)

        print(f"Experiment Model : {model.layer}")
        print(f"Logistic Regression Attack Results (PCA={self.use_pca}):")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")

        return {
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            #"confusion_matrix": conf_matrix,
            #"predictions": test_preds,
            #"test_labels": model.test_labels
        }

    def cross_validate_attack(self, model: ModelExperiment) -> dict[str, Any]:

        data = model.data
        labels = model.labels
        groups = model.groups

        logo = LeaveOneGroupOut()

        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))
        
        steps.append(("Logistic_Regression", LogisticRegression(max_iter=self.max_iter)))

        fold_accuracies = []
        for train_idx, test_idx in logo.split(data, labels, groups):
            pipeline = Pipeline(steps)
            pipeline.fit(data[train_idx], labels[train_idx])
            preds = pipeline.predict(data[test_idx])
            fold_accuracies.append(accuracy_score(labels[test_idx], preds))

        max_acc = float(np.max(fold_accuracies)) if fold_accuracies else 0.0
        mean_acc = float(np.mean(fold_accuracies))

        print(f"Experiment Model : {model.layer}")
        print(f"Logistic Regression Attack Results (PCA={self.use_pca}):")
        print(f"Max CV Accuracy:  {max_acc:.4f}")
        print(f"Mean CV Accuracy: {mean_acc:.4f}")

        return {
            "fold_accuracies": fold_accuracies,
            "max_accuracy": max_acc,
            "mean_accuracy": mean_acc
        }


class SVM_classifier(AttackModel):

    def __init__(self, use_pca: bool = True, n_components = 10, max_iter: int = 1000):
        self.use_pca = use_pca
        self.n_components = n_components
        self.max_iter = max_iter
        self.pipeline = None

    def attack(self, model: ModelExperiment):

        steps = [('scaler', StandardScaler())]

        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))

        steps.append(("support-vector-machine", SVC(max_iter=self.max_iter)))

        self.pipeline = Pipeline(steps)

        self.pipeline.fit(model.train, model.train_labels)

        val_preds = self.pipeline.predict(model.valid)
        val_accuracy = accuracy_score(model.valid_labels, val_preds)

        test_preds = self.pipeline.predict(model.test)
        test_accuracy = accuracy_score(model.test_labels, test_preds)
        report = classification_report(model.test_labels, test_preds, output_dict=True,zero_division=0)
        #conf_matrix = confusion_matrix(model.test_labels, test_preds)

        print(f"Experiment Model : {model.layer}")
        print(f"SVM Attack Results (PCA={self.use_pca}):")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy:       {test_accuracy:.4f}")

        return {
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            #"confusion_matrix": conf_matrix,
            #"predictions": test_preds,
            #"test_labels": model.test_labels
        }

    def cross_validate_attack(self, model: ModelExperiment) -> dict[str, Any]:

        data = model.data
        labels = model.labels
        groups = model.groups

        logo = LeaveOneGroupOut()

        steps = [('scaler', StandardScaler())]
        if self.use_pca:
            steps.append(('pca', PCA(n_components=self.n_components)))
        
        steps.append(("support-vector-machine", SVC(max_iter=self.max_iter)))

        fold_accuracies = []
        for train_idx, test_idx in logo.split(data, labels, groups):
            pipeline = Pipeline(steps)
            pipeline.fit(data[train_idx], labels[train_idx])
            preds = pipeline.predict(data[test_idx])
            fold_accuracies.append(accuracy_score(labels[test_idx], preds))

        max_acc = float(np.max(fold_accuracies)) if fold_accuracies else 0.0
        mean_acc = float(np.mean(fold_accuracies))

        print(f"Experiment Model : {model.layer}")
        print(f"SVM Attack Results (PCA={self.use_pca}):")
        print(f"Max CV Accuracy:  {max_acc:.4f}")
        print(f"Mean CV Accuracy: {mean_acc:.4f}")

        return {
            "fold_accuracies": fold_accuracies,
            "max_accuracy": max_acc,
            "mean_accuracy": mean_acc
        }

