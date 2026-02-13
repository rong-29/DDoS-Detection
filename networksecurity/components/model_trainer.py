import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array_data, load_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.logging.logger import logging


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def _build_models(self):
        return {
            "RandomForest": (
                RandomForestClassifier(random_state=42, class_weight="balanced"),
                {"n_estimators": [150, 250], "max_depth": [10, 20, None]},
            ),
            "SVM": (
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(class_weight="balanced")),
                ]),
                {"model__C": [1, 10], "model__kernel": ["rbf", "linear"]},
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                {"max_depth": [8, 16, None], "min_samples_split": [2, 5]},
            ),
        }

    def _save_comparison_plot(self, result_df: pd.DataFrame, save_dir: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(result_df["model"], result_df["test_accuracy"], color=["#3b82f6", "#22c55e", "#f97316"])
        ax.set_ylim(0, 1)
        ax.set_title("CIC-IDS2017 DDoS检测模型准确率对比")
        ax.set_ylabel("Test Accuracy")
        for i, value in enumerate(result_df["test_accuracy"]):
            ax.text(i, value + 0.01, f"{value:.4f}", ha="center")
        plot_path = os.path.join(save_dir, "model_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = self._build_models()
            results = []
            best_model_name = None
            best_model = None
            best_cv_score = -1

            for model_name, (model, params) in models.items():
                logging.info(f"开始训练模型: {model_name}")
                search = GridSearchCV(model, params, cv=3, n_jobs=-1, scoring="f1", verbose=1)
                search.fit(x_train, y_train)

                y_pred_test = search.best_estimator_.predict(x_test)
                y_pred_train = search.best_estimator_.predict(x_train)
                test_acc = accuracy_score(y_test, y_pred_test)

                results.append({
                    "model": model_name,
                    "best_params": search.best_params_,
                    "cv_f1": float(search.best_score_),
                    "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
                    "test_accuracy": float(test_acc),
                })

                if search.best_score_ > best_cv_score:
                    best_cv_score = float(search.best_score_)
                    best_model_name = model_name
                    best_model = search.best_estimator_

            if best_cv_score < self.model_trainer_config.expected_accuracy:
                raise ValueError(f"最佳模型CV F1 {best_cv_score:.4f} 低于阈值 {self.model_trainer_config.expected_accuracy}")

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            os.makedirs("final_models", exist_ok=True)
            save_object(os.path.join("final_models", "model.pkl"), best_model)

            result_df = pd.DataFrame(results).sort_values(by="cv_f1", ascending=False)
            summary_json_path = os.path.join(os.path.dirname(self.model_trainer_config.trained_model_file_path), "model_comparison.json")
            result_df.to_json(summary_json_path, orient="records", force_ascii=False, indent=2)
            plot_path = self._save_comparison_plot(result_df, os.path.dirname(self.model_trainer_config.trained_model_file_path))

            logging.info(f"最佳模型: {best_model_name}, CV F1={best_cv_score:.4f}")
            logging.info(f"模型对比结果文件: {summary_json_path}")
            logging.info(f"可视化图片: {plot_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            return self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
