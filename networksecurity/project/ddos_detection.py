from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DROP_COLUMNS = {
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Timestamp",
    "SimillarHTTP",
}
LABEL_COLUMN_CANDIDATES = ["Label", "label"]


@dataclass
class ExperimentArtifacts:
    train_path: str
    test_path: str
    dropped_columns: list[str]
    selected_features: list[str]


@dataclass
class ExperimentResults:
    artifacts: ExperimentArtifacts
    metrics: dict[str, dict[str, float]]
    reports: dict[str, dict[str, Any]]
    feature_importances: list[dict[str, float]]


@dataclass
class PreparedDataset:
    features: pd.DataFrame
    labels: pd.Series
    dropped_columns: list[str]


class NumpyCNNMLPClassifier(BaseEstimator, ClassifierMixin):
    """使用 NumPy 实现的轻量级 1D CNN-MLP 二分类器。"""

    def __init__(
        self,
        n_filters: int = 8,
        kernel_size: int = 3,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        epochs: int = 40,
        batch_size: int = 256,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_parameters(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.random_state)
        conv_output_len = input_dim - self.kernel_size + 1
        if conv_output_len <= 0:
            raise ValueError("kernel_size 不能大于特征数量")
        pooled_len = conv_output_len // 2
        if pooled_len <= 0:
            raise ValueError("卷积后的长度不足以进行最大池化，请减小 kernel_size")

        self.input_dim_ = input_dim
        self.conv_output_len_ = conv_output_len
        self.pooled_len_ = pooled_len
        self.flatten_dim_ = self.n_filters * pooled_len + input_dim

        self.conv_kernels_ = rng.normal(0, 0.1, size=(self.n_filters, self.kernel_size))
        self.conv_bias_ = np.zeros(self.n_filters)
        self.w1_ = rng.normal(0, 0.1, size=(self.flatten_dim_, self.hidden_dim))
        self.b1_ = np.zeros(self.hidden_dim)
        self.w2_ = rng.normal(0, 0.1, size=(self.hidden_dim, 1))
        self.b2_ = np.zeros(1)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _forward(self, x_batch: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        batch_size, input_dim = x_batch.shape
        windows = np.lib.stride_tricks.sliding_window_view(
            x_batch, window_shape=self.kernel_size, axis=1
        )
        conv_linear = np.tensordot(windows, self.conv_kernels_, axes=([2], [1]))
        conv_linear = np.transpose(conv_linear, (0, 2, 1)) + self.conv_bias_[None, :, None]
        conv_activated = np.maximum(conv_linear, 0.0)

        usable_len = self.pooled_len_ * 2
        conv_trimmed = conv_activated[:, :, :usable_len]
        pool_source = conv_trimmed.reshape(batch_size, self.n_filters, self.pooled_len_, 2)
        pool_indices = np.argmax(pool_source, axis=3)
        pooled = np.max(pool_source, axis=3)

        flattened = np.concatenate([pooled.reshape(batch_size, -1), x_batch], axis=1)
        hidden_linear = flattened @ self.w1_ + self.b1_
        hidden = np.maximum(hidden_linear, 0.0)
        logits = hidden @ self.w2_ + self.b2_
        probabilities = self._sigmoid(logits).reshape(-1)

        cache = {
            "x_batch": x_batch,
            "windows": windows,
            "conv_linear": conv_linear,
            "conv_activated": conv_activated,
            "pool_source": pool_source,
            "pool_indices": pool_indices,
            "pooled": pooled,
            "flattened": flattened,
            "hidden_linear": hidden_linear,
            "hidden": hidden,
            "probabilities": probabilities,
        }
        return probabilities, cache

    def _backward(self, cache: dict[str, np.ndarray], y_batch: np.ndarray) -> None:
        x_batch = cache["x_batch"]
        windows = cache["windows"]
        conv_linear = cache["conv_linear"]
        conv_activated = cache["conv_activated"]
        pool_indices = cache["pool_indices"]
        hidden_linear = cache["hidden_linear"]
        hidden = cache["hidden"]
        flattened = cache["flattened"]
        probabilities = cache["probabilities"]

        batch_size = len(y_batch)
        d_logits = (probabilities - y_batch).reshape(-1, 1) / batch_size
        grad_w2 = hidden.T @ d_logits
        grad_b2 = d_logits.sum(axis=0)

        d_hidden = d_logits @ self.w2_.T
        d_hidden[hidden_linear <= 0] = 0.0
        grad_w1 = flattened.T @ d_hidden
        grad_b1 = d_hidden.sum(axis=0)

        d_flattened = d_hidden @ self.w1_.T
        pooled_part = d_flattened[:, : self.n_filters * self.pooled_len_]
        d_pooled = pooled_part.reshape(batch_size, self.n_filters, self.pooled_len_)

        d_pool_source = np.zeros((batch_size, self.n_filters, self.pooled_len_, 2))
        rows = np.arange(batch_size)[:, None, None]
        filters = np.arange(self.n_filters)[None, :, None]
        pool_positions = np.arange(self.pooled_len_)[None, None, :]
        d_pool_source[rows, filters, pool_positions, pool_indices] = d_pooled
        d_conv_trimmed = d_pool_source.reshape(batch_size, self.n_filters, self.pooled_len_ * 2)

        d_conv_activated = np.zeros_like(conv_activated)
        d_conv_activated[:, :, : self.pooled_len_ * 2] = d_conv_trimmed
        d_conv_linear = d_conv_activated * (conv_linear > 0)

        grad_conv_bias = d_conv_linear.sum(axis=(0, 2))
        grad_conv_kernels = np.zeros_like(self.conv_kernels_)
        for filter_index in range(self.n_filters):
            grad_conv_kernels[filter_index] = np.tensordot(
                d_conv_linear[:, filter_index, :], windows, axes=([0, 1], [0, 1])
            )

        self.w2_ -= self.learning_rate * grad_w2
        self.b2_ -= self.learning_rate * grad_b2
        self.w1_ -= self.learning_rate * grad_w1
        self.b1_ -= self.learning_rate * grad_b1
        self.conv_kernels_ -= self.learning_rate * grad_conv_kernels
        self.conv_bias_ -= self.learning_rate * grad_conv_bias

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyCNNMLPClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self._initialize_parameters(x.shape[1])
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            indices = rng.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start in range(0, len(x_shuffled), self.batch_size):
                end = start + self.batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                probabilities, cache = self._forward(x_batch)
                self._backward(cache, y_batch)

            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                epoch_loss = self._binary_cross_entropy(y, self.predict_proba(x)[:, 1])
                print(f"[CNN-MLP] epoch={epoch + 1}, loss={epoch_loss:.6f}")

        self.classes_ = np.array([0, 1])
        return self

    def _binary_cross_entropy(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        probabilities, _ = self._forward(x)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def discover_dataset_files(dataset_dir: str | Path) -> tuple[Path, Path]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")

    thursday_candidates = sorted(dataset_path.glob("*Thursday*.csv"))
    friday_candidates = sorted(dataset_path.glob("*Friday*.csv"))

    if not thursday_candidates:
        raise FileNotFoundError("未找到 Thursday CSV 文件")
    if not friday_candidates:
        raise FileNotFoundError("未找到 Friday CSV 文件")

    return thursday_candidates[0], friday_candidates[0]


def _resolve_label_column(df: pd.DataFrame) -> str:
    for label_column in LABEL_COLUMN_CANDIDATES:
        if label_column in df.columns:
            return label_column
    raise KeyError("未找到标签列，预期为 Label 或 label")


def _drop_non_numeric_columns(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    columns_to_drop = sorted(
        {column for column in object_columns if column != label_column}.union(
            column for column in DROP_COLUMNS if column in df.columns and column != label_column
        )
    )
    return df.drop(columns=columns_to_drop, errors="ignore"), columns_to_drop


def load_and_prepare_datasets(train_path: str | Path, test_path: str | Path) -> tuple[PreparedDataset, PreparedDataset]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_column = _resolve_label_column(train_df)
    train_df, dropped_columns = _drop_non_numeric_columns(train_df, label_column)
    test_df, _ = _drop_non_numeric_columns(test_df, _resolve_label_column(test_df))

    test_label_column = _resolve_label_column(test_df)

    train_labels = (train_df[label_column].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    test_labels = (test_df[test_label_column].astype(str).str.strip().str.upper() != "BENIGN").astype(int)

    train_features = train_df.drop(columns=[label_column]).replace([np.inf, -np.inf], np.nan)
    test_features = test_df.drop(columns=[test_label_column]).replace([np.inf, -np.inf], np.nan)

    common_columns = sorted(set(train_features.columns).intersection(test_features.columns))
    if not common_columns:
        raise ValueError("训练集与测试集没有共同特征列")

    return (
        PreparedDataset(train_features[common_columns], train_labels, dropped_columns),
        PreparedDataset(test_features[common_columns], test_labels, dropped_columns),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


class CICDDoSDetector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_selector = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.models = {
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                learning_rate_init=0.001,
                batch_size=64,
                max_iter=80,
                early_stopping=True,
                random_state=random_state,
            ),
            "RF": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            "CNN-MLP": NumpyCNNMLPClassifier(
                n_filters=8,
                kernel_size=3,
                hidden_dim=32,
                learning_rate=0.005,
                epochs=40,
                batch_size=256,
                random_state=random_state,
            ),
        }

    def _select_top_features(
        self, x_train: pd.DataFrame, y_train: pd.Series, top_k: int = 20
    ) -> tuple[list[str], list[dict[str, float]]]:
        self.feature_selector.fit(x_train, y_train)
        importance_df = (
            pd.DataFrame({"feature": x_train.columns, "importance": self.feature_selector.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        selected_features = importance_df.head(top_k)["feature"].tolist()
        return selected_features, importance_df.to_dict(orient="records")

    def run(self, train_path: str | Path, test_path: str | Path) -> ExperimentResults:
        train_data, test_data = load_and_prepare_datasets(train_path, test_path)

        x_train = train_data.features.copy()
        x_test = test_data.features.copy()
        y_train = train_data.labels.to_numpy()
        y_test = test_data.labels.to_numpy()

        x_train_imputed = pd.DataFrame(
            self.imputer.fit_transform(x_train), columns=x_train.columns
        )
        x_test_imputed = pd.DataFrame(
            self.imputer.transform(x_test), columns=x_test.columns
        )

        selected_features, feature_importances = self._select_top_features(x_train_imputed, y_train, top_k=20)
        x_train_selected = x_train_imputed[selected_features]
        x_test_selected = x_test_imputed[selected_features]

        x_train_scaled = self.scaler.fit_transform(x_train_selected)
        x_test_scaled = self.scaler.transform(x_test_selected)

        metrics: dict[str, dict[str, float]] = {}
        reports: dict[str, dict[str, Any]] = {}

        for model_name, model in self.models.items():
            if model_name == "RF":
                train_features = x_train_selected.to_numpy()
                test_features = x_test_selected.to_numpy()
            else:
                train_features = x_train_scaled
                test_features = x_test_scaled

            model.fit(train_features, y_train)
            predictions = model.predict(test_features)
            metrics[model_name] = compute_metrics(y_test, predictions)
            reports[model_name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)

        return ExperimentResults(
            artifacts=ExperimentArtifacts(
                train_path=str(train_path),
                test_path=str(test_path),
                dropped_columns=train_data.dropped_columns,
                selected_features=selected_features,
            ),
            metrics=metrics,
            reports=reports,
            feature_importances=feature_importances,
        )


def save_results(results: ExperimentResults, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "ddos_experiment_results.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(asdict(results), file, ensure_ascii=False, indent=2)

    comparison_file = output_path / "model_comparison.csv"
    comparison_df = pd.DataFrame(results.metrics).T.sort_values("f1_score", ascending=False)
    comparison_df.to_csv(comparison_file, encoding="utf-8-sig")
    return results_file


def format_results(results: ExperimentResults) -> str:
    comparison_df = pd.DataFrame(results.metrics).T.sort_values("f1_score", ascending=False)
    lines = [
        "=" * 80,
        "基于数据流特征的 DDoS 攻击检测实验结果",
        "=" * 80,
        f"训练集: {results.artifacts.train_path}",
        f"测试集: {results.artifacts.test_path}",
        f"删除的非数值列: {', '.join(results.artifacts.dropped_columns) or '无'}",
        f"随机森林筛选的 Top-20 特征: {', '.join(results.artifacts.selected_features)}",
        "",
        "模型性能对比:",
        comparison_df.to_string(float_format=lambda x: f"{x:.4f}"),
        "",
    ]

    for model_name, report in results.reports.items():
        lines.append(f"[{model_name}] 分类报告:")
        report_df = pd.DataFrame(report).T
        lines.append(report_df.to_string(float_format=lambda x: f"{x:.4f}"))
        lines.append("")

    best_model = comparison_df.index[0]
    lines.append(
        f"综合 F1-score 最优模型: {best_model} (F1={comparison_df.iloc[0]['f1_score']:.4f})"
    )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIC-IDS2017 Thursday/Friday DDoS 检测实验")
    parser.add_argument("--dataset-dir", type=str, help="包含 Thursday 和 Friday CSV 的目录")
    parser.add_argument("--train-path", type=str, help="Thursday CSV 路径")
    parser.add_argument("--test-path", type=str, help="Friday CSV 路径")
    parser.add_argument("--output-dir", type=str, default="artifacts/ddos_detection", help="结果输出目录")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.dataset_dir:
        train_path, test_path = discover_dataset_files(args.dataset_dir)
    elif args.train_path and args.test_path:
        train_path, test_path = Path(args.train_path), Path(args.test_path)
    else:
        parser.error("请提供 --dataset-dir，或同时提供 --train-path 和 --test-path")

    detector = CICDDoSDetector()
    results = detector.run(train_path, test_path)
    save_results(results, args.output_dir)
    print(format_results(results))


if __name__ == "__main__":
    main()
