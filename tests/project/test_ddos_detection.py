import numpy as np
import pandas as pd

from networksecurity.project.ddos_detection import (
    CICDDoSDetector,
    NumpyCNNMLPClassifier,
    compute_metrics,
    load_and_prepare_datasets,
)


def test_load_and_prepare_datasets(tmp_path):
    train_df = pd.DataFrame(
        {
            "Flow ID": ["a", "b", "c"],
            "Source IP": ["1.1.1.1", "1.1.1.2", "1.1.1.3"],
            "F1": [1.0, 2.0, np.inf],
            "F2": [3.0, np.nan, 5.0],
            "Label": ["BENIGN", "DoS Hulk", "PortScan"],
        }
    )
    test_df = pd.DataFrame(
        {
            "Flow ID": ["d", "e"],
            "Source IP": ["2.2.2.1", "2.2.2.2"],
            "F1": [4.0, 5.0],
            "F2": [6.0, 7.0],
            "Label": ["BENIGN", "DDoS"],
        }
    )
    train_path = tmp_path / "Thursday-WorkingHours.csv"
    test_path = tmp_path / "Friday-WorkingHours.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_data, test_data = load_and_prepare_datasets(train_path, test_path)

    assert train_data.dropped_columns == ["Flow ID", "Source IP"]
    assert list(train_data.features.columns) == ["F1", "F2"]
    assert train_data.labels.tolist() == [0, 1, 1]
    assert test_data.labels.tolist() == [0, 1]
    assert not np.isnan(train_data.features.iloc[0, 0])
    assert np.isnan(train_data.features.iloc[2, 0])



def test_numpy_cnn_mlp_classifier_learns_simple_pattern():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 20))
    y = ((x[:, :5].sum(axis=1) + 0.5 * x[:, 5:10].sum(axis=1)) > 0).astype(int)

    model = NumpyCNNMLPClassifier(epochs=40, batch_size=32, learning_rate=0.01, random_state=0)
    model.fit(x, y)
    predictions = model.predict(x)
    metrics = compute_metrics(y, predictions)

    assert metrics["accuracy"] >= 0.74
    assert metrics["f1_score"] >= 0.68



def test_detector_runs_end_to_end_with_synthetic_data(tmp_path):
    feature_names = [f"F{i}" for i in range(1, 31)]
    rng = np.random.default_rng(1)

    def build_frame(size: int, attack_shift: float) -> pd.DataFrame:
        benign = rng.normal(0, 1, size=(size // 2, len(feature_names)))
        attack = rng.normal(attack_shift, 1, size=(size - size // 2, len(feature_names)))
        features = np.vstack([benign, attack])
        labels = np.array(["BENIGN"] * (size // 2) + ["DDoS"] * (size - size // 2))
        df = pd.DataFrame(features, columns=feature_names)
        df.insert(0, "Flow ID", [f"id_{i}" for i in range(size)])
        df.insert(1, "Source IP", [f"10.0.0.{i}" for i in range(size)])
        df["Label"] = labels
        return df

    train_path = tmp_path / "Thursday.csv"
    test_path = tmp_path / "Friday.csv"
    build_frame(120, 1.0).to_csv(train_path, index=False)
    build_frame(80, 1.0).to_csv(test_path, index=False)

    detector = CICDDoSDetector(random_state=0)
    results = detector.run(train_path, test_path)

    assert len(results.artifacts.selected_features) == 20
    assert {"MLP", "RF", "CNN-MLP"} == set(results.metrics)
    assert results.metrics["RF"]["accuracy"] >= 0.7
