from scipy.stats import ks_2samp
import pandas as pd
import os
import sys

from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
        """将 CIC-IDS2017 的标签统一到 Label 列，Benign->0，其余攻击->1。"""
        label_candidates = ["Label", "label", "Result"]
        found = next((c for c in label_candidates if c in df.columns), None)
        if not found:
            raise ValueError(f"未找到标签列，候选列: {label_candidates}")

        if found != TARGET_COLUMN:
            df = df.rename(columns={found: TARGET_COLUMN})

        label_series = df[TARGET_COLUMN]
        if label_series.dtype == object:
            normalized = label_series.astype(str).str.strip().str.lower().map(lambda x: 0 if x == "benign" else 1)
        else:
            normalized = label_series.replace({-1: 0})
            normalized = (normalized != 0).astype(int)

        df[TARGET_COLUMN] = normalized
        return df

    def is_required_columns_exists(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = [list(item.keys())[0] for item in self.schema_config["columns"]]
            missing_columns = [col for col in expected_columns if col not in dataframe.columns]
            unexpected_columns = [col for col in dataframe.columns if col not in expected_columns]
            if missing_columns:
                logging.error(f"缺失字段: {missing_columns}")
                return False
            if unexpected_columns:
                logging.error(f"存在非schema字段: {unexpected_columns}")
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            common_cols = [c for c in base_df.columns if c in current_df.columns and c != TARGET_COLUMN]
            for column in common_cols:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                same_distribution = bool(is_same_dist.pvalue > threshold)
                if not same_distribution:
                    status = False
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "same_distribution": same_distribution,
                }

            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report)
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("开始数据验证流程")
            train_dataframe = self.normalize_label_column(self.read_data(self.data_ingestion_artifact.train_file_path))
            test_dataframe = self.normalize_label_column(self.read_data(self.data_ingestion_artifact.test_file_path))

            if not self.is_required_columns_exists(train_dataframe):
                raise ValueError("训练集字段与 schema 不匹配")
            if not self.is_required_columns_exists(test_dataframe):
                raise ValueError("测试集字段与 schema 不匹配")

            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            if not drift_status:
                logging.warning("检测到数据漂移，但继续执行训练流程。")

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
