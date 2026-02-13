from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def _find_local_dataset_path() -> str:
        candidates = [
            "Network_Data/cic_ids2017.csv",
            "Network_Data/CIC-IDS2017.csv",
            "Network_Data/cicids2017.csv",
            "Network_Data/phisingData.csv",  # 兼容历史数据
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"未找到本地数据集，请将 CIC-IDS2017 CSV 放到 {candidates[0]}")

    def export_collection_as_dataframe(self):
        """优先从 MongoDB 读取；失败则回退到本地 CIC-IDS2017 CSV。"""
        try:
            mongo_uri = MONGO_DB_URL.strip() if MONGO_DB_URL and MONGO_DB_URL.strip() else "mongodb://localhost:27017"
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if not df.empty:
                if "_id" in df.columns:
                    df = df.drop(columns=["_id"], axis=1)
                df.replace({"na": np.nan}, inplace=True)
                logging.info(f"成功从 MongoDB 导出 {len(df)} 条记录")
                return df

            logging.warning("MongoDB 中没有数据，切换到本地 CSV。")
            csv_path = self._find_local_dataset_path()
            df = pd.read_csv(csv_path)
            df.replace({"na": np.nan}, inplace=True)
            logging.info(f"成功从本地 CSV 读取 {len(df)} 条记录: {csv_path}")
            return df

        except (pymongo.errors.ServerSelectionTimeoutError, pymongo.errors.ConnectionFailure) as e:
            logging.warning(f"MongoDB 连接失败: {e}，切换到本地 CSV")
            csv_path = self._find_local_dataset_path()
            df = pd.read_csv(csv_path)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["Label"] if "Label" in dataframe.columns else None,
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("训练集/测试集切分并保存完成")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
