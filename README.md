# 基于数据流特征的 DDoS 攻击检测系统（CIC-IDS2017）

本项目在保留原有训练/验证/预测流水线框架的基础上，聚焦于 **CIC-IDS2017 数据集** 的流特征，完成 DDoS 攻击检测算法设计与实现。

## 主要改造点

- 使用 CIC-IDS2017 的数据流特征（Flow-based features）。
- 标签统一为二分类：`Benign=0`，其余攻击流量=`1`。
- 模型对比仅保留三种机器学习算法：
  - **Random Forest（主模型）**
  - **SVM（对比）**
  - **Decision Tree（对比）**
- 训练后自动输出模型对比结果：
  - `model_comparison.json`（结构化评估结果）
  - `model_comparison.png`（可视化柱状图）

## 数据要求

将 CIC-IDS2017 CSV 文件放到以下任一路径（按优先级查找）：

1. `Network_Data/cic_ids2017.csv`
2. `Network_Data/CIC-IDS2017.csv`
3. `Network_Data/cicids2017.csv`

## 快速运行

```bash
pip install -r requirements.txt
python test_app.py
```

触发训练接口：`POST /api/train`

## 训练产物

训练完成后可在最新 `Artifacts/<timestamp>/model_trainer/` 下看到：

- `model.pkl`：最佳模型封装
- `model_comparison.json`：三模型对比指标
- `model_comparison.png`：对比可视化图

同时在 `final_models/` 下保存：

- `model.pkl`
- `preprocessor.pkl`
