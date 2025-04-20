# 中文文本纠错系统 (Chinese Text Correction System)

该项目实现了一个完整的中文文本纠错系统，包括基于规则、统计和神经网络的多种纠错方法。系统能够检测并纠正中文文本中的错别字、语法错误、标点符号错误等问题。

## 项目特点

- 多种纠错方法实现：规则、统计和BERT-LSTM神经网络模型
- 完整的数据分析和可视化模块
- 详细的评估指标计算
- 交互式文本纠错界面
- 支持批量测试和单句纠错

## 环境需求

本系统在以下环境中开发和测试：
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (可选，用于GPU加速)

## 快速安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd src
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
src/
├── data/                      # 数据目录
│   ├── train.jsonl            # 训练数据
│   └── test.jsonl             # 测试数据
├── models/                    # 模型保存目录
│   └── bert_lstm_model_*.pt   # 已训练的BERT-LSTM模型
├── analysis_results/          # 数据分析结果保存目录
├── main.py                    # 主程序入口
├── test_single_sentence.py    # 单句测试交互式界面
├── run_analysis.py            # 运行数据分析
├── data_analysis.py           # 数据分析模块
├── rule_based.py              # 基于规则的纠错器
├── statistical.py             # 基于统计的纠错器
├── bert_lstm_corrector.py     # BERT-LSTM神经网络纠错器
├── evaluation.py              # 评估模块
└── requirements.txt           # 依赖包列表
```

## 数据说明

项目已包含训练和测试数据集，位于`data/`目录下：
- `train.jsonl`：训练数据集，约2160条样本
- `test.jsonl`：测试数据集，约2312条样本

数据使用JSONL格式，每行包含一个JSON对象，格式如下：
```json
{"source": "原始文本", "target": "正确文本", "label": 1}
```
- source: 原始文本，可能包含错误
- target: 正确文本，人工校对后的版本
- label: 标记是否有错误，1表示有错误，0表示无错误

## 预训练模型

项目已包含预训练的BERT-LSTM模型，位于`models/`目录下：
- `bert_lstm_model_final_*.pt`：已训练好的BERT-LSTM模型

您可以直接使用这些模型进行推理，无需重新训练：
```bash
python main.py --method bert_lstm --skip_training --load_bert_model bert_lstm_model_final_20250420_164745
```

或者用于单句纠错：
```bash
python test_single_sentence.py
```

## 使用方法

### 1. 数据分析

运行以下命令分析训练和测试数据集的错误分布：

```bash
python run_analysis.py --train_file data/train.jsonl --test_file data/test.jsonl --analyze_set both --output_dir analysis_results
```

### 2. 训练和评估模型

运行以下命令训练和评估纠错模型：

```bash
# 基于规则的方法
python main.py --method rule --train_file data/train.jsonl --test_file data/test.jsonl

# 基于统计的方法
python main.py --method statistical --train_file data/train.jsonl --test_file data/test.jsonl

# BERT-LSTM神经网络模型（建议使用GPU）
python main.py --method bert_lstm --train_file data/train.jsonl --test_file data/test.jsonl --bert_epochs 3 --bert_batch_size 16
```

可选参数：
- `--eval_mode`：选择评估模式，可选值为 'dev', 'test', 'both'
- `--debug`：启用调试模式，输出更详细的信息
- `--skip_training`：跳过训练过程，直接使用现有模型进行推理
- `--load_bert_model`：指定加载预训练BERT模型的路径

### 3. 单句纠错（交互式界面）

运行以下命令启动交互式文本纠错界面：

```bash
python test_single_sentence.py --threshold 0.1
```

如果要使用已训练好的模型：
```bash
python test_single_sentence.py --model_path models/bert_lstm_model_final_20250420_164745.pt --threshold 0.1
```

在交互模式下，您可以：
- 输入中文文本，系统会自动纠正并显示修改
- 调整错误检测阈值（`threshold 0.2`）
- 查看历史记录（`history`）
- 清屏（`clear`）
- 退出程序（`exit`或`quit`或`q`）

对单句进行纠错：

```bash
python test_single_sentence.py --single --sentence "我昨天去公园里玩的很开心。"
```

## 模型说明

### 1. 规则纠错器 (RuleBasedCorrector)

基于人工定义的规则进行文本纠错，包括：
- 混淆字符对的检测和替换
- 标点符号规则
- 语法规则
- 成语纠正
- 冗余词修正
- 词序错误修正

### 2. 统计纠错器 (StatisticalCorrector)

基于统计方法进行文本纠错，包括：
- N-gram语言模型
- 混淆矩阵
- 编辑距离计算
- 机器学习特征提取

### 3. BERT-LSTM神经网络纠错器 (BertLSTMCorrector)

基于预训练BERT和LSTM网络的纠错模型：
- 利用BERT提取上下文特征
- LSTM处理序列信息
- 字符级别的错误检测和纠正
- 细粒度的错误类型识别

## 评估指标

系统使用以下指标评估纠错性能：
- 准确率 (Accuracy)：完全正确预测的样本比例
- 字符准确率 (Character Accuracy)：正确预测的字符比例
- 检测性能 (Detection)：发现错误的精确率、召回率、F1值和F0.5值
- 纠正性能 (Correction)：纠正错误的精确率、召回率、F1值和F0.5值

最终评分使用纠正的F0.5值，该指标更重视精确率而非召回率。

## 运行示例

以下是一些运行示例及其预期输出：

1. 使用规则方法进行纠错：
```
python main.py --method rule
```
预期输出：
```
Loading data...
Train set: 1944 samples
Validation set: 216 samples
Test set: 2312 samples

Initializing rule-based corrector...
Evaluating on test data...
100%|██████████| 2312/2312 [00:08<00:00, 267.43it/s]

Test set results:
========== Chinese Text Correction Evaluation Results ==========

Sample-level Evaluation:
Accuracy: 0.7384
Character Accuracy: 0.9857

Detection Evaluation:
Precision: 0.6841
Recall: 0.3162
F1 Score: 0.4326
F0.5 Score: 0.5632

Correction Evaluation:
Precision: 0.6841
Recall: 0.3162
F1 Score: 0.4326
F0.5 Score: 0.5632

Final Score:
F0.5 Score: 0.5632
=============================================
```

2. 使用BERT-LSTM模型进行单句纠错：
```
python test_single_sentence.py --single --sentence "我昨天去公园里玩的很开心。" --model_path models/bert_lstm_model_final_20250420_164745.pt
```
预期输出：
```
初始化BERT-LSTM模型...
加载模型: models/bert_lstm_model_final_20250420_164745
原文: 我昨天去公园里玩的很开心。
纠正: 我昨天去公园里玩得很开心。
修改: 位置 9: '的' -> '得'
耗时: 0.3245 秒
```

## 问题排查

1. **CUDA相关错误**：
   - 检查CUDA版本是否与PyTorch兼容
   - 如果无法使用GPU，可以降低batch size并使用CPU运行

2. **内存不足**：
   - 减小batch size
   - 降低最大序列长度（max_seq_len）
   - 减少训练epochs数量

3. **缺少依赖包**：
   - 确保已安装所有requirements.txt中列出的依赖
   - 某些可视化特性需要matplotlib，如果不需要可视化可以忽略相关警告

## 性能与资源要求

- **CPU模式**：所有模型都可以在CPU上运行，但BERT-LSTM模型运行速度较慢
- **GPU模式**：BERT-LSTM模型在GPU上可获得显著加速，推荐使用CUDA支持的GPU
- **内存要求**：
  - 规则和统计模型：≥4GB RAM
  - BERT-LSTM模型：≥8GB RAM (训练时推荐≥16GB)
- **磁盘空间**：
  - 基本安装：约100MB
  - BERT模型：约400-700MB
  - 训练数据：约2MB
  - 训练后模型：约630MB/模型 