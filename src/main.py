#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for Chinese Text Correction task.
This script provides a framework for analyzing and correcting errors in Chinese text.
"""

import os
import json
import argparse
import traceback
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Import modules
from data_analysis import analyze_data, visualize_error_distribution
from rule_based import RuleBasedCorrector
from statistical import StatisticalCorrector
from evaluation import evaluate_performance, print_detailed_metrics

# 设置更长的超时时间，解决网络问题
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '500'  # 设置为500秒

# 导入新的BERT-LSTM纠错器
try:
    from bert_lstm_corrector import BertLSTMCorrector
    BERT_LSTM_AVAILABLE = True
except ImportError:
    BERT_LSTM_AVAILABLE = False
    print("Warning: BERT-LSTM corrector not available. Some features will be disabled.")


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from jsonl file.

    Args:
        file_path: Path to the jsonl file.

    Returns:
        List of dictionaries containing the data.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def main():
    """
    Main function to run the Chinese text correction pipeline.
    """
    parser = argparse.ArgumentParser(description='Chinese Text Correction')
    parser.add_argument('--train_file', type=str, default='data/train.jsonl', help='Path to training data')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl', help='Path to test data')
    parser.add_argument(
        '--method',
        type=str,
        choices=['rule', 'statistical', 'ensemble', 'bert_lstm'],
        default='statistical',
        help='Correction method to use',
    )
    parser.add_argument('--analyze', action='store_true', help='Perform data analysis')
    parser.add_argument('--statistical_method', type=str, default='ngram', help='Statistical method to use')
    
    # BERT相关参数
    parser.add_argument('--local_bert_path', type=str, default='../bert', 
                       help='Local path to BERT model files')
    parser.add_argument('--bert_epochs', type=int, default=3, help='Number of epochs for BERT training')
    parser.add_argument('--bert_batch_size', type=int, default=16, help='Batch size for BERT training')
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5, help='Learning rate for BERT')
    parser.add_argument('--load_bert_model', type=str, default='', 
                       help='Path to pretrained BERT model to load')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip training and use model directly for inference (only with --local_bert_path)')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run inference only mode (requires --local_bert_path or --load_bert_model)')
    
    # 模型评估操作
    parser.add_argument('--eval_mode', choices=['dev', 'test', 'both'], default='test',
                       help='Evaluation mode: dev for validation set, test for test set, both for both')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()

    # 如果启用调试模式，打印更多信息
    if args.debug:
        print("Debug mode enabled")
        print(f"Arguments: {args}")
        # 检查本地模型路径是否存在
        if args.local_bert_path:
            if os.path.exists(args.local_bert_path):
                print(f"Local model path exists: {args.local_bert_path}")
                # 检查必要的文件
                required_files = ['config.json', 'vocab.txt', 'pytorch_model.bin']
                for file in required_files:
                    file_path = os.path.join(args.local_bert_path, file)
                    if os.path.exists(file_path):
                        print(f"  - {file}: Found ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")
                    else:
                        print(f"  - {file}: Missing")
            else:
                print(f"Local model path does not exist: {args.local_bert_path}")
    
    try:
        # Load data
        print("Loading data...")
        train_data = load_data(args.train_file)
        test_data = load_data(args.test_file)
        
        # 创建验证集（从训练集中分割）
        split_idx = int(len(train_data) * 0.9)
        train_subset = train_data[:split_idx]
        val_subset = train_data[split_idx:]
        
        if args.debug:
            print(f"Train set: {len(train_subset)} samples")
            print(f"Validation set: {len(val_subset)} samples")
            print(f"Test set: {len(test_data)} samples")

        # Data analysis
        if args.analyze:
            print("\nPerforming data analysis...")
            analysis_results = analyze_data(train_data)
            visualize_error_distribution(analysis_results)

        # Initialize corrector based on method
        if args.method == 'rule':
            print("\nInitializing rule-based corrector...")
            corrector = RuleBasedCorrector()
            corrector.train(train_data)
        elif args.method == 'statistical':
            print("\nInitializing statistical corrector...")
            corrector = StatisticalCorrector(args.statistical_method)
            corrector.train(train_data)
        elif args.method == 'bert_lstm':
            if not BERT_LSTM_AVAILABLE:
                print("BERT-LSTM method not available. Falling back to statistical method.")
                corrector = StatisticalCorrector('ml')
                corrector.train(train_data)
            else:
                print("\nInitializing BERT-LSTM corrector...")
                local_model_path = args.local_bert_path if args.local_bert_path else None
                
                # 显示模型路径信息
                if args.debug and local_model_path:
                    print(f"Using local model path for BERT-LSTM: {os.path.abspath(local_model_path)}")
                
                try:
                    bert_lstm_corrector = BertLSTMCorrector(
                        local_model_path=local_model_path,
                        batch_size=args.bert_batch_size,
                        learning_rate=args.bert_learning_rate,
                        epochs=args.bert_epochs
                    )
                    
                    # 尝试加载预训练模型
                    if args.load_bert_model:
                        print(f"Loading pretrained BERT-LSTM model from {args.load_bert_model}...")
                        if not bert_lstm_corrector.load_model(args.load_bert_model):
                            print(f"Failed to load model from {args.load_bert_model}, falling back to training mode")
                            args.skip_training = False
                    
                    # 判断是否需要训练
                    if not (args.skip_training or args.inference_only):
                        # 训练模型
                        print("Training BERT-LSTM model...")
                        
                        try:
                            bert_lstm_corrector.train(
                                train_subset, 
                                val_subset, 
                                epochs=args.bert_epochs, 
                                batch_size=args.bert_batch_size,
                                learning_rate=args.bert_learning_rate
                            )
                        except Exception as e:
                            print(f"Error during BERT-LSTM training: {e}")
                            if args.debug:
                                traceback.print_exc()
                            print("Using model without fine-tuning...")
                    else:
                        print("Skipping training as requested. Using model directly for inference.")
                    
                    corrector = bert_lstm_corrector
                except Exception as e:
                    print(f"Error initializing BERT-LSTM corrector: {e}")
                    if args.debug:
                        traceback.print_exc()
                    print("Falling back to statistical method...")
                    corrector = StatisticalCorrector('ml')
                    corrector.train(train_data)
        elif args.method == 'ensemble':
            print("\nInitializing ensemble corrector...")
            # 组合规则方法和统计方法
            rule_corrector = RuleBasedCorrector()
            rule_corrector.train(train_data)

            stat_corrector = StatisticalCorrector()
            stat_corrector.train(train_data)
            
            # 创建一个组合器
            class EnsembleCorrector:
                def __init__(self, rule_corrector, stat_corrector):
                    self.rule_corrector = rule_corrector
                    self.stat_corrector = stat_corrector
                    
                def correct(self, text):
                    # 先用规则方法纠正
                    rule_corrected = self.rule_corrector.correct(text)
                    # 然后用统计方法进一步纠正
                    final_corrected = self.stat_corrector.correct(rule_corrected)
                    return final_corrected
            
            corrector = EnsembleCorrector(rule_corrector, stat_corrector)

        # 评估，根据模式选择评估集
        if args.eval_mode in ['dev', 'both']:
            print("\nEvaluating on validation data...")
            val_predictions = []
            for sample in tqdm(val_subset, ncols=100):
                source = sample['source']
                try:
                    corrected = corrector.correct(source)
                    val_predictions.append(
                        {'source': source, 'prediction': corrected, 'target': sample['target'], 'label': sample['label']}
                    )
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    if args.debug:
                        print(f"Source: {source}")
                        traceback.print_exc()
                    val_predictions.append(
                        {'source': source, 'prediction': source, 'target': sample['target'], 'label': sample['label']}
                    )

            print("\nValidation set results:")
            val_metrics = evaluate_performance(val_predictions)
            print_detailed_metrics(val_metrics)
        
        if args.eval_mode in ['test', 'both']:
            # Evaluate on test data
            print("\nEvaluating on test data...")
            test_predictions = []
            for sample in tqdm(test_data, ncols=100):
                source = sample['source']
                try:
                    corrected = corrector.correct(source)
                    test_predictions.append(
                        {'source': source, 'prediction': corrected, 'target': sample['target'], 'label': sample['label']}
                    )
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    if args.debug:
                        print(f"Source: {source}")
                        traceback.print_exc()
                    test_predictions.append(
                        {'source': source, 'prediction': source, 'target': sample['target'], 'label': sample['label']}
                    )

            print("\nTest set results:")
            test_metrics = evaluate_performance(test_predictions)
            print_detailed_metrics(test_metrics)

    except Exception as e:
        print(f"An error occurred: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
