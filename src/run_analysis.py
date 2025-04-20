#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run data analysis and visualization for Chinese Text Correction task.
"""

import os
import json
import argparse
import glob
from typing import Dict, List, Any

from data_analysis import analyze_data, visualize_error_distribution


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


def find_data_files():
    """查找数据目录中的jsonl文件"""
    data_dir = 'data'
    jsonl_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    
    train_file = None
    test_file = None
    
    for file in jsonl_files:
        if 'train' in file.lower():
            train_file = file
        elif 'test' in file.lower():
            test_file = file
    
    if not train_file:
        train_file = jsonl_files[0] if jsonl_files else 'data/train.jsonl'
    if not test_file:
        test_file = jsonl_files[-1] if len(jsonl_files) > 1 else 'data/test.jsonl'
    
    return train_file, test_file


def main():
    # 自动查找数据文件
    default_train_file, default_test_file = find_data_files()
    
    parser = argparse.ArgumentParser(description='Data Analysis for Chinese Text Correction')
    parser.add_argument('--train_file', type=str, default=default_train_file, 
                        help='Path to training data')
    parser.add_argument('--test_file', type=str, default=default_test_file, 
                        help='Path to test data')
    parser.add_argument('--analyze_set', type=str, choices=['train', 'test', 'both'], default='train',
                        help='Which dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    args = parser.parse_args()
    
    # 打印找到的数据文件
    print(f"使用训练集文件: {args.train_file}")
    print(f"使用测试集文件: {args.test_file}")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 根据用户选择加载数据
    if args.analyze_set in ['train', 'both']:
        print(f"Loading training data from {args.train_file}...")
        train_data = load_data(args.train_file)
        print(f"Loaded {len(train_data)} training samples.")
        
        print("\nAnalyzing training data...")
        train_analysis = analyze_data(train_data)
        
        # 保存分析结果
        train_output_path = os.path.join(args.output_dir, 'train_analysis.json')
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_analysis, f, ensure_ascii=False, indent=2)
        print(f"Training data analysis saved to {train_output_path}")
        
        # 可视化
        print("\nVisualizing training data analysis...")
        train_plot_path = os.path.join(args.output_dir, 'train_error_distribution.png')
        visualize_error_distribution(train_analysis, output_path=train_plot_path)
    
    if args.analyze_set in ['test', 'both']:
        print(f"\nLoading test data from {args.test_file}...")
        test_data = load_data(args.test_file)
        print(f"Loaded {len(test_data)} test samples.")
        
        print("\nAnalyzing test data...")
        test_analysis = analyze_data(test_data)
        
        # 保存分析结果
        test_output_path = os.path.join(args.output_dir, 'test_analysis.json')
        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump(test_analysis, f, ensure_ascii=False, indent=2)
        print(f"Test data analysis saved to {test_output_path}")
        
        # 可视化
        print("\nVisualizing test data analysis...")
        test_plot_path = os.path.join(args.output_dir, 'test_error_distribution.png')
        visualize_error_distribution(test_analysis, output_path=test_plot_path)
    
    print("\nData analysis complete!")


if __name__ == "__main__":
    main() 