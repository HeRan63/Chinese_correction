#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data analysis module for Chinese Text Correction task.
This module provides functions for analyzing error patterns in the dataset.
"""

import re
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import jieba

# Try to import optional dependencies for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    
    # 使用最简单的字体设置
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization features will be disabled.")


def analyze_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the dataset to extract statistics and error patterns.

    Args:
        data: List of dictionaries containing the data.

    Returns:
        Dictionary containing analysis results.
    """
    results = {
        'total_samples': len(data),
        'error_types': defaultdict(int),
        'error_positions': defaultdict(int),
        'common_errors': defaultdict(int),
        'sentence_lengths': [],
        'error_counts': [],
        'char_error_rate': [],
        'sentences_with_errors': 0
    }
    
    for sample in data:
        source = sample['source']
        target = sample['target']
        
        # 记录句子长度
        results['sentence_lengths'].append(len(source))
        
        # 如果有错误
        if source != target:
            results['sentences_with_errors'] += 1
            
            # 分析错误类型和位置
            error_count = 0
            for i, (s_char, t_char) in enumerate(zip(source, target)):
                if s_char != t_char:
                    error_count += 1
                    # 错误位置（句子前部、中部、后部）
                    position = "前部" if i < len(source) * 0.33 else "中部" if i < len(source) * 0.66 else "后部"
                    results['error_positions'][position] += 1
                    
                    # 记录常见错误对
                    error_pair = f"{s_char}→{t_char}"
                    results['common_errors'][error_pair] += 1
                    
                    # 尝试判断错误类型
                    if s_char == '' and t_char != '':
                        results['error_types']['漏字'] += 1
                    elif s_char != '' and t_char == '':
                        results['error_types']['多字'] += 1
                    else:
                        # 简单区分音近、形近
                        # (更完善的区分需要使用拼音和字形特征)
                        results['error_types']['替换'] += 1
            
            # 记录每个句子的错误数
            results['error_counts'].append(error_count)
            
            # 计算字符错误率（CER）
            results['char_error_rate'].append(error_count / len(source))
    
    # 计算统计量
    results['error_rate'] = results['sentences_with_errors'] / results['total_samples']
    if results['error_counts']:
        results['avg_errors_per_sentence'] = np.mean(results['error_counts'])
        results['avg_char_error_rate'] = np.mean(results['char_error_rate'])
    else:
        results['avg_errors_per_sentence'] = 0
        results['avg_char_error_rate'] = 0
    
    # 统计词语错误
    results['word_errors'] = analyze_word_errors(data)
    
    # 将defaultdicts转换为普通dicts，方便序列化
    results['error_types'] = dict(results['error_types'])
    results['error_positions'] = dict(results['error_positions'])
    results['common_errors'] = dict(sorted(results['common_errors'].items(), 
                                          key=lambda x: x[1], reverse=True)[:20])
    
    return results


def analyze_word_errors(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """分析词级别的错误"""
    word_errors = defaultdict(int)
    
    for sample in data:
        source = sample['source']
        target = sample['target']
        
        if source != target:
            # 对源文本和目标文本进行分词
            source_words = list(jieba.cut(source))
            target_words = list(jieba.cut(target))
            
            # 简单比较分词结果的差异
            source_set = set(source_words)
            target_set = set(target_words)
            
            # 记录源文本中有但目标文本中没有的词
            for word in source_set - target_set:
                if len(word) > 1:  # 只关注多字词
                    word_errors[f"{word}(错)"] += 1
            
            # 记录目标文本中有但源文本中没有的词
            for word in target_set - source_set:
                if len(word) > 1:  # 只关注多字词
                    word_errors[f"{word}(正)"] += 1
    
    # 返回前20个最常见的词语错误
    return dict(sorted(word_errors.items(), key=lambda x: x[1], reverse=True)[:20])


def visualize_error_distribution(analysis_results: Dict[str, Any], output_path=None) -> None:
    """
    Visualize the error distribution from analysis results.

    Args:
        analysis_results: Dictionary containing analysis results.
        output_path: Optional path to save the visualization image.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize results: matplotlib not available.")
        return
    
    # 创建翻译映射，用于中英文对照
    translations = {
        '漏字': 'Missing Character',
        '多字': 'Extra Character',
        '替换': 'Substitution',
        '前部': 'Front',
        '中部': 'Middle',
        '后部': 'End'
    }
    
    # 创建一个新的图形，包含多个子图
    plt.figure(figsize=(18, 15))
    
    # 1. 错误类型分布
    plt.subplot(2, 3, 1)
    error_types = analysis_results['error_types']
    if error_types:
        # 转换标签为英文
        labels = [translations.get(k, k) for k in error_types.keys()]
        values = list(error_types.values())
        plt.pie(values, labels=labels, autopct='%1.1f%%')
        plt.title('Error Type Distribution')
    else:
        plt.text(0.5, 0.5, 'No error data', horizontalalignment='center', verticalalignment='center')
    
    # 2. 错误位置分布
    plt.subplot(2, 3, 2)
    error_positions = analysis_results['error_positions']
    if error_positions:
        # 转换标签为英文
        positions = {translations.get(k, k): v for k, v in error_positions.items()}
        plt.bar(positions.keys(), positions.values())
        plt.title('Error Position Distribution')
        plt.ylabel('Error Count')
    else:
        plt.text(0.5, 0.5, 'No position data', horizontalalignment='center', verticalalignment='center')
    
    # 3. 常见错误对
    plt.subplot(2, 3, 3)
    common_errors = analysis_results['common_errors']
    if common_errors:
        # 只取前10个最常见的错误对
        top_errors = dict(list(common_errors.items())[:10])
        plt.barh(list(top_errors.keys()), list(top_errors.values()))
        plt.title('Top 10 Common Error Pairs')
        plt.xlabel('Occurrence Count')
    else:
        plt.text(0.5, 0.5, 'No error pair data', horizontalalignment='center', verticalalignment='center')
    
    # 4. 句子长度分布
    plt.subplot(2, 3, 4)
    sentence_lengths = analysis_results['sentence_lengths']
    if sentence_lengths:
        plt.hist(sentence_lengths, bins=20)
        plt.title('Sentence Length Distribution')
        plt.xlabel('Sentence Length')
        plt.ylabel('Number of Sentences')
    else:
        plt.text(0.5, 0.5, 'No sentence length data', horizontalalignment='center', verticalalignment='center')
    
    # 5. 每句错误数量分布
    plt.subplot(2, 3, 5)
    error_counts = analysis_results['error_counts']
    if error_counts:
        plt.hist(error_counts, bins=range(1, max(error_counts) + 2))
        plt.title('Error Count Distribution per Sentence')
        plt.xlabel('Error Count')
        plt.ylabel('Number of Sentences')
    else:
        plt.text(0.5, 0.5, 'No error count data', horizontalalignment='center', verticalalignment='center')
    
    # 6. 常见词语错误
    plt.subplot(2, 3, 6)
    word_errors = analysis_results.get('word_errors', {})
    if word_errors:
        # 只取前10个最常见的词语错误
        top_word_errors = dict(list(word_errors.items())[:10])
        plt.barh(list(top_word_errors.keys()), list(top_word_errors.values()))
        plt.title('Top 10 Common Word Errors')
        plt.xlabel('Occurrence Count')
    else:
        plt.text(0.5, 0.5, 'No word error data', horizontalalignment='center', verticalalignment='center')
    
    # 添加图表说明
    plt.figtext(0.5, 0.02, 
                "Error Type (错误类型): Missing Character (漏字), Extra Character (多字), Substitution (替换)\n"
                "Error Position (错误位置): Front (前部), Middle (中部), End (后部)", 
                ha='center', fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 保存图像
    if output_path:
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path, dpi=300)
        print(f"图像已保存到: {output_path}")
    else:
        plt.savefig('error_distribution.png', dpi=300)
        print("图像已保存到: error_distribution.png")
        
    # 显示图像
    plt.show()
    
    # 打印一些统计信息
    print(f"总样本数: {analysis_results['total_samples']}")
    print(f"含错样本数: {analysis_results['sentences_with_errors']}")
    print(f"错误率: {analysis_results['error_rate']:.2%}")
    if 'avg_errors_per_sentence' in analysis_results:
        print(f"平均每句错误数: {analysis_results['avg_errors_per_sentence']:.2f}")
    if 'avg_char_error_rate' in analysis_results:
        print(f"平均字符错误率: {analysis_results['avg_char_error_rate']:.2%}")
