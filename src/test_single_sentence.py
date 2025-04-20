#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用BERT-LSTM模型对单个句子进行纠错的脚本。
支持交互式对话模式，持续接收输入并输出纠正后的句子。
"""

import os
import argparse
import torch
from bert_lstm_corrector import BertLSTMCorrector
import time
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='使用BERT-LSTM模型进行交互式文本纠错')
    parser.add_argument('--model_path', type=str, default='', help='模型路径，如果不指定，将使用最新的模型')
    parser.add_argument('--local_bert_path', type=str, default='/home/andy/ML2025/NLP/bert', 
                        help='BERT模型本地路径')
    parser.add_argument('--threshold', type=float, default=0.1, help='错误检测阈值')
    parser.add_argument('--single', action='store_true', help='单句模式，只处理一个句子')
    parser.add_argument('--sentence', type=str, default='', help='要纠错的句子（仅在单句模式下使用）')
    return parser.parse_args()

def find_latest_model():
    """查找models目录中最新的模型文件"""
    model_files = []
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 '{model_dir}' 不存在。")
        return None
    
    for file in os.listdir(model_dir):
        if file.endswith('.pt') and file.startswith('bert_lstm_model'):
            model_files.append(os.path.join(model_dir, file))
    
    if not model_files:
        print(f"错误: 在 '{model_dir}' 目录中未找到模型文件。")
        return None
    
    # 按修改时间排序
    latest_model = max(model_files, key=os.path.getmtime)
    model_name = os.path.basename(latest_model)[:-3]  # 去掉.pt后缀
    
    return model_name

def process_sentence(model, sentence, threshold=0.1, show_time=True):
    """处理单个句子并返回结果"""
    if not sentence.strip():
        return "输入为空，请重新输入。"
    
    start_time = time.time()
    corrected = model.correct(sentence, error_threshold=threshold)
    end_time = time.time()
    
    result = f"原文: {sentence}\n"
    result += f"纠正: {corrected}\n"
    
    # 显示差异
    changes = []
    if sentence != corrected:
        for i, (orig, corr) in enumerate(zip(sentence, corrected)):
            if orig != corr:
                changes.append(f"位置 {i+1}: '{orig}' -> '{corr}'")
        
        if changes:
            result += "修改: " + ", ".join(changes) + "\n"
        else:
            # 处理长度不同的情况
            if len(sentence) < len(corrected):
                result += f"添加: '{corrected[len(sentence):]}'\n"
            elif len(sentence) > len(corrected):
                result += f"删除: '{sentence[len(corrected):]}'\n"
    else:
        result += "未检测到错误。\n"
    
    if show_time:
        result += f"耗时: {(end_time - start_time):.4f} 秒\n"
    
    return result

def interactive_mode(model, threshold=0.1):
    """交互式对话模式"""
    print("\n=== BERT-LSTM 中文文本纠错系统 ===")
    print("输入中文文本，系统将自动纠正错误。")
    print("输入 'exit'、'quit' 或 'q' 退出程序。")
    print("输入 'help' 或 '?' 查看帮助。")
    print("=" * 35)
    
    history = []
    
    try:
        while True:
            # 使用不同颜色标记用户输入提示
            sys.stdout.write("\033[92m用户>\033[0m ")
            sys.stdout.flush()
            
            user_input = input().strip()
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("再见！")
                break
            
            # 帮助命令
            if user_input.lower() in ['help', '?']:
                print("\n帮助信息:")
                print("  - 直接输入中文文本，系统将自动纠正错误")
                print("  - 输入 'exit'、'quit' 或 'q' 退出程序")
                print("  - 输入 'help' 或 '?' 查看此帮助信息")
                print("  - 输入 'threshold 数值' 调整错误检测阈值 (当前: {:.2f})".format(threshold))
                print("  - 输入 'history' 查看历史记录")
                print("  - 输入 'clear' 清屏")
                continue
            
            # 阈值调整命令
            if user_input.lower().startswith('threshold '):
                try:
                    new_threshold = float(user_input.split()[1])
                    if 0 <= new_threshold <= 1:
                        threshold = new_threshold
                        print(f"错误检测阈值已调整为: {threshold:.2f}")
                    else:
                        print("错误: 阈值必须在0到1之间")
                except:
                    print("错误: 无效的阈值数值")
                continue
            
            # 历史记录命令
            if user_input.lower() == 'history':
                if not history:
                    print("历史记录为空")
                else:
                    print("\n=== 历史记录 ===")
                    for i, (input_text, corrected_text) in enumerate(history):
                        print(f"{i+1}. 原文: {input_text}")
                        print(f"   纠正: {corrected_text}")
                    print("=" * 16)
                continue
            
            # 清屏命令
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            # 进行纠错处理
            # 使用不同颜色标记系统输出
            sys.stdout.write("\033[94m系统>\033[0m ")
            result = process_sentence(model, user_input, threshold)
            print(result)
            
            # 保存到历史记录
            corrected_text = result.split('\n')[1].replace("纠正: ", "")
            history.append((user_input, corrected_text))
            
            print()  # 添加空行分隔对话
    
    except KeyboardInterrupt:
        print("\n程序已中断。再见！")
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("程序终止。")

def main():
    args = parse_args()
    
    # 创建模型实例
    print(f"初始化BERT-LSTM模型...")
    model = BertLSTMCorrector(
        local_model_path=args.local_bert_path,
        batch_size=1,
        max_seq_len=128
    )
    
    # 加载模型
    model_name = args.model_path
    if not model_name:
        model_name = find_latest_model()
        if not model_name:
            print("错误: 找不到模型文件，请使用--model_path指定模型路径。")
            return
    
    print(f"加载模型: {model_name}")
    success = model.load_model(model_name)
    
    if not success:
        print(f"错误: 加载模型 '{model_name}' 失败。")
        return
    
    # 单句模式
    if args.single or args.sentence:
        sentence = args.sentence
        if not sentence:
            sentence = input("请输入要纠错的句子: ")
        result = process_sentence(model, sentence, args.threshold)
        print(result)
    else:
        # 交互模式
        interactive_mode(model, args.threshold)

if __name__ == "__main__":
    main() 