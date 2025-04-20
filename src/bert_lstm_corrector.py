#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BERT-LSTM based corrector for Chinese Text Correction task.
This module uses the BERT model as encoder and LSTM as decoder for correcting errors in Chinese text.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time

try:
    from transformers import BertTokenizer, BertModel, BertConfig
    from transformers import get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. BERT features will be disabled.")


class ChineseTextCorrectionDataset(Dataset):
    """Dataset for Chinese text correction task"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_data(data)
        
        # 统计一些数据集信息
        error_samples = sum(1 for ex in self.examples if ex['label'] == 1)
        total_samples = len(self.examples)
        print(f"Dataset: {total_samples} samples, {error_samples} error samples ({error_samples/max(1, total_samples)*100:.2f}%)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def _prepare_data(self, data):
        examples = []
        errors_marked = 0
        total_chars = 0
        
        for sample in data:
            source = sample['source']
            target = sample['target']
            
            # Tokenize input
            source_encoding = self.tokenizer.encode_plus(
                source,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            # Create labels for detection and correction
            detection_labels = torch.zeros(self.max_length, dtype=torch.long)
            correction_labels = torch.zeros(self.max_length, dtype=torch.long)
            
            # Set default values to ignore in loss calculation
            detection_labels[:] = -100
            correction_labels[:] = -100
            
            # Set default label for tokens corresponding to real text (not padding/special)
            for i in range(min(len(source) + 2, self.max_length)):  # +2 for [CLS] and [SEP]
                detection_labels[i] = 0  # 默认为0，表示字符正确
            
            # If source and target are different and same length, mark errors
            if source != target and len(source) == len(target):
                # Manually create character to token mapping
                char_to_token_mapping = self._create_char_to_token_mapping(source)
                
                # Mark errors and their corrections
                for i, (s_char, t_char) in enumerate(zip(source, target)):
                    total_chars += 1
                    if s_char != t_char:
                        # Find the token index for this character position
                        if i in char_to_token_mapping:
                            token_idx = char_to_token_mapping[i]
                            if token_idx < self.max_length:
                                detection_labels[token_idx] = 1  # Mark as error
                                correct_id = self.tokenizer.convert_tokens_to_ids(t_char)
                                correction_labels[token_idx] = correct_id
                                errors_marked += 1
            
            example = {
                'input_ids': source_encoding['input_ids'].squeeze(),
                'attention_mask': source_encoding['attention_mask'].squeeze(),
                'detection_labels': detection_labels,
                'correction_labels': correction_labels,
                'source': source,
                'target': target,
                'label': 1 if source != target else 0
            }
            examples.append(example)
        
        if total_chars > 0:
            print(f"Dataset preparation: {errors_marked} errors marked out of {total_chars} characters ({errors_marked/total_chars*100:.2f}%)")
        
        return examples
    
    def _create_char_to_token_mapping(self, text):
        """Create a mapping from character positions to token indices"""
        mapping = {}
        
        # 使用更精确的方法创建字符到token的映射
        tokens = []
        
        # 先获取单字符token列表
        for char in text:
            char_tokens = self.tokenizer.tokenize(char)
            tokens.extend(char_tokens)
        
        # 计算原始文本中每个字符的位置对应的token索引
        # 添加1是因为[CLS]标记占据了第一个位置
        char_idx = 0
        token_idx = 1
        
        while char_idx < len(text) and token_idx < self.max_length - 1:  # -1为[SEP]预留空间
            mapping[char_idx] = token_idx
            
            # 处理一个字符可能对应多个token的情况
            char_tokens = self.tokenizer.tokenize(text[char_idx])
            token_idx += len(char_tokens)
            char_idx += 1
        
        return mapping


class BertLSTMCorrector(nn.Module):
    """BERT-LSTM based corrector for Chinese text"""
    
    def __init__(
            self,
            local_model_path="../bert",
            batch_size=16,
            learning_rate=5e-5,
            epochs=3,
            max_seq_len=128,
            device=None
        ):
        super(BertLSTMCorrector, self).__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required for BertLSTMCorrector")
        
        self.local_model_path = local_model_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_seq_len = max_seq_len
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        self._initialize_model()
        self.confusion_pairs = defaultdict(Counter)
        # 增加常见错误模式字典
        self.error_patterns = {}
        # 增加语法错误模式字典
        self.grammar_errors = {}
        # 错误检测阈值 - 可通过参数调整
        self.error_threshold = 0.1
    
    def _initialize_model(self):
        """Initialize the model using local BERT model"""
        try:
            print(f"Loading model from local path: {self.local_model_path}")
            
            # Load configuration
            config_path = os.path.join(self.local_model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            config = BertConfig.from_json_file(config_path)
            print(f"Model config loaded: hidden_size={config.hidden_size}")
            
            # Load tokenizer
            vocab_path = os.path.join(self.local_model_path, "vocab.txt")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocab file not found at {vocab_path}")
            
            self.tokenizer = BertTokenizer(vocab_path)
            print(f"Tokenizer loaded: vocab_size={len(self.tokenizer)}")
            
            # Load BERT model
            self.bert_encoder = BertModel.from_pretrained(self.local_model_path, config=config)
            print("BERT encoder loaded successfully")
            
            # 改进：增加注意力层
            self.attention = nn.MultiheadAttention(config.hidden_size, num_heads=8, dropout=0.1)
            
            # 改进：增加层归一化
            self.layer_norm1 = nn.LayerNorm(config.hidden_size)
            self.layer_norm2 = nn.LayerNorm(config.hidden_size)
            
            # 改进：增加前馈神经网络
            self.feed_forward = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Dropout(0.1)
            )
            
            # Initialize detection head (binary classification: error or not)
            self.detection_head = nn.Linear(config.hidden_size, 2)
            
            # 改进：使用改进的LSTM解码器
            self.correction_lstm = nn.LSTM(
                config.hidden_size, 
                config.hidden_size,
                num_layers=2,  # 增加层数
                batch_first=True,
                bidirectional=True,
                dropout=0.2  # 增加dropout
            )
            
            # Initialize correction head (predict correct character)
            self.correction_head = nn.Linear(config.hidden_size * 2, config.vocab_size)
            
            # 改进：加入额外的correction验证层
            self.correction_verifier = nn.Linear(config.hidden_size * 3, 2)  # BERT隐藏层大小 + 双向LSTM输出大小
            
            # Move model components to device
            self.bert_encoder.to(self.device)
            self.attention.to(self.device)
            self.layer_norm1.to(self.device)
            self.layer_norm2.to(self.device)
            self.feed_forward.to(self.device)
            self.detection_head.to(self.device)
            self.correction_lstm.to(self.device)
            self.correction_head.to(self.device)
            self.correction_verifier.to(self.device)
            
            print("Model initialization complete")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def train(self, train_data, val_data=None, epochs=None, batch_size=None, learning_rate=None, error_threshold=None):
        """Train the model"""
        # Update parameters if provided
        self.epochs = epochs or self.epochs
        self.batch_size = batch_size or self.batch_size
        self.learning_rate = learning_rate or self.learning_rate
        if error_threshold is not None:
            self.error_threshold = error_threshold
        
        # Extract confusion pairs from training data
        self._extract_confusion_pairs(train_data)
        # 提取错误模式
        self._extract_error_patterns(train_data)
        
        # Prepare datasets
        print("Preparing datasets...")
        train_dataset = ChineseTextCorrectionDataset(train_data, self.tokenizer, self.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if val_data:
            val_dataset = ChineseTextCorrectionDataset(val_data, self.tokenizer, self.max_seq_len)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Set up optimizer
        # 改进：使用分组不同学习率和权重衰减
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.bert_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.learning_rate, 'weight_decay': 0.01},
            {'params': [p for n, p in self.bert_encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.learning_rate, 'weight_decay': 0.0},
            {'params': self.detection_head.parameters(), 'lr': self.learning_rate * 10, 'weight_decay': 0.01},
            {'params': self.correction_lstm.parameters(), 'lr': self.learning_rate * 5, 'weight_decay': 0.01},
            {'params': self.correction_head.parameters(), 'lr': self.learning_rate * 5, 'weight_decay': 0.01},
            {'params': list(self.attention.parameters()) + 
                      list(self.layer_norm1.parameters()) + 
                      list(self.layer_norm2.parameters()) + 
                      list(self.feed_forward.parameters()) + 
                      list(self.correction_verifier.parameters()),
             'lr': self.learning_rate * 5, 'weight_decay': 0.01}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # 改进：使用更合理的损失权重
        detection_weight = 2.0
        correction_weight = 1.0
        verification_weight = 0.5  # 验证层的权重
        
        # 统计样本中的错误率
        error_count = 0
        total_chars = 0
        for sample in train_data:
            if sample['label'] == 1 and len(sample['source']) == len(sample['target']):
                for s_char, t_char in zip(sample['source'], sample['target']):
                    total_chars += 1
                    if s_char != t_char:
                        error_count += 1
        
        error_rate = error_count / max(1, total_chars)
        print(f"Error rate in training data: {error_rate:.4f}")
        
        # 计算类别权重来平衡训练
        pos_weight = (1.0 - error_rate) / max(0.001, error_rate)
        print(f"Using positive class weight: {pos_weight:.4f}")
        
        # 加载预训练权重
        if hasattr(self, 'best_val_f05') and self.best_val_f05 > 0:
            print(f"Resuming from previous training with F0.5: {self.best_val_f05:.4f}")
        else:
            self.best_val_f05 = 0.0
        
        # Training loop
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Set to training mode
            self.bert_encoder.train()
            self.attention.train()
            self.layer_norm1.train()
            self.layer_norm2.train()
            self.feed_forward.train()
            self.detection_head.train()
            self.correction_lstm.train()
            self.correction_head.train()
            self.correction_verifier.train()
            
            train_loss = 0
            detection_losses = 0
            correction_losses = 0
            verification_losses = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                detection_labels = batch['detection_labels'].to(self.device)
                correction_labels = batch['correction_labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Get BERT outputs
                bert_outputs = self.bert_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                last_hidden_state = bert_outputs.last_hidden_state
                
                # 改进：应用自注意力机制 - 强化上下文理解
                attn_output, _ = self.attention(
                    last_hidden_state.transpose(0, 1),
                    last_hidden_state.transpose(0, 1),
                    last_hidden_state.transpose(0, 1),
                    key_padding_mask=(1 - attention_mask).bool()  # 修复：使用key_padding_mask替代attn_mask
                )
                attn_output = attn_output.transpose(0, 1)
                
                # 残差连接和层归一化
                hidden_state = self.layer_norm1(last_hidden_state + attn_output)
                
                # 前馈网络
                ff_output = self.feed_forward(hidden_state)
                enhanced_hidden_state = self.layer_norm2(hidden_state + ff_output)
                
                # Error detection
                detection_logits = self.detection_head(enhanced_hidden_state)
                
                # 使用类别权重计算损失
                detection_loss = F.cross_entropy(
                    detection_logits.view(-1, 2),
                    detection_labels.view(-1),
                    ignore_index=-100,
                    weight=torch.tensor([1.0, pos_weight], device=self.device)
                )
                
                # Error correction
                lstm_outputs, _ = self.correction_lstm(enhanced_hidden_state)
                correction_logits = self.correction_head(lstm_outputs)
                correction_loss = F.cross_entropy(
                    correction_logits.view(-1, self.tokenizer.vocab_size),
                    correction_labels.view(-1),
                    ignore_index=-100
                )
                
                # 改进：验证层 - 结合原始token表示和LSTM输出，提供额外验证
                # 创建验证标签 - 只对有错误的位置计算损失
                verification_features = torch.cat([enhanced_hidden_state, lstm_outputs], dim=-1)
                verification_logits = self.correction_verifier(verification_features)
                
                # 创建验证标签：1表示需要纠正，0表示不需要
                verification_labels = torch.zeros_like(detection_labels)
                verification_labels[detection_labels == 1] = 1  # 如果检测标签是错误，那么验证标签就是需要纠正
                
                verification_loss = F.cross_entropy(
                    verification_logits.view(-1, 2),
                    verification_labels.view(-1),
                    ignore_index=-100,
                    weight=torch.tensor([1.0, pos_weight], device=self.device)
                )
                
                # Total loss with weights
                loss = detection_weight * detection_loss + correction_weight * correction_loss + verification_weight * verification_loss
                
                # Backward pass
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸 - 单独对每个组件进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.bert_encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.attention.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.layer_norm1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.layer_norm2.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.feed_forward.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.detection_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.correction_lstm.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.correction_head.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.correction_verifier.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                detection_losses += detection_loss.item()
                correction_losses += correction_loss.item()
                verification_losses += verification_loss.item()
                
                # Print progress
                if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1:
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, D_Loss: {detection_loss.item():.4f}, "
                          f"C_Loss: {correction_loss.item():.4f}, V_Loss: {verification_loss.item():.4f}")
            
            # Epoch complete
            avg_loss = train_loss / len(train_loader)
            avg_det_loss = detection_losses / len(train_loader)
            avg_corr_loss = correction_losses / len(train_loader)
            avg_ver_loss = verification_losses / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} completed, "
                  f"Avg Loss: {avg_loss:.4f}, Det Loss: {avg_det_loss:.4f}, "
                  f"Corr Loss: {avg_corr_loss:.4f}, Ver Loss: {avg_ver_loss:.4f}")
            
            # Validate if validation data is provided
            if val_data:
                val_metrics = self._validate(val_loader)
                print(f"Validation - F0.5: {val_metrics['correction_f05']:.4f}, "
                      f"Precision: {val_metrics['correction_precision']:.4f}, "
                      f"Recall: {val_metrics['correction_recall']:.4f}")
                
                # Save best model
                if val_metrics['correction_f05'] > self.best_val_f05:
                    self.best_val_f05 = val_metrics['correction_f05']
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self._save_model(f"bert_lstm_model_{timestamp}")
                    print(f"New best model saved with F0.5: {self.best_val_f05:.4f}")
        
        # Save final model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._save_model(f"bert_lstm_model_final_{timestamp}")
        print("Training completed!")
    
    def _extract_error_patterns(self, train_data):
        """提取常见错误模式和语法错误"""
        print("Extracting error patterns from training data...")
        
        pattern_counts = defaultdict(int)
        context_errors = defaultdict(lambda: defaultdict(int))
        
        for sample in train_data:
            if sample['label'] == 1 and len(sample['source']) == len(sample['target']):
                source = sample['source']
                target = sample['target']
                
                # 查找错误并记录上下文
                for i, (s_char, t_char) in enumerate(zip(source, target)):
                    if s_char != t_char:
                        # 获取错误的上下文
                        left_context = source[max(0, i-2):i]
                        right_context = source[i+1:min(i+3, len(source))]
                        
                        # 记录错误模式
                        pattern = f"{left_context}_{s_char}_{right_context}"
                        pattern_counts[pattern] += 1
                        
                        # 记录特定上下文的错误
                        context = f"{left_context}_{right_context}"
                        context_errors[context][(s_char, t_char)] += 1
                        
                        # 检查常见语法错误
                        # 例如：的/地/得
                        if s_char in "的地得" and t_char in "的地得":
                            self.grammar_errors[(s_char, left_context, right_context)] = t_char
        
        # 过滤出高频错误模式
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:100]:
            self.error_patterns[pattern] = count
        
        # 过滤出特定上下文的高频错误修正
        for context, corrections in context_errors.items():
            if len(corrections) > 0:
                # 找出这个上下文中最常见的修正
                most_common = max(corrections.items(), key=lambda x: x[1])
                if most_common[1] >= 3:  # 至少出现3次才记录
                    s_char, t_char = most_common[0]
                    self.confusion_pairs[s_char][t_char] += most_common[1] * 2  # 增加权重
        
        print(f"Extracted {len(self.error_patterns)} error patterns")
        print(f"Extracted {len(self.grammar_errors)} grammar error patterns") 
    
    def correct(self, text, error_threshold=None):
        """Correct errors in input text"""
        if len(text) == 0:
            return text
        
        # 使用传入的阈值，否则使用默认值
        threshold = error_threshold if error_threshold is not None else self.error_threshold
        
        self.bert_encoder.eval()
        self.attention.eval()
        self.layer_norm1.eval()
        self.layer_norm2.eval()
        self.feed_forward.eval()
        self.detection_head.eval()
        self.correction_lstm.eval()
        self.correction_head.eval()
        self.correction_verifier.eval()
        
        try:
            # Tokenize input without using return_offset_mapping
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Create a manual mapping from token positions to character positions
            char_mapping = self._token_to_char_mapping(text)
            
            # Get model outputs
            with torch.no_grad():
                # BERT encoder
                outputs = self.bert_encoder(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                
                # 应用注意力机制
                attn_output, _ = self.attention(
                    last_hidden_state.transpose(0, 1),
                    last_hidden_state.transpose(0, 1),
                    last_hidden_state.transpose(0, 1),
                    key_padding_mask=(1 - attention_mask).bool()  # 修复：使用key_padding_mask替代attn_mask
                )
                attn_output = attn_output.transpose(0, 1)
                
                # 残差连接和层归一化
                hidden_state = self.layer_norm1(last_hidden_state + attn_output)
                
                # 前馈网络
                ff_output = self.feed_forward(hidden_state)
                enhanced_hidden_state = self.layer_norm2(hidden_state + ff_output)
                
                # Error detection
                detection_logits = self.detection_head(enhanced_hidden_state)
                detection_probs = torch.softmax(detection_logits, dim=-1)
                error_probs = detection_probs[0, :, 1]  # Error probability
                
                # Error correction with LSTM
                correction_outputs, _ = self.correction_lstm(enhanced_hidden_state)
                correction_logits = self.correction_head(correction_outputs)
                correction_probs = torch.softmax(correction_logits, dim=-1)
                
                # Verification layer
                verification_features = torch.cat([enhanced_hidden_state, correction_outputs], dim=-1)
                verification_logits = self.correction_verifier(verification_features)
                verification_probs = torch.softmax(verification_logits, dim=-1)
                verification_scores = verification_probs[0, :, 1]  # 需要纠正的概率
            
            # Find and correct errors
            result = list(text)
            corrections_made = 0
            
            # 预处理：检查常见错误模式
            for i in range(len(text)):
                left_context = text[max(0, i-2):i]
                right_context = text[i+1:min(i+3, len(text))]
                pattern = f"{left_context}_{text[i]}_{right_context}"
                
                # 如果是已知的错误模式，提高检测概率
                if pattern in self.error_patterns:
                    # 找到对应的token位置
                    for token_idx, char_pos in char_mapping.items():
                        if char_pos == i:
                            error_probs[token_idx] *= 1.5  # 提高错误概率
                
                # 检查语法错误模式
                for (error_char, err_left, err_right), correct_char in self.grammar_errors.items():
                    if text[i] == error_char and left_context.endswith(err_left) and right_context.startswith(err_right):
                        # 找到对应token位置
                        for token_idx, char_pos in char_mapping.items():
                            if char_pos == i:
                                # 直接应用已知的语法修正
                                result[i] = correct_char
                                corrections_made += 1
            
            # 主要纠错逻辑
            for token_idx in range(1, min(len(char_mapping) + 1, len(error_probs) - 1)):  # Skip CLS and SEP
                # 结合错误检测和验证分数
                combined_score = error_probs[token_idx] * verification_scores[token_idx]
                
                # Check if token is likely to be an error
                if combined_score > threshold or error_probs[token_idx] > threshold * 1.5:
                    if token_idx in char_mapping:
                        char_pos = char_mapping[token_idx]
                        
                        if char_pos < len(text):
                            # Get top correction candidates
                            top_corrections = torch.topk(correction_probs[0, token_idx], k=5)
                            
                            # 查看多个候选项
                            for i in range(min(3, len(top_corrections.indices))):
                                correction_id = top_corrections.indices[i].item()
                                correction = self.tokenizer.convert_ids_to_tokens([correction_id])[0]
                                correction_prob = top_corrections.values[i].item()
                                
                                # 根据修正概率和验证分数决定是否修正
                                confidence_score = correction_prob * verification_scores[token_idx]
                                
                                # 只有当概率足够高，且修正后的字符不同于原字符时才进行修正
                                if (confidence_score > 0.3 or correction_prob > 0.5) and \
                                   correction not in ["[UNK]", "[PAD]", "[CLS]", "[SEP]"] and \
                                   (correction != text[char_pos] or i == 0):  # 第一个候选始终考虑
                                    
                                    if correction.startswith("##"):
                                        correction = correction[2:]
                                    
                                    # 查看混淆对中是否有这个修正
                                    if text[char_pos] in self.confusion_pairs and correction in self.confusion_pairs[text[char_pos]]:
                                        # 如果是已知的混淆对，增加置信度
                                        confidence_boost = self.confusion_pairs[text[char_pos]][correction] / 10.0
                                        if confidence_boost > 0.2 or confidence_score > 0.4:
                                            result[char_pos] = correction
                                            corrections_made += 1
                                            break
                                    elif i == 0 and confidence_score > 0.5:  # 如果是第一个候选，且置信度很高
                                        result[char_pos] = correction
                                        corrections_made += 1
                                        break
            
            # 如果没有修正，但有高置信度的错误，尝试使用混淆集
            if corrections_made == 0:
                for i, char in enumerate(text):
                    if char in self.confusion_pairs:
                        for correction, count in self.confusion_pairs[char].most_common(1):
                            if count > 5:  # 只有当混淆对出现次数足够多时
                                # 获取上下文
                                left_context = text[max(0, i-2):i]
                                right_context = text[i+1:min(i+3, len(text))]
                                
                                # 检查上下文中是否有这个错误
                                context = f"{left_context}_{right_context}"
                                if context in self.error_patterns:
                                    result[i] = correction
                                    corrections_made += 1
                                    break
            
            return "".join(result)
            
        except Exception as e:
            print(f"Error correcting text: {e}")
            return text
    
    def _token_to_char_mapping(self, text):
        """Create a mapping from token positions to character positions for Chinese text"""
        mapping = {}
        
        # 直接字符位置映射可能不够准确
        # 但对于大多数中文字符，一个字符通常对应一个token
        char_idx = 0
        token_idx = 1  # 1是因为[CLS]标记
        
        while char_idx < len(text) and token_idx < self.max_seq_len - 1:  # -1为[SEP]预留空间
            char_tokens = self.tokenizer.tokenize(text[char_idx])
            mapping[token_idx] = char_idx
            
            # 处理一个字符可能对应多个token的情况
            token_idx += len(char_tokens)
            char_idx += 1
        
        return mapping
    
    def _extract_confusion_pairs(self, train_data):
        """Extract character confusion pairs from training data"""
        confusion_counts = {}
        total_errors = 0
        
        for sample in train_data:
            if sample['label'] == 1:  # Only process samples with errors
                source = sample['source']
                target = sample['target']
                
                # For character substitution errors (when lengths are equal)
                if len(source) == len(target):
                    for s_char, t_char in zip(source, target):
                        if s_char != t_char:
                            total_errors += 1
                            self.confusion_pairs[s_char][t_char] += 1
                            
                            # 统计错误频率
                            if s_char not in confusion_counts:
                                confusion_counts[s_char] = 0
                            confusion_counts[s_char] += 1
        
        # 找出最常见的错误字符
        if confusion_counts:
            top_errors = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            print("Top 20 most common error characters:")
            for char, count in top_errors:
                percentage = (count / total_errors) * 100
                corrections = self.confusion_pairs[char].most_common(3)
                corrections_str = ", ".join([f"'{c}': {n}" for c, n in corrections])
                print(f"  '{char}': {count} times ({percentage:.2f}%), corrections: {corrections_str}")
        
        print(f"Extracted {len(self.confusion_pairs)} confusion pairs from {total_errors} total errors")
    
    def _save_model(self, name):
        """Save model weights"""
        os.makedirs('models', exist_ok=True)
        
        # Save model weights
        torch.save({
            'bert_encoder': self.bert_encoder.state_dict(),
            'detection_head': self.detection_head.state_dict(),
            'correction_lstm': self.correction_lstm.state_dict(),
            'correction_head': self.correction_head.state_dict(),
            'correction_verifier': self.correction_verifier.state_dict(),
            'attention': self.attention.state_dict(),
            'layer_norm1': self.layer_norm1.state_dict(),
            'layer_norm2': self.layer_norm2.state_dict(),
            'feed_forward': self.feed_forward.state_dict(),
            'confusion_pairs': self.confusion_pairs,
            'error_patterns': self.error_patterns,
            'grammar_errors': self.grammar_errors,
        }, f'models/{name}.pt')
        
        print(f"Model saved to models/{name}.pt")
    
    def load_model(self, name):
        """Load model weights"""
        model_path = f'models/{name}.pt'
        if not os.path.exists(model_path):
            print(f"Model {name} not found at {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.bert_encoder.load_state_dict(checkpoint['bert_encoder'])
            self.detection_head.load_state_dict(checkpoint['detection_head'])
            self.correction_lstm.load_state_dict(checkpoint['correction_lstm'])
            self.correction_head.load_state_dict(checkpoint['correction_head'])
            
            # 尝试加载新添加的组件，如果存在的话
            if 'correction_verifier' in checkpoint:
                self.correction_verifier.load_state_dict(checkpoint['correction_verifier'])
            if 'attention' in checkpoint:
                self.attention.load_state_dict(checkpoint['attention'])
            if 'layer_norm1' in checkpoint:
                self.layer_norm1.load_state_dict(checkpoint['layer_norm1'])
            if 'layer_norm2' in checkpoint:
                self.layer_norm2.load_state_dict(checkpoint['layer_norm2'])
            if 'feed_forward' in checkpoint:
                self.feed_forward.load_state_dict(checkpoint['feed_forward'])
                
            # 加载混淆对和错误模式（如果有）
            if 'confusion_pairs' in checkpoint:
                self.confusion_pairs = checkpoint['confusion_pairs']
            if 'error_patterns' in checkpoint:
                self.error_patterns = checkpoint['error_patterns']
            if 'grammar_errors' in checkpoint:
                self.grammar_errors = checkpoint['grammar_errors']
            
            print(f"Successfully loaded model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _validate(self, val_loader):
        """Validate model performance"""
        self.bert_encoder.eval()
        self.detection_head.eval()
        self.correction_lstm.eval()
        self.correction_head.eval()
        
        predictions = []
        error_samples = 0
        correct_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                sources = batch['source']
                targets = batch['target']
                labels = batch['label']
                
                for i in range(input_ids.size(0)):
                    source = sources[i]
                    target = targets[i]
                    label = labels[i].item()
                    
                    # 对错误样本和非错误样本分别计数
                    if label == 1:
                        error_samples += 1
                    else:
                        correct_samples += 1
                    
                    # 只对有错误的样本进行纠正，对无错误的样本直接返回原文
                    if label == 1:
                        try:
                            corrected = self.correct(source)
                            predictions.append({
                                'source': source,
                                'target': target,
                                'prediction': corrected,
                                'label': label
                            })
                        except Exception as e:
                            print(f"Error correcting validation text: {e}")
                            predictions.append({
                                'source': source,
                                'target': target,
                                'prediction': source,  # fallback to source
                                'label': label
                            })
                    else:
                        # 无错误样本应该保持不变
                        predictions.append({
                            'source': source,
                            'target': source,  # target与source相同
                            'prediction': source,
                            'label': label
                        })
        
        print(f"Validation set: {error_samples} error samples, {correct_samples} correct samples")
        
        # 分析预测结果
        corrections_made = sum(1 for p in predictions if p['source'] != p['prediction'])
        correct_corrections = sum(1 for p in predictions if p['prediction'] == p['target'] and p['source'] != p['target'])
        
        print(f"Made {corrections_made} corrections, {correct_corrections} were correct")
        
        # Calculate metrics
        metrics = self._evaluate_performance(predictions)
        return metrics
    
    def _evaluate_performance(self, predictions):
        """Evaluate model performance"""
        detection_TP = 0
        detection_FP = 0
        detection_FN = 0
        
        correction_TP = 0
        correction_FP = 0
        correction_FN = 0
        
        for sample in predictions:
            source = sample['source']
            target = sample['target']
            prediction = sample['prediction']
            
            # Get edit operations
            target_edits = self._get_edits(source, target)
            pred_edits = self._get_edits(source, prediction)
            
            # Detection evaluation
            target_edit_pos = {(edit[1], edit[2]) for edit in target_edits}
            pred_edit_pos = {(edit[1], edit[2]) for edit in pred_edits}
            
            detection_TP += len(target_edit_pos & pred_edit_pos)
            detection_FP += len(pred_edit_pos - target_edit_pos)
            detection_FN += len(target_edit_pos - pred_edit_pos)
            
            # Correction evaluation
            target_edit_dict = {(edit[1], edit[2]): edit for edit in target_edits}
            pred_edit_dict = {(edit[1], edit[2]): edit for edit in pred_edits}
            
            for pos in target_edit_pos & pred_edit_pos:
                target_edit = target_edit_dict[pos]
                pred_edit = pred_edit_dict[pos]
                
                if len(target_edit) > 3 and len(pred_edit) > 3:
                    if target_edit[3] == pred_edit[3]:
                        correction_TP += 1
                    else:
                        correction_FP += 1
                else:
                    correction_TP += 1
            
            # Uncorrected errors
            correction_FN += len(target_edit_pos - pred_edit_pos)
            # False corrections
            correction_FP += len(pred_edit_pos - target_edit_pos)
        
        # Calculate metrics
        detection_precision = detection_TP / (detection_TP + detection_FP) if (detection_TP + detection_FP) > 0 else 0
        detection_recall = detection_TP / (detection_TP + detection_FN) if (detection_TP + detection_FN) > 0 else 0
        detection_f1 = 2 * detection_precision * detection_recall / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
        detection_f05 = (1 + 0.5**2) * detection_precision * detection_recall / ((0.5**2 * detection_precision) + detection_recall) if (detection_precision + detection_recall) > 0 else 0
        
        correction_precision = correction_TP / (correction_TP + correction_FP) if (correction_TP + correction_FP) > 0 else 0
        correction_recall = correction_TP / (correction_TP + correction_FN) if (correction_TP + correction_FN) > 0 else 0
        correction_f1 = 2 * correction_precision * correction_recall / (correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0
        correction_f05 = (1 + 0.5**2) * correction_precision * correction_recall / ((0.5**2 * correction_precision) + correction_recall) if (correction_precision + correction_recall) > 0 else 0
        
        return {
            'detection_precision': detection_precision,
            'detection_recall': detection_recall,
            'detection_f1': detection_f1,
            'detection_f05': detection_f05,
            'correction_precision': correction_precision,
            'correction_recall': correction_recall,
            'correction_f1': correction_f1,
            'correction_f05': correction_f05,
        }
    
    def _get_edits(self, source, target):
        """Get edit operations between source and target text"""
        result = []
        i, j = 0, 0
        
        while i < len(source) and j < len(target):
            if source[i] == target[j]:
                i += 1
                j += 1
            else:
                # Simple replacement
                result.append(('S', i, i+1, target[j]))
                i += 1
                j += 1
        
        # Handle remaining characters
        while i < len(source):
            result.append(('R', i, i+1))
            i += 1
        
        while j < len(target):
            result.append(('M', i, i, target[j]))
            j += 1
        
        return result 