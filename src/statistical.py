#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical corrector for Chinese Text Correction task.
This module implements statistical methods for correcting errors in Chinese text.
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # Import CRF if available
    try:
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics

        CRF_AVAILABLE = True
    except ImportError:
        CRF_AVAILABLE = False
        print("Warning: sklearn_crfsuite not available. CRF features will be disabled.")

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    CRF_AVAILABLE = False
    print("Warning: scikit-learn not available. Some features will be disabled.")


class StatisticalCorrector:
    """
    A statistical corrector for Chinese text.
    """

    def __init__(self, method='ngram'):
        """
        Initialize the statistical corrector.

        Args:
            method: The statistical method to use. Options: 'ngram', 'ml', 'crf'.
        """
        self.method = method

        # N-gram language model
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.fourgram_counts = Counter()  # 4-gram for better context modeling

        # Character-level confusion matrix
        self.confusion_matrix = defaultdict(Counter)

        # Character error probabilities
        self.error_probs = defaultdict(float)

        # Phonetic and visual similarity matrices
        self.phonetic_similarity = defaultdict(dict)
        self.visual_similarity = defaultdict(dict)

        # Interpolation weights for different n-gram models
        self.lambda_1 = 0.1  # Weight for unigram
        self.lambda_2 = 0.3  # Weight for bigram
        self.lambda_3 = 0.4  # Weight for trigram
        self.lambda_4 = 0.2  # Weight for 4-gram

        # Machine learning models
        self.ml_model = None
        self.vectorizer = None
        self.feature_scaler = None

        # Character corrections dictionary
        self.char_corrections = defaultdict(Counter)

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the statistical corrector using the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        if self.method == 'ngram':
            self._train_ngram_model(train_data)
        elif self.method == 'ml' and SKLEARN_AVAILABLE:
            self._train_ml_model(train_data)
        else:
            print(f"Warning: Method '{self.method}' not available. Falling back to n-gram model.")
            self._train_ngram_model(train_data)

    def _train_ngram_model(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train an n-gram language model for text correction.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # Build n-gram language model from correct sentences
        for sample in train_data:
            # Use target (correct) text for building the language model
            text = sample['target']

            # Count unigrams (single characters)
            for char in text:
                self.unigram_counts[char] += 1

            # Count bigrams
            for i in range(len(text) - 1):
                bigram = text[i:i+2]
                self.bigram_counts[bigram] += 1
            
            # Count trigrams
            for i in range(len(text) - 2):
                trigram = text[i:i+3]
                self.trigram_counts[trigram] += 1
            
            # Count 4-grams
            for i in range(len(text) - 3):
                fourgram = text[i:i+4]
                self.fourgram_counts[fourgram] += 1

            # Build confusion matrix from error pairs
            if sample['label'] == 1:  # Only for sentences with errors
                source = sample['source']
                target = sample['target']

                # For character substitution errors (when lengths are equal)
                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            # Record this confusion pair with context
                            left_context = source[max(0, i - 2) : i]
                            right_context = source[i + 1 : min(len(source), i + 3)]
                            context = left_context + '_' + right_context

                            self.confusion_matrix[(s_char, context)][t_char] += 1

                            # Also record general confusion without context
                            self.confusion_matrix[(s_char, '')][t_char] += 1

                            # Record error probability for this character
                            self.error_probs[s_char] += 1

                            # Record correction pair
                            self.char_corrections[s_char][t_char] += 1

        # Normalize error probabilities
        for char, count in self.error_probs.items():
            self.error_probs[char] = count / self.unigram_counts.get(char, 1)

        print(
            f"Trained n-gram model with {len(self.unigram_counts)} unigrams, "
            f"{len(self.bigram_counts)} bigrams, and {len(self.trigram_counts)} trigrams."
        )

    def _train_ml_model(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train a machine learning model for text correction.

        Args:
            train_data: List of dictionaries containing the training data.
        """

        if not SKLEARN_AVAILABLE:
            print("Cannot train ML model: scikit-learn not available.")
            return

        # 分为两个模型：错误检测模型和错误纠正模型
        print("Training ML models for Chinese text correction...")
        
        # 准备训练数据
        error_detection_data = []
        error_detection_labels = []
        correction_data = []
        correction_labels = []
        
        # 用于特征提取的窗口大小
        window_size = 2
        
        for sample in train_data:
            source = sample['source']
            target = sample['target']
            
            # 对于错误检测模型
            error_detection_data.append(source)
            error_detection_labels.append(sample['label'])  # 0表示无错误，1表示有错误
            
            # 对于错误纠正模型，只处理存在错误且长度相等的样本（替换错误）
            if sample['label'] == 1 and len(source) == len(target):
                for i, (s_char, t_char) in enumerate(zip(source, target)):
                    if s_char != t_char:  # 找到错误位置
                        # 提取错误字符的上下文
                        left_context = source[max(0, i - window_size):i]
                        right_context = source[min(i + 1, len(source)):min(i + window_size + 1, len(source))]
                        
                        # 如果上下文长度不足，用空白填充
                        if len(left_context) < window_size:
                            left_context = ' ' * (window_size - len(left_context)) + left_context
                        if len(right_context) < window_size:
                            right_context = right_context + ' ' * (window_size - len(right_context))
                        
                        # 组合特征
                        feature = left_context + s_char + right_context
                        correction_data.append(feature)
                        correction_labels.append(t_char)
        
        # 分割数据集
        X_train_detect, X_val_detect, y_train_detect, y_val_detect = train_test_split(
            error_detection_data, error_detection_labels, test_size=0.2, random_state=42
        )
        
        # 1. 训练错误检测模型
        print("Training error detection model...")
        # 使用TF-IDF向量化器提取特征
        self.detect_vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(1, 3), max_features=5000
        )
        X_train_detect_vec = self.detect_vectorizer.fit_transform(X_train_detect)
        X_val_detect_vec = self.detect_vectorizer.transform(X_val_detect)
        
        # 使用逻辑回归模型进行错误检测
        self.detect_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.detect_model.fit(X_train_detect_vec, y_train_detect)
        
        # 评估检测模型
        detect_val_pred = self.detect_model.predict(X_val_detect_vec)
        detect_accuracy = accuracy_score(y_val_detect, detect_val_pred)
        print(f"Error detection model accuracy: {detect_accuracy:.4f}")
        
        # 2. 训练错误纠正模型（如果有足够的纠正样本）
        if len(correction_data) > 10:
            print(f"Training error correction model with {len(correction_data)} samples...")
            
            # 分割纠正数据集
            X_train_corr, X_val_corr, y_train_corr, y_val_corr = train_test_split(
                correction_data, correction_labels, test_size=0.2, random_state=42
            )
            
            # 使用字符级别的CountVectorizer
            self.correct_vectorizer = CountVectorizer(
                analyzer='char', ngram_range=(1, 4), max_features=10000
            )
            X_train_corr_vec = self.correct_vectorizer.fit_transform(X_train_corr)
            X_val_corr_vec = self.correct_vectorizer.transform(X_val_corr)
            
            # 使用多类逻辑回归进行纠正
            self.correct_model = LogisticRegression(
                multi_class='multinomial', solver='lbfgs', max_iter=1000
            )
            self.correct_model.fit(X_train_corr_vec, y_train_corr)
            
            # 评估纠正模型
            corr_val_pred = self.correct_model.predict(X_val_corr_vec)
            corr_accuracy = accuracy_score(y_val_corr, corr_val_pred)
            print(f"Error correction model accuracy: {corr_accuracy:.4f}")
        else:
            print("Not enough correction samples to train correction model")
            self.correct_model = None
        
        # 为了提高效率，预先构建常见汉字列表
        common_chars = []
        if JIEBA_AVAILABLE:
            # 使用jieba分词器的词典
            common_chars = list(jieba.dt.FREQ.keys())
            # 只保留单字
            common_chars = [char for char in common_chars if len(char) == 1]
            # 按频率排序并取前3000个
            common_chars = sorted(common_chars, key=lambda x: jieba.dt.FREQ.get(x, 0), reverse=True)[:3000]
        else:
            # 如果没有jieba，则使用训练数据中的常用字
            char_freq = Counter()
            for sample in train_data:
                for char in sample['target']:
                    char_freq[char] += 1
            common_chars = [char for char, _ in char_freq.most_common(3000)]
        
        self.common_chars = common_chars
        
        # 构建字符相似度矩阵（可选，如果有这样的资源）
        self.build_similarity_matrices(train_data)
        
        print("ML models training completed")

    def build_similarity_matrices(self, train_data: List[Dict[str, Any]]) -> None:
        """
        构建字符相似度矩阵，包括形近字和音近字
        
        Args:
            train_data: 训练数据集
        """
        # 从错误样本中学习相似性
        for sample in train_data:
            if sample['label'] == 1 and len(sample['source']) == len(sample['target']):
                source = sample['source']
                target = sample['target']
                
                for i, (s_char, t_char) in enumerate(zip(source, target)):
                    if s_char != t_char:
                        # 记录这两个字符之间的相似度
                        if s_char not in self.visual_similarity:
                            self.visual_similarity[s_char] = {}
                        if t_char not in self.visual_similarity[s_char]:
                            self.visual_similarity[s_char][t_char] = 0
                        
                        # 增加相似度计数
                        self.visual_similarity[s_char][t_char] += 1
        
        # 归一化相似度矩阵
        for char in self.visual_similarity:
            total = sum(self.visual_similarity[char].values())
            for similar_char in self.visual_similarity[char]:
                self.visual_similarity[char][similar_char] /= total

    def correct(self, text: str) -> str:
        """
        Apply statistical correction to the input text.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        if self.method == 'ngram':
            return self._correct_with_ngram(text)
        elif self.method == 'ml' and SKLEARN_AVAILABLE and self.ml_model is not None:
            return self._correct_with_ml(text)
        else:
            return self._correct_with_ngram(text)

    def _correct_with_ngram(self, text: str) -> str:
        """
        Correct text using the n-gram language model.

        Args:
            text: Input text.

        Returns:
            Corrected text.
        """
        corrected_text = list(text)  # Convert to list for character-by-character editing

        # Check each character for potential errors
        for i in range(len(text)):
            char = text[i]

            # Skip characters with low error probability
            if self.error_probs.get(char, 0) < 0.01:
                continue

            # Get context for this character
            left_context = text[max(0, i - 2) : i]
            right_context = text[i + 1 : min(len(text), i + 3)]
            context = left_context + '_' + right_context

            # Check if we have seen this character in this context before
            if (char, context) in self.confusion_matrix and self.confusion_matrix[(char, context)]:
                # Get the most common correction for this character in this context
                correction = self.confusion_matrix[(char, context)].most_common(1)[0][0]
                corrected_text[i] = correction
                continue

            # If no specific context match, check general confusion matrix
            if (char, '') in self.confusion_matrix and self.confusion_matrix[(char, '')]:
                # Get the most common correction for this character
                correction = self.confusion_matrix[(char, '')].most_common(1)[0][0]
                # Only apply if it's a common error
                if self.confusion_matrix[(char, '')][correction] > 2:
                    corrected_text[i] = correction
                    continue

            # If no direct match, use interpolated n-gram model for characters with high error probability
            if self.error_probs.get(char, 0) >= 0.05 and i > 0 and i < len(text) - 1:
                # Generate candidate corrections
                candidates = set()

                # Add common characters as candidates
                candidates.update(list(self.unigram_counts.keys())[:300])  # Top 300 most common characters

                # Add correction candidates from confusion matrix
                for context_key in self.confusion_matrix:
                    if context_key[0] == char:
                        candidates.update(self.confusion_matrix[context_key].keys())

                # Try all candidates and find the one with highest probability
                best_score = -float('inf')
                best_char = char

                for candidate in candidates:
                    # Skip the original character
                    if candidate == char:
                        continue

                    # Calculate interpolated score using all n-gram models
                    score = 0

                    # Unigram probability (with smoothing)
                    unigram_prob = (self.unigram_counts.get(candidate, 0) + 1) / (
                        sum(self.unigram_counts.values()) + len(self.unigram_counts)
                    )
                    score += self.lambda_1 * unigram_prob

                    # Bigram probabilities
                    if len(left_context) > 0:
                        bigram_left = left_context[-1] + candidate
                        bigram_left_prob = (self.bigram_counts.get(bigram_left, 0) + 1) / (
                            self.unigram_counts.get(left_context[-1], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_2 * 0.5 * bigram_left_prob
                    
                    if len(right_context) > 0:
                        bigram_right = candidate + right_context[0]
                        bigram_right_prob = (self.bigram_counts.get(bigram_right, 0) + 1) / (
                            self.unigram_counts.get(candidate, 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_2 * 0.5 * bigram_right_prob

                    # Trigram probabilities
                    if len(left_context) >= 2:
                        trigram_left = left_context[-2:] + candidate
                        trigram_left_prob = (self.trigram_counts.get(trigram_left, 0) + 1) / (
                            self.bigram_counts.get(left_context[-2:], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_3 * 0.3 * trigram_left_prob
                    
                    if len(left_context) >= 1 and len(right_context) >= 1:
                        trigram_mid = left_context[-1] + candidate + right_context[0]
                        trigram_mid_prob = (self.trigram_counts.get(trigram_mid, 0) + 1) / (
                            self.bigram_counts.get(left_context[-1] + candidate, 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_3 * 0.4 * trigram_mid_prob
                    
                    if len(right_context) >= 2:
                        trigram_right = candidate + right_context[:2]
                        trigram_right_prob = (self.trigram_counts.get(trigram_right, 0) + 1) / (
                            self.bigram_counts.get(candidate + right_context[0], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_3 * 0.3 * trigram_right_prob

                    # 4-gram probabilities
                    if len(left_context) >= 3:
                        fourgram_left = left_context[-3:] + candidate
                        fourgram_left_prob = (self.fourgram_counts.get(fourgram_left, 0) + 1) / (
                            self.trigram_counts.get(left_context[-3:], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_4 * 0.25 * fourgram_left_prob
                    
                    if len(left_context) >= 2 and len(right_context) >= 1:
                        fourgram_midleft = left_context[-2:] + candidate + right_context[0]
                        fourgram_midleft_prob = (self.fourgram_counts.get(fourgram_midleft, 0) + 1) / (
                            self.trigram_counts.get(left_context[-2:] + candidate, 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_4 * 0.25 * fourgram_midleft_prob
                    
                    if len(left_context) >= 1 and len(right_context) >= 2:
                        fourgram_midright = left_context[-1] + candidate + right_context[:2]
                        fourgram_midright_prob = (self.fourgram_counts.get(fourgram_midright, 0) + 1) / (
                            self.trigram_counts.get(candidate + right_context[:2], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_4 * 0.25 * fourgram_midright_prob
                    
                    if len(right_context) >= 3:
                        fourgram_right = candidate + right_context[:3]
                        fourgram_right_prob = (self.fourgram_counts.get(fourgram_right, 0) + 1) / (
                            self.trigram_counts.get(right_context[:3], 0) + len(self.unigram_counts)
                        )
                        score += self.lambda_4 * 0.25 * fourgram_right_prob

                    if score > best_score:
                        best_score = score
                        best_char = candidate

                # Calculate score for the original character
                original_score = 0

                # Unigram probability
                original_unigram_prob = (self.unigram_counts.get(char, 0) + 1) / (
                    sum(self.unigram_counts.values()) + len(self.unigram_counts)
                )
                original_score += self.lambda_1 * original_unigram_prob

                # Bigram probabilities
                if len(left_context) > 0:
                    original_bigram_left = left_context[-1] + char
                    original_bigram_left_prob = (self.bigram_counts.get(original_bigram_left, 0) + 1) / (
                        self.unigram_counts.get(left_context[-1], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_2 * 0.5 * original_bigram_left_prob
                
                if len(right_context) > 0:
                    original_bigram_right = char + right_context[0]
                    original_bigram_right_prob = (self.bigram_counts.get(original_bigram_right, 0) + 1) / (
                        self.unigram_counts.get(char, 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_2 * 0.5 * original_bigram_right_prob

                # Trigram probabilities
                if len(left_context) >= 2:
                    original_trigram_left = left_context[-2:] + char
                    original_trigram_left_prob = (self.trigram_counts.get(original_trigram_left, 0) + 1) / (
                        self.bigram_counts.get(left_context[-2:], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_3 * 0.3 * original_trigram_left_prob
                
                if len(left_context) >= 1 and len(right_context) >= 1:
                    original_trigram_mid = left_context[-1] + char + right_context[0]
                    original_trigram_mid_prob = (self.trigram_counts.get(original_trigram_mid, 0) + 1) / (
                        self.bigram_counts.get(left_context[-1] + char, 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_3 * 0.4 * original_trigram_mid_prob
                
                if len(right_context) >= 2:
                    original_trigram_right = char + right_context[:2]
                    original_trigram_right_prob = (self.trigram_counts.get(original_trigram_right, 0) + 1) / (
                        self.bigram_counts.get(char + right_context[0], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_3 * 0.3 * original_trigram_right_prob

                # 4-gram probabilities
                if len(left_context) >= 3:
                    original_fourgram_left = left_context[-3:] + char
                    original_fourgram_left_prob = (self.fourgram_counts.get(original_fourgram_left, 0) + 1) / (
                        self.trigram_counts.get(left_context[-3:], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_4 * 0.25 * original_fourgram_left_prob
                
                if len(left_context) >= 2 and len(right_context) >= 1:
                    original_fourgram_midleft = left_context[-2:] + char + right_context[0]
                    original_fourgram_midleft_prob = (self.fourgram_counts.get(original_fourgram_midleft, 0) + 1) / (
                        self.trigram_counts.get(left_context[-2:] + char, 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_4 * 0.25 * original_fourgram_midleft_prob
                
                if len(left_context) >= 1 and len(right_context) >= 2:
                    original_fourgram_midright = left_context[-1] + char + right_context[:2]
                    original_fourgram_midright_prob = (self.fourgram_counts.get(original_fourgram_midright, 0) + 1) / (
                        self.trigram_counts.get(char + right_context[:2], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_4 * 0.25 * original_fourgram_midright_prob
                
                if len(right_context) >= 3:
                    original_fourgram_right = char + right_context[:3]
                    original_fourgram_right_prob = (self.fourgram_counts.get(original_fourgram_right, 0) + 1) / (
                        self.trigram_counts.get(right_context[:3], 0) + len(self.unigram_counts)
                    )
                    original_score += self.lambda_4 * 0.25 * original_fourgram_right_prob

                # Only replace if the new score is significantly better
                threshold = 1.2 + self.error_probs.get(char, 0) * 3  # Dynamic threshold based on error probability
                if best_score > original_score * threshold:
                    corrected_text[i] = best_char

        return ''.join(corrected_text)

    def _correct_with_ml(self, text: str) -> str:
        """
        使用机器学习模型纠正文本

        Args:
            text: 输入文本

        Returns:
            纠正后的文本
        """
        if not SKLEARN_AVAILABLE or self.detect_model is None:
            print("ML models not available, falling back to n-gram method")
            return self._correct_with_ngram(text)
        
        # 检测整个文本是否有错误，但不直接返回（用于调试）
        text_vec = self.detect_vectorizer.transform([text])
        text_has_error_prob = self.detect_model.predict_proba(text_vec)[0][1]  # 错误的概率
        debug_info = {
            'text_error_prob': text_has_error_prob,
            'chars_examined': 0,
            'chars_corrected': 0
        }
        
        # 修正：不再使用全局错误检测，而改为字符级别的错误检测
        # 但增加整体检测的参考信息
        corrected_text = list(text)
        window_size = 2  # 与训练时保持一致
        
        # 对每个字符位置进行检查和可能的纠正
        for i in range(len(text)):
            # 提取上下文特征
            left_context = text[max(0, i - window_size):i]
            right_context = text[min(i + 1, len(text)):min(i + window_size + 1, len(text))]
            
            # 填充上下文
            if len(left_context) < window_size:
                left_context = ' ' * (window_size - len(left_context)) + left_context
            if len(right_context) < window_size:
                right_context = right_context + ' ' * (window_size - len(right_context))
            
            # 组合特征
            char_feature = left_context + text[i] + right_context
            
            # 判断当前字符是否可能有错误
            # 使用错误概率和已知错误字符信息
            is_error = False
            
            # 检查是否是常见错误字符
            if text[i] in self.error_probs and self.error_probs[text[i]] > 0.005:  # 降低错误概率阈值
                is_error = True
            
            # 检查是否在字符纠正字典中
            if text[i] in self.char_corrections and len(self.char_corrections[text[i]]) > 0:
                is_error = True
            
            # 检查上下文是否可能有错误
            context_key = left_context + '_' + right_context
            if (text[i], context_key) in self.confusion_matrix and len(self.confusion_matrix[(text[i], context_key)]) > 0:
                is_error = True
            
            # 全局错误概率大于阈值，增加检查的概率
            if text_has_error_prob > 0.3:  # 如果整个文本有错误概率较高
                is_error = True  # 扩大检查范围
            
            debug_info['chars_examined'] += 1
            
            # 如果可能有错误，则使用纠正模型进行纠正
            if is_error and self.correct_model is not None:
                char_feature_vec = self.correct_vectorizer.transform([char_feature])
                
                # 计算纠正概率
                correction_probs = self.correct_model.predict_proba(char_feature_vec)[0]
                
                # 获取预测的类别（索引）
                top_classes_idx = correction_probs.argsort()[-5:][::-1]  # 获取前5个最可能的类别
                top_classes = [self.correct_model.classes_[idx] for idx in top_classes_idx]
                top_probs = [correction_probs[idx] for idx in top_classes_idx]
                
                # 检查原字符是否在顶部预测中
                original_char = text[i]
                if original_char in top_classes:
                    original_idx = top_classes.index(original_char)
                    original_prob = top_probs[original_idx]
                    
                    # 再次降低阈值，减少对原字符的偏好
                    # 从1.2降低到1.1，从0.5降低到0.4
                    if top_probs[0] > original_prob * 1.1 and top_probs[0] > 0.4 and top_classes[0] != original_char:
                        corrected_text[i] = top_classes[0]
                        debug_info['chars_corrected'] += 1
                else:
                    # 原字符不在顶部预测中，也降低阈值
                    # 从0.5降低到0.4
                    if top_probs[0] > 0.4:
                        corrected_text[i] = top_classes[0]
                        debug_info['chars_corrected'] += 1
                
                # 使用相似度矩阵辅助判断（如果可用）
                if original_char in self.visual_similarity:
                    for candidate in self.visual_similarity[original_char]:
                        if candidate in top_classes:
                            candidate_idx = top_classes.index(candidate)
                            # 再次降低形近字的阈值要求
                            if top_probs[candidate_idx] > top_probs[0] * 0.6:
                                corrected_text[i] = candidate
                                debug_info['chars_corrected'] += 1
                                break
        
        # 如果整个文本被检测为有错误，但没有做任何修改，尝试强制修正
        if text_has_error_prob > 0.7 and debug_info['chars_corrected'] == 0:
            # 使用ngram模型作为后备
            return self._correct_with_ngram(text)
        
        result = ''.join(corrected_text)
        
        # 如果没有做任何修改，但预测为有错误，检查混淆矩阵
        if result == text and text_has_error_prob > 0.5:
            # 查找常见错误字符
            for i, char in enumerate(text):
                if char in self.confusion_pairs and self.confusion_pairs[char]:
                    correction = max(self.confusion_pairs[char].items(), key=lambda x: x[1])[0]
                    if self.confusion_pairs[char][correction] > 3:  # 只有有足够证据才修改
                        corrected_text[i] = correction
                        debug_info['chars_corrected'] += 1
        
        # 添加调试信息
        # print(f"Input: {text}")
        # print(f"Output: {''.join(corrected_text)}")
        # print(f"Text error probability: {text_has_error_prob:.4f}")
        # print(f"Chars examined: {debug_info['chars_examined']}")
        # print(f"Chars corrected: {debug_info['chars_corrected']}")
        
        return ''.join(corrected_text)
