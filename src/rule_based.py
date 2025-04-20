#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rule-based corrector for Chinese Text Correction task.
This module implements rule-based methods for correcting errors in Chinese text.
"""

import re
import json
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

# Try to import optional dependencies
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("Warning: jieba not available. Some features will be disabled.")


class RuleBasedCorrector:
    """
    A rule-based corrector for Chinese text.
    """

    def __init__(self):
        """
        Initialize the rule-based corrector.
        """
        # Common confusion pairs (similar characters)
        self.confusion_pairs = {}
        
        # 上下文相关的混淆词典
        self.context_confusion = {}

        # Punctuation rules
        self.punctuation_rules = {
            '，。': '。',  # 顿号+句号 -> 句号
            '。，': '。',  # 句号+顿号 -> 句号
            '!。': '！',   # 感叹号+句号 -> 感叹号
            '。!': '！',   # 句号+感叹号 -> 感叹号
            '?。': '？',   # 问号+句号 -> 问号
            '。?': '？',   # 句号+问号 -> 问号
            ',.': '.',     # 英文逗号+句号 -> 句号
            '.,': '.',     # 英文句号+逗号 -> 句号
            '  ': ' ',     # 两个空格 -> 一个空格
        }

        # 标点符号转换（全角和半角转换）
        self.punctuation_conversion = {
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            ';': '；',
            ':': '：',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】',
            '"': '"',
            "'": "'",
        }

        # Grammar rules
        self.grammar_rules = {}

        # Common word pairs (for word-level correction)
        self.word_confusion = {}

        # Quantifier-noun pairs (for measure word correction)
        self.quantifier_noun_pairs = {}
        
        # 常见错字
        self.common_typos = {
            '的地得': {'的': ['地', '得'], '地': ['的', '得'], '得': ['的', '地']},
            '也都': {'也': ['都'], '都': ['也']},
            '吗嘛': {'吗': ['嘛'], '嘛': ['吗']},
            '里离': {'里': ['离'], '离': ['里']},
            '在再': {'在': ['再'], '再': ['在']},
            '别贝': {'别': ['贝'], '贝': ['别']},
            '那哪': {'那': ['哪'], '哪': ['那']},
            '称成': {'称': ['成'], '成': ['称']},
            '它牠祂': {'它': ['牠', '祂'], '牠': ['它', '祂'], '祂': ['它', '牠']},
            '这这个这些': {'这': ['这个', '这些'], '这个': ['这', '这些'], '这些': ['这', '这个']},
            '了啦': {'了': ['啦'], '啦': ['了']},
            '他她它': {'他': ['她', '它'], '她': ['他', '它'], '它': ['他', '她']},
        }
        
        # 常见成语修正
        self.idiom_corrections = {}
        
        # 新增：冗余词模式
        self.redundant_patterns = {
            '送人给': '送给',
            '来到了到': '来到了',
            '说道说': '说道',
            '走去到': '走到',
            '回去到': '回到',
            '然后接着': '然后',
            '立刻马上': '立刻',
            '非常很': '非常',
            '原本是': '原本',
            '已经是': '已经',
            '曾经是': '曾经',
            '就是是': '就是',
            '只有有': '只有',
            '只是是': '只是',
            '因为所以': '因为',
            '如果的话': '如果',
            '将会': '将',
            '正在着': '正在',
            '特别地': '特别',
        }
        
        # 新增：词序错误
        self.word_order_errors = {
            '不但但是': '不但',
            '很非常': '非常',
            '地方哪里': '哪里',
            '以所为': '以为所',
            '先已经': '已经',
            '了没有': '没有了',
            '们我': '我们',
            '来去': '去来',
            '好很': '很好',
            '去回': '回去',
        }
        
        # 新增：缺词模式
        self.missing_word_patterns = {
            '因此': '因此，',
            '所以': '所以，',
            '但是': '但是，',
            '虽然': '虽然',  # 后文应有"但是"
            '不但': '不但',  # 后文应有"而且"
        }

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        self._extract_confusion_pairs(train_data)
        self._extract_punctuation_rules(train_data)
        self._extract_grammar_rules(train_data)
        self._extract_word_confusion(train_data)

    def _extract_confusion_pairs(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract character confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # Extract character-level confusion pairs from error examples
        for sample in train_data:
            if sample['label'] == 1:  # Only for sentences with errors
                source = sample['source']
                target = sample['target']

                # For character substitution errors (when lengths are equal)
                if len(source) == len(target):
                    for i, (s_char, t_char) in enumerate(zip(source, target)):
                        if s_char != t_char:
                            # Get context (surrounding characters)
                            left_context = source[max(0, i - 2) : i]
                            right_context = source[i + 1 : min(len(source), i + 3)]
                            context = left_context + '_' + right_context

                            # Add to confusion pairs with context
                            if s_char not in self.confusion_pairs:
                                self.confusion_pairs[s_char] = defaultdict(int)
                            self.confusion_pairs[s_char][t_char] += 1
                            
                            # 记录上下文，以便更精确的纠正
                            context_key = f"{left_context}_{right_context}"
                            if context_key not in self.context_confusion:
                                self.context_confusion[context_key] = defaultdict(int)
                            self.context_confusion[context_key][(s_char, t_char)] += 1

        # Filter confusion pairs to keep only the most common ones
        filtered_pairs = {}
        for wrong_char, corrections in self.confusion_pairs.items():
            # Keep only corrections that appear at least twice
            common_corrections = {correct: count for correct, count in corrections.items() if count >= 2}
            if common_corrections:
                filtered_pairs[wrong_char] = common_corrections

        self.confusion_pairs = filtered_pairs

    def _extract_punctuation_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract punctuation correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # 分析数据中的标点符号错误
        punc_patterns = {}
        
        for sample in train_data:
            if sample['label'] == 1:  # 只对有错误的样本
                source = sample['source']
                target = sample['target']
                
                # 提取标点符号模式
                for i in range(len(source) - 1):
                    if source[i:i+2] != target[i:i+2] and (self._is_punctuation(source[i]) or self._is_punctuation(source[i+1])):
                        src_pattern = source[i:i+2]
                        
                        # 找到对应的目标模式
                        for j in range(max(0, i-1), min(len(target), i+3)):
                            if self._is_punctuation(target[j]):
                                # 简单匹配，假设修正后的标点在相近位置
                                tgt_pattern = target[j]
                                
                                if src_pattern not in punc_patterns:
                                    punc_patterns[src_pattern] = defaultdict(int)
                                punc_patterns[src_pattern][tgt_pattern] += 1
        
        # 过滤并添加到标点规则
        for src_pattern, corrections in punc_patterns.items():
            if corrections:  # 如果有修正
                most_common = max(corrections.items(), key=lambda x: x[1])[0]
                if corrections[most_common] >= 2:  # 至少出现两次
                    self.punctuation_rules[src_pattern] = most_common
    
    def _is_punctuation(self, char: str) -> bool:
        """
        Check if a character is a punctuation mark.
        
        Args:
            char: Character to check.
            
        Returns:
            True if the character is a punctuation, False otherwise.
        """
        punctuations = ',.?!;:()[]{}"\'"\'、，。？！；：（）【】｛｝""'''
        return char in punctuations
        
    def _extract_grammar_rules(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract grammar correction rules from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        # 提取语法模式
        grammar_patterns = defaultdict(lambda: defaultdict(int))
        
        for sample in train_data:
            if sample['label'] == 1:
                source = sample['source']
                target = sample['target']
                
                # 提取常见的语法错误模式
                # 通过n-gram方法提取
                for n in range(2, 5):  # 2到4元语法
                    for i in range(len(source) - n + 1):
                        src_ngram = source[i:i+n]
                        
                        # 寻找目标文本中对应的修正
                        for j in range(max(0, i-2), min(len(target), i+n+2)):
                            if j + n <= len(target):
                                tgt_ngram = target[j:j+n]
                                if src_ngram != tgt_ngram and self._similar_ngrams(src_ngram, tgt_ngram):
                                    grammar_patterns[src_ngram][tgt_ngram] += 1
        
        # 过滤出常见的语法规则
        for src_pattern, corrections in grammar_patterns.items():
            most_common = max(corrections.items(), key=lambda x: x[1]) if corrections else (None, 0)
            if most_common[1] >= 3:  # 至少出现3次
                self.grammar_rules[src_pattern] = most_common[0]
    
    def _similar_ngrams(self, ngram1: str, ngram2: str) -> bool:
        """检查两个n-gram是否相似（有共同字符）"""
        common_chars = set(ngram1) & set(ngram2)
        return len(common_chars) >= min(len(ngram1), len(ngram2)) * 0.5
                
    def _extract_word_confusion(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Extract word-level confusion pairs from the training data.

        Args:
            train_data: List of dictionaries containing the training data.
        """
        if not JIEBA_AVAILABLE:
            print("Jieba not available, skipping word confusion extraction.")
            return
            
        word_pairs = defaultdict(lambda: defaultdict(int))
        
        for sample in train_data:
            if sample['label'] == 1:
                source = sample['source']
                target = sample['target']
                
                # 分词
                source_words = list(jieba.cut(source))
                target_words = list(jieba.cut(target))
                
                # 寻找单词级别的混淆
                for i, src_word in enumerate(source_words):
                    if len(src_word) > 1:  # 只关注多字词
                        for j, tgt_word in enumerate(target_words):
                            if len(tgt_word) > 1 and abs(i - j) <= 2:  # 位置相近
                                if src_word != tgt_word and self._similar_words(src_word, tgt_word):
                                    word_pairs[src_word][tgt_word] += 1
        
        # 过滤常见的词语混淆
        for src_word, corrections in word_pairs.items():
            most_common = max(corrections.items(), key=lambda x: x[1]) if corrections else (None, 0)
            if most_common[1] >= 2:  # 至少出现2次
                self.word_confusion[src_word] = most_common[0]
                
        # 收集成语
        self._extract_idioms(train_data)
    
    def _similar_words(self, word1: str, word2: str) -> bool:
        """检查两个词是否相似（编辑距离小）"""
        if abs(len(word1) - len(word2)) > 1:
            return False
            
        # 简单编辑距离
        common_chars = set(word1) & set(word2)
        return len(common_chars) >= min(len(word1), len(word2)) * 0.7
    
    def _extract_idioms(self, train_data: List[Dict[str, Any]]) -> None:
        """提取成语纠错规则"""
        for sample in train_data:
            if sample['label'] == 1:
                source = sample['source']
                target = sample['target']
                
                # 匹配可能的成语（4字词组）
                for i in range(len(source) - 3):
                    src_idiom = source[i:i+4]
                    
                    # 在目标文本中寻找相似的4字词组
                    for j in range(max(0, i-2), min(len(target), i+5)):
                        if j + 4 <= len(target):
                            tgt_idiom = target[j:j+4]
                            if src_idiom != tgt_idiom and self._similar_ngrams(src_idiom, tgt_idiom):
                                # 可能是成语修正
                                self.idiom_corrections[src_idiom] = tgt_idiom

    def correct(self, text: str) -> str:
        """
        Apply rule-based correction to the input text.

        Args:
            text: Input text to correct.

        Returns:
            Corrected text.
        """
        # 先进行初步纠正
        corrected = self._correct_punctuation(text)
        corrected = self._correct_redundant_words(corrected)
        corrected = self._correct_word_order(corrected)
        corrected = self._correct_confusion_chars(corrected)
        corrected = self._correct_grammar(corrected)
        corrected = self._correct_word_confusion(corrected)
        corrected = self._check_missing_words(corrected)
        
        return corrected

    def _correct_punctuation(self, text: str) -> str:
        """
        Correct punctuation errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected punctuation.
        """
        corrected_text = text
        
        # 1. 修正半角符号为全角符号（在中文语境中）
        for eng_punct, chn_punct in self.punctuation_conversion.items():
            # 只替换中文字符环境中的半角符号
            i = 0
            while i < len(corrected_text):
                if corrected_text[i] == eng_punct:
                    # 检查左右是否有中文字符
                    has_chinese = False
                    if i > 0 and self._is_chinese_char(corrected_text[i-1]):
                        has_chinese = True
                    if i < len(corrected_text) - 1 and self._is_chinese_char(corrected_text[i+1]):
                        has_chinese = True
                    
                    if has_chinese:
                        corrected_text = corrected_text[:i] + chn_punct + corrected_text[i+1:]
                
                i += 1
        
        # 2. 修正标点符号序列
        for punct_seq, correct_punct in self.punctuation_rules.items():
            corrected_text = corrected_text.replace(punct_seq, correct_punct)
        
        # 3. 标点符号前后空格的处理
        # 在中文中，标点符号前后通常不需要空格
        corrected_text = re.sub(r'([，。？！；：、]) ', r'\1', corrected_text)
        corrected_text = re.sub(r' ([，。？！；：、])', r'\1', corrected_text)
        
        return corrected_text
    
    def _is_chinese_char(self, char: str) -> bool:
        """检查是否为中文字符"""
        if not char:
            return False
        return '\u4e00' <= char <= '\u9fff'

    def _correct_confusion_chars(self, text: str) -> str:
        """
        Correct character confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected characters.
        """
        corrected_text = list(text)  # Convert to list for character-by-character editing

        # 1. 基于上下文的错字修正
        for i, char in enumerate(text):
            # 获取上下文
            left_context = text[max(0, i - 2) : i]
            right_context = text[i + 1 : min(len(text), i + 3)]
            context_key = f"{left_context}_{right_context}"
            
            # 基于上下文的修正
            if context_key in self.context_confusion:
                for (wrong, correct), count in self.context_confusion[context_key].items():
                    if char == wrong and count >= 2:  # 至少出现两次才进行修正
                        corrected_text[i] = correct
                        break
        
        # 2. 基于混淆对的修正
        for i, char in enumerate(text):
            if char in self.confusion_pairs and self.confusion_pairs[char]:
                # 获取最常见的修正
                correct_char = max(self.confusion_pairs[char].items(), key=lambda x: x[1])[0]
                
                # 应用规则
                # 处理"的地得"的情况
                if char in "的地得" and correct_char in "的地得":
                    # 判断是否需要修正
                    should_correct = False
                    
                    # '地'通常修饰副词，后面接动词
                    if char == '的' and correct_char == '地':
                        # 检查后面是否有动词特征
                        if i < len(text) - 1 and text[i+1] in "来去走跑跳说读写叫喊看听闻":
                            should_correct = True
                    
                    # '得'通常接在动词后作补语
                    elif char == '的' and correct_char == '得':
                        # 检查前面是否有动词特征
                        if i > 0 and text[i-1] in "说写走看读听站坐跑跳":
                            should_correct = True
                    
                    if should_correct:
                        corrected_text[i] = correct_char
                
                # 处理"在再"的情况
                elif char in "在再" and correct_char in "在再":
                    if char == '在' and correct_char == '再':
                        # '再'表示重复或者将来，通常前后有时间词
                        if i < len(text) - 1 and text[i+1] in "次第三来日后":
                            corrected_text[i] = '再'
                    elif char == '再' and correct_char == '在':
                        # '在'表示存在或位置
                        if i < len(text) - 1 and text[i+1] in "家中这那哪里外内":
                            corrected_text[i] = '在'
                
                # 处理'他她它'的情况
                elif char in "他她它" and correct_char in "他她它":
                    if i > 0:
                        # 根据上下文推断性别或物体
                        prev_chars = text[max(0, i-5):i]
                        
                        if any(word in prev_chars for word in ["女", "妈", "姐", "妹", "婆", "奶", "阿姨"]):
                            corrected_text[i] = '她'
                        elif any(word in prev_chars for word in ["桌", "椅", "车", "门", "窗", "书", "笔"]):
                            corrected_text[i] = '它'
                
                # 其他情况，只在高置信度时修正
                elif self.confusion_pairs[char][correct_char] > 5 and len(self.confusion_pairs[char]) == 1:
                    corrected_text[i] = correct_char
                
                # 修正错别字
                for typo_group, corrections in self.common_typos.items():
                    if char in typo_group:
                        for possible_correction in corrections.get(char, []):
                            # 简单的上下文判断
                            if (i > 0 and text[i-1:i+1] in self.grammar_rules and 
                                self.grammar_rules[text[i-1:i+1]][-1] == possible_correction):
                                corrected_text[i] = possible_correction
                                break

        return ''.join(corrected_text)

    def _correct_grammar(self, text: str) -> str:
        """
        Correct grammar errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected grammar.
        """
        corrected_text = text
        
        # 应用语法规则
        for error_pattern, correction in self.grammar_rules.items():
            if error_pattern in corrected_text:
                corrected_text = corrected_text.replace(error_pattern, correction)
        
        return corrected_text

    def _correct_word_confusion(self, text: str) -> str:
        """
        Correct word-level confusion errors in the text.

        Args:
            text: Input text.

        Returns:
            Text with corrected words.
        """
        if not JIEBA_AVAILABLE:
            return text
            
        # 分词
        words = list(jieba.cut(text))
        corrected_words = []
        
        # 应用词级别的纠错
        for word in words:
            if word in self.word_confusion:
                corrected_words.append(self.word_confusion[word])
            else:
                corrected_words.append(word)
        
        # 成语纠错
        corrected_text = ''.join(corrected_words)
        for wrong_idiom, correct_idiom in self.idiom_corrections.items():
            if wrong_idiom in corrected_text:
                corrected_text = corrected_text.replace(wrong_idiom, correct_idiom)
        
        return corrected_text

    def _correct_redundant_words(self, text: str) -> str:
        """
        修正冗余词错误
        
        Args:
            text: 输入文本
            
        Returns:
            修正后的文本
        """
        corrected = text
        
        # 应用冗余词模式
        for pattern, correction in self.redundant_patterns.items():
            corrected = corrected.replace(pattern, correction)
            
        # 特殊处理一些复杂情况
        # 例如：处理"把...送人给某人"的模式
        match = re.search(r'把(.*?)送人给', corrected)
        if match:
            corrected = corrected.replace(match.group(0), f'把{match.group(1)}送给')
            
        return corrected
        
    def _correct_word_order(self, text: str) -> str:
        """
        修正词序错误
        
        Args:
            text: 输入文本
            
        Returns:
            修正后的文本
        """
        corrected = text
        
        # 应用词序错误修正
        for error, correction in self.word_order_errors.items():
            corrected = corrected.replace(error, correction)
            
        return corrected
        
    def _check_missing_words(self, text: str) -> str:
        """
        检查并修正缺失的词语
        
        Args:
            text: 输入文本
            
        Returns:
            修正后的文本
        """
        corrected = text
        
        # 检查常见的缺词模式
        for pattern, correction in self.missing_word_patterns.items():
            # 如果是需要成对出现的词，如"虽然...但是"
            if pattern == '虽然' and '虽然' in corrected and '但是' not in corrected[corrected.index('虽然'):]:
                pos = corrected.index('虽然') + 2
                while pos < len(corrected) and corrected[pos] not in '，。！？、；：':
                    pos += 1
                if pos < len(corrected):
                    corrected = corrected[:pos] + '，但是' + corrected[pos:]
            
            # 如果是"不但...而且"
            elif pattern == '不但' and '不但' in corrected and '而且' not in corrected[corrected.index('不但'):]:
                pos = corrected.index('不但') + 2
                while pos < len(corrected) and corrected[pos] not in '，。！？、；：':
                    pos += 1
                if pos < len(corrected):
                    corrected = corrected[:pos] + '，而且' + corrected[pos:]
            
            # 简单情况：缺少标点符号
            elif pattern in ['因此', '所以', '但是'] and pattern in corrected:
                for match in re.finditer(pattern, corrected):
                    pos = match.end()
                    if pos < len(corrected) and corrected[pos] not in '，。！？、；：':
                        # 检查是否已经有标点
                        if not self._is_punctuation(corrected[pos]):
                            corrected = corrected[:pos] + '，' + corrected[pos:]
        
        return corrected
