#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chinese-text-correction",
    version="1.0.0",
    author="ML2025 NLP Team",
    author_email="author@example.com",
    description="中文文本纠错系统 (Chinese Text Correction System)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chinese-text-correction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "tqdm>=4.62.0",
        "jieba>=0.42.1",
        "Levenshtein>=0.20.0",
        "matplotlib>=3.5.0",
        "tokenizers>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "ctc-main=main:main",
            "ctc-analyze=run_analysis:main",
            "ctc-interactive=test_single_sentence:main",
        ],
    },
) 