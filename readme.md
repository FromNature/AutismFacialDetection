# Naturalistic Facial Dynamics Analysis for ASD / 孤独症自然社交面部特征分析项目

## Project Overview / 项目概述

Existing facial-expression studies in children with autism spectrum disorder (ASD) focus on discrete and task-driven facial measures. Such approaches cannot capture the sustained emotional fluctuations and ambiguous expressions observed in naturalistic interactions, which more directly reflect clinical phenotypes. To address this gap between subjective assessment and objective modeling, this study aims to quantify atypical facial expression patterns in children with ASD during naturalistic interactions, enabling ASD identification and severity assessment, providing a promising tool for large-scale ASD screening.

现有的孤独症谱系障碍（ASD）儿童面部表情研究主要关注离散和任务驱动的面部测量。这种方法无法捕捉自然社交互动中观察到的持续情绪波动和模糊表情，而这些更直接地反映了临床表型。为了弥补主观评估和客观建模之间的差距，本研究旨在量化ASD儿童在自然社交互动中的非典型面部表情模式，实现ASD的识别与患病程度评估，为大规模ASD筛查提供有前景的工具。

## Core Features / 核心特征

- **Emotion Variation (情绪变化)**: Temporal stability of emotional states (情绪状态的时间稳定性)
- **Expression Intensity (表情强度)**: Magnitude of facial muscle activation (面部肌肉激活的幅度)  
- **Facial Coordination (面部协调性)**: Synchrony across facial muscles (面部肌肉间的同步性)

These features integrate holistic and processual representations across coarse- and fine-grained levels, enabling detailed quantification of atypical expression patterns for ASD classification and symptom severity assessment.

这些特征整合了粗粒度和细粒度的整体性和过程性表征，能够详细量化ASD分类和症状严重程度评估的异常表达模式。

## File Structure / 文件结构

### Core Analysis Files / 核心分析文件
- **`emotion_analysis.ipynb`**: Emotion feature extraction and analysis, including global explained variance, frequency, duration, transformation matrix statistics / 情绪特征提取和分析，包括全局解释方差、频率、持续时间、转移矩阵统计特征
- **`faceAU_analysis.ipynb`**: Facial Action Unit (AU) analysis, including AU intensity changes and correlation analysis / 面部动作单元(AU)分析，包括AU强度变化和相关性分析
- **`demographic_analysis.ipynb`**: Demographic analysis, including age, scale scores, and session duration analysis / 人口统计学特征分析，包括年龄、量表分数、Session duration分析
- **`classify.ipynb`**: ASD/TD classification task using multiple feature combinations for performance comparison / ASD/TD分类任务，使用多种特征组合进行分类性能比较
- **`regression.ipynb`**: Symptom severity prediction for ABC and CABS scale scores / 症状严重程度预测，预测ABC和CABS量表分数
- **`tsne.py`**: t-SNE dimensionality reduction visualization for high-dimensional feature space / t-SNE降维可视化，用于高维特征空间的可视化

### Feature Extraction Scripts / 特征提取脚本
- **`emotion_extract_by_openvino.py`**: OpenVINO-based emotion recognition for real-time video emotion analysis / 基于OpenVINO的情绪识别，进行实时视频情绪分析
- **`faceAU_extract_by_openface.py`**: OpenFace-based AU extraction for facial action unit detection / 基于OpenFace的AU提取，检测面部动作单元

### Utility Libraries / 工具库
- **`lib/EventTracking.py`**: Event label processing tool for JSON event file parsing and time segment extraction / 事件标签处理工具，用于JSON事件文件解析和时间片段提取
- **`lib/ffhq_align.py`**: Face alignment tool / 人脸对齐工具
- **`lib/transfer_json.py`**: JSON data conversion tool / JSON数据转换工具

### Model Files / 模型文件
- **`model/face_landmarker_v2_with_blendshapes.task`**: Facial landmark detection model / 面部关键点检测模型

## Key Results / 主要结果

- **Classification Performance (分类性能)**: Best feature combination (EIC) accuracy 91.8%, AUC-ROC 0.962 / 最佳特征组合(EIC)准确率91.8%，AUC-ROC 0.962
- **Regression Performance (回归性能)**: ABC prediction R²=0.432, CABS prediction R²=0.400 / ABC预测R²=0.432，CABS预测R²=0.400

## Environment Requirements / 环境要求

- Python 3.8+, OpenVINO 2023.1.0, OpenFace 2.2.0
- scikit-learn, pandas, numpy, matplotlib
- GPU support (recommended) / GPU支持（推荐）

## Usage / 使用方法

```bash
# Feature Extraction / 特征提取
python emotion_extract_by_openvino.py
python faceAU_extract_by_openface.py

# Data Analysis / 数据分析
jupyter notebook emotion_analysis.ipynb
jupyter notebook classify.ipynb
```

## Citation / 引用

This research is based on naturalistic facial dynamics for quantitative clinical assessment of Autism Spectrum Disorder. Related paper is under preparation.

本研究基于自然社交面部动态特征进行孤独症谱系障碍的定量临床评估。相关论文正在准备中。
