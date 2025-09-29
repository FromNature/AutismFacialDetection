# 孤独症自然社交面部特征分析项目 / Naturalistic Facial Dynamics Analysis for ASD

## 项目概述 / Project Overview

现有的孤独症谱系障碍（ASD）儿童面部表情研究主要关注离散和任务驱动的面部测量。这种方法无法捕捉自然社交互动中观察到的持续情绪波动和模糊表情，而这些更直接地反映了临床表型。为了弥补主观评估和客观建模之间的差距，本研究旨在量化ASD儿童在自然社交互动中的非典型面部表情模式，实现ASD的识别与患病程度评估，为大规模ASD筛查提供有前景的工具。

Existing facial-expression studies in children with autism spectrum disorder (ASD) focus on discrete and task-driven facial measures. Such approaches cannot capture the sustained emotional fluctuations and ambiguous expressions observed in naturalistic interactions, which more directly reflect clinical phenotypes. To address this gap between subjective assessment and objective modeling, this study aims to quantify atypical facial expression patterns in children with ASD during naturalistic interactions, enabling ASD identification and severity assessment, providing a promising tool for large-scale ASD screening.

## 核心特征 / Core Features

- **情绪变化 (Emotion Variation)**: 情绪状态的时间稳定性 (Temporal stability of emotional states)
- **表情强度 (Expression Intensity)**: 面部肌肉激活的幅度 (Magnitude of facial muscle activation)  
- **面部协调性 (Facial Coordination)**: 面部肌肉间的同步性 (Synchrony across facial muscles)

这些特征整合了粗粒度和细粒度的整体性和过程性表征，能够详细量化ASD分类和症状严重程度评估的异常表达模式。

These features integrate holistic and processual representations across coarse- and fine-grained levels, enabling detailed quantification of atypical expression patterns for ASD classification and symptom severity assessment.

## 文件结构 / File Structure

### 核心分析文件 / Core Analysis Files
- **`emotion_analysis.ipynb`**: 情绪特征提取和分析，包括全局解释方差、频率、持续时间、转移矩阵统计特征 / Emotion feature extraction and analysis, including global explained variance, frequency, duration, transformation matrix statistics
- **`faceAU_analysis.ipynb`**: 面部动作单元(AU)分析，包括AU强度变化和相关性分析 / Facial Action Unit (AU) analysis, including AU intensity changes and correlation analysis
- **`demographic_analysis.ipynb`**: 人口统计学特征分析，包括年龄、量表分数、Session duration分析 / Demographic analysis, including age, scale scores, and session duration analysis
- **`classify.ipynb`**: ASD/TD分类任务，使用多种特征组合进行分类性能比较 / ASD/TD classification task using multiple feature combinations for performance comparison
- **`regression.ipynb`**: 症状严重程度预测，预测ABC和CABS量表分数 / Symptom severity prediction for ABC and CABS scale scores
- **`tsne.py`**: t-SNE降维可视化，用于高维特征空间的可视化 / t-SNE dimensionality reduction visualization for high-dimensional feature space

### 特征提取脚本 / Feature Extraction Scripts
- **`emotion_extract_by_openvino.py`**: 基于OpenVINO的情绪识别，进行实时视频情绪分析 / OpenVINO-based emotion recognition for real-time video emotion analysis
- **`faceAU_extract_by_openface.py`**: 基于OpenFace的AU提取，检测面部动作单元 / OpenFace-based AU extraction for facial action unit detection

### 工具库 / Utility Libraries
- **`lib/EventTracking.py`**: 事件标签处理工具，用于JSON事件文件解析和时间片段提取 / Event label processing tool for JSON event file parsing and time segment extraction
- **`lib/ffhq_align.py`**: 人脸对齐工具 / Face alignment tool
- **`lib/transfer_json.py`**: JSON数据转换工具 / JSON data conversion tool

### 模型文件 / Model Files
- **`model/face_landmarker_v2_with_blendshapes.task`**: 面部关键点检测模型 / Facial landmark detection model

## 主要结果 / Key Results

- **分类性能 / Classification Performance**: 最佳特征组合(EIC)准确率91.8%，AUC-ROC 0.962 / Best feature combination (EIC) accuracy 91.8%, AUC-ROC 0.962
- **回归性能 / Regression Performance**: ABC预测R²=0.432，CABS预测R²=0.400 / ABC prediction R²=0.432, CABS prediction R²=0.400

## 环境要求 / Environment Requirements

- Python 3.8+, OpenVINO 2023.1.0, OpenFace 2.2.0
- scikit-learn, pandas, numpy, matplotlib
- GPU支持（推荐）/ GPU support (recommended)

## 使用方法 / Usage

```bash
# 特征提取 / Feature Extraction
python emotion_extract_by_openvino.py
python faceAU_extract_by_openface.py

# 数据分析 / Data Analysis
jupyter notebook emotion_analysis.ipynb
jupyter notebook classify.ipynb
```

## 引用 / Citation

本研究基于自然社交面部动态特征进行孤独症谱系障碍的定量临床评估。相关论文正在准备中。

This research is based on naturalistic facial dynamics for quantitative clinical assessment of Autism Spectrum Disorder. Related paper is under preparation.
