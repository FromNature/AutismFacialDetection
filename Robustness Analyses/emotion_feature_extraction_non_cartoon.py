"""
情绪特征提取 - 排除看动画片任务后的特征提取
参考 emotion_feature_extraction_cartoon.py
提取除看动画片任务外的所有时间段的情绪特征
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path

def get_interference_frames(subject, events_dir="result/Events"):
    """
    获取干扰帧位置（口罩和动画片）
    参考 faceAU_analysis.ipynb 中的实现
    
    Parameters:
    -----------
    subject : str
        受试者姓名
    events_dir : str
        事件文件目录
    
    Returns:
    --------
    list: 干扰帧位置列表 [[start_frame, end_frame], ...]
    """
    from lib.EventTracking import Events
    
    # 读取事件文件
    event_file = os.path.join(events_dir, f"{subject}.json")
    if not os.path.exists(event_file):
        return []
    
    # 加载事件数据
    events = Events(event_file)
    
    # 过滤干扰事件（口罩和动画片）
    interference_events = events.filter_by_label(stay_labels=['口罩', '自选动画', '天鹅动画'])
    
    # 获取干扰帧位置列表
    if not interference_events.events:  # 如果没有干扰事件
        return []
    
    # 获取干扰帧位置列表 [[start_frame, end_frame], ...]
    interference_frames = interference_events.get_frame_list(sr=50)  # 视频帧率为50fps
    return interference_frames

def smooth_emotions2(emotion_series, window_size=10, min_duration=10):
    """
    Smooth emotion sequence using rule-based method
    window_size: Sliding window size
    min_duration: Minimum duration frames
    """
    emotions = emotion_series.values
    smoothed = emotions.copy()
    n = len(emotions)

    # 1. Use sliding window majority voting for initial smoothing
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window = emotions[start:end]

        # If current frame is valid emotion, perform majority voting smoothing
        if emotions[i] != 'undefined':  # Only process when current emotion is not 'undefined'
            # Get valid emotions in window
            valid_emotions = window[window != 'undefined']
            
            # If there are valid emotions, perform majority voting
            if len(valid_emotions) > 0:
                unique, counts = np.unique(valid_emotions, return_counts=True)
                majority = unique[counts.argmax()]
                smoothed[i] = majority
            else:
                smoothed[i] = emotions[i]
        else:
            smoothed[i] = emotions[i]  # No processing for undefined emotions

    # 2. Split emotions with duration less than min_duration frames, loop until no segments smaller than min_duration
    while True:
        i = 0
        modified = False  # Record whether any modifications were made
        while i < n - 1:
            # Find consecutive same emotion segments
            start = i
            while i < n - 1 and smoothed[i] == smoothed[i + 1]:
                i += 1
            end = i + 1  # Current emotion segment end position
            
            # If current segment duration is less than min_duration, split it
            if end - start < min_duration:
                # Determine previous and next emotion categories
                if start > 0:
                    prev_emotion = smoothed[start - 1]
                else:
                    prev_emotion = 'undefined'

                if end < n:
                    next_emotion = smoothed[end] 
                else:
                    next_emotion = 'undefined'

                # Split current segment, first half goes to previous emotion, second half goes to next emotion
                mid = start + (end - start) // 2
                smoothed[start:mid] = prev_emotion
                smoothed[mid:end] = next_emotion

                modified = True  # Mark as modified
                break  # Exit loop for next round of checking

            i += 1

        # If no modifications were made, splitting is complete, exit loop
        if not modified:
            break

    return pd.Series(smoothed, index=emotion_series.index)

def extract_emotion_features_non_cartoon(feature_type, emotion_dir="result/Emotions_smooth2", 
                                         output_dir="result/Emotion_Features_Non_Cartoon", 
                                         remove_outlier=True):
    """
    提取排除看动画片任务后的情绪特征
    
    Parameters:
    -----------
    feature_type : str
        特征类型：'GEV', 'Frequency', 'Duration', 'Probability', 'TransWithSelf', 'TransWithOutSelf'
    emotion_dir : str
        情绪数据目录
    output_dir : str
        输出目录
    remove_outlier : bool
        是否移除异常值
    """
    # 定义常量
    EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    FRAME_DURATION = 20  # 20 milliseconds per frame
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取人口统计学数据
    demo_df = pd.read_csv("result/demographics.csv", encoding='utf-8-sig')
    
    # 获取所有情绪CSV文件
    csv_files = [f for f in os.listdir(emotion_dir) if f.endswith('.csv')]
    total_subjects = len(csv_files)
    
    print(f"\n开始提取{feature_type}特征（排除看动画片任务）...")
    print(f"共{total_subjects}个受试者")
    
    # 存储所有受试者的结果
    results = []
    
    for i, file in enumerate(csv_files, 1):
        try:
            # 获取受试者姓名
            subject_name = os.path.splitext(file)[0]
            
            print(f"\r处理进度：{i}/{total_subjects} ({i/total_subjects*100:.1f}%) - 当前处理：{subject_name}", 
                  end='', flush=True)
            
            # 读取情绪数据
            emotion_file = os.path.join(emotion_dir, file)
            df = pd.read_csv(emotion_file, skiprows=2, encoding='utf-8-sig')
            
            # 获取干扰帧位置
            interference_frames = get_interference_frames(subject_name)
            
            # 创建干扰帧掩码（True表示非干扰帧）
            interference_mask = np.ones(len(df), dtype=bool)
            if interference_frames:  # 只有当存在干扰帧时才更新掩码
                for start_frame, end_frame in interference_frames:
                    start_frame = max(0, min(start_frame, len(df) - 1))
                    end_frame = max(0, min(end_frame, len(df) - 1))
                    interference_mask[start_frame:end_frame+1] = False
            
            # 筛选有效数据：非干扰帧
            valid_mask = interference_mask
            
            if valid_mask.sum() == 0:
                continue
            
            # 筛选有效帧的数据
            df_filtered = df[valid_mask].copy().reset_index(drop=True)
            
            # 过滤无效情绪
            df_filtered = df_filtered[df_filtered['Emotion'].isin(EMOTIONS)]
            
            if len(df_filtered) == 0:
                continue
            
            # 平滑情绪序列
            df_filtered['SmoothedEmotion'] = smooth_emotions2(df_filtered['Emotion'])  # 使用默认参数 window_size=10, min_duration=10
            
            # 计算总时长（秒）
            total_duration = len(df_filtered) * FRAME_DURATION / 1000
            
            # 初始化特征字典
            subject_features = {'Person': subject_name}
            
            # 计算GEV特征
            if feature_type == 'GEV':
                gev_series = df_filtered.groupby('SmoothedEmotion').size() / len(df_filtered)
                for emotion in EMOTIONS:
                    gev = gev_series.get(emotion, 0)  # 如果情绪不存在，返回0
                    if emotion == 'happy':
                        subject_features[f'GEV_sad'] = gev
                    elif emotion == 'sad':
                        subject_features[f'GEV_happy'] = gev
                    else:
                        subject_features[f'GEV_{emotion}'] = gev
            
            # 计算出现频率（每秒变化次数）
            elif feature_type == 'Frequency':
                emotion_sequence = df_filtered['SmoothedEmotion'].values
                # 计算每种情绪的出现频率
                for emotion in EMOTIONS:
                    changes = 0
                    for j in range(1, len(emotion_sequence)):
                        if emotion_sequence[j] == emotion and emotion_sequence[j-1] != emotion:
                            changes += 1
                    # 计算频率（每秒变化次数）
                    frequency = changes / total_duration if total_duration > 0 else 0
                    if emotion == 'happy':
                        subject_features[f'Frequency_sad'] = frequency
                    elif emotion == 'sad':
                        subject_features[f'Frequency_happy'] = frequency
                    else:
                        subject_features[f'Frequency_{emotion}'] = frequency
            
            # 计算平均持续时间
            elif feature_type == 'Duration':
                # 初始化所有情绪的持续时间为0
                for emotion in EMOTIONS:
                    subject_features[f'Duration_{emotion}'] = 0
                # 检测情绪变化点
                emotion_segments = df_filtered['SmoothedEmotion'] != df_filtered['SmoothedEmotion'].shift()
                # 计算每种情绪的片段数
                emotion_runs = df_filtered[emotion_segments]['SmoothedEmotion'].value_counts()
                # 计算每种情绪的总帧数
                emotion_total_frames = df_filtered.groupby('SmoothedEmotion').size()
                # 计算每种情绪的平均持续时间（毫秒）
                for emotion in EMOTIONS:
                    if emotion in emotion_runs.index and emotion in emotion_total_frames.index:
                        avg_duration = (emotion_total_frames[emotion] / emotion_runs[emotion]) * FRAME_DURATION
                        subject_features[f'Duration_{emotion}'] = avg_duration
                    else:
                        subject_features[f'Duration_{emotion}'] = 0
            
            # 计算平均识别概率
            elif feature_type == 'Probability':
                mean_probability = df_filtered.groupby('SmoothedEmotion')['Probability'].mean().round(2)
                std_probability = df_filtered.groupby('SmoothedEmotion')['Probability'].std().round(2)
                for emotion in EMOTIONS:
                    mean = mean_probability.get(emotion, 0)  # 如果情绪不存在，返回0
                    std = std_probability.get(emotion, 0)
                    subject_features[f'Probability_mean_{emotion}'] = mean
                    subject_features[f'Probability_std_{emotion}'] = std
            
            # 状态转移矩阵（包含自转移，用于弦图）
            elif feature_type == 'TransWithSelf':
                transition_counts = pd.crosstab(
                    df_filtered['SmoothedEmotion'], 
                    df_filtered['SmoothedEmotion'].shift(-1), 
                    normalize='index'
                )
                # 添加转移概率
                for e1 in EMOTIONS:
                    for e2 in EMOTIONS:
                        subject_features[f'TransWithSelf_{e1}_to_{e2}'] = transition_counts.get(e2, {}).get(e1, 0)
            
            # 状态转移矩阵（排除自转移，用于统计分析）
            elif feature_type == 'TransWithOutSelf':
                current_emotions = df_filtered['SmoothedEmotion'].iloc[:-1].reset_index(drop=True)  # 移除最后一个
                next_emotions = df_filtered['SmoothedEmotion'].iloc[1:].reset_index(drop=True)      # 移除第一个
                # 只保留不同情绪之间的转移
                mask = current_emotions != next_emotions
                current_emotions = current_emotions[mask]
                next_emotions = next_emotions[mask]
                # 计算转移概率矩阵
                transition_counts = pd.crosstab(
                    current_emotions,
                    next_emotions,
                    normalize='index'
                )
                # 添加转移概率
                for e1 in EMOTIONS:
                    for e2 in EMOTIONS:
                        subject_features[f'TransWithOutSelf_{e1}_to_{e2}'] = transition_counts.get(e2, {}).get(e1, 0)
            
            # 添加到结果列表
            results.append(subject_features)
            
        except Exception as e:
            print(f"\n警告: 处理{subject_name}时出错 - {str(e)}")
            continue
    
    if not results:
        print(f"\n错误: 没有找到{feature_type}特征的有效数据")
        return None
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 读取人口统计学数据并合并
    info_cols = ['姓名', '组别', 'ABC', 'S1', 'R', 'B', 'L', 'S2', '克氏', 'Age']
    demographic_df = demo_df[info_cols].copy()
    results_df = pd.merge(demographic_df, results_df, left_on='姓名', right_on='Person', how='inner')
    results_df = results_df.drop('Person', axis=1)
    
    # 获取特征列
    feature_cols = [col for col in results_df.columns if col.startswith(feature_type)]
    
    # 移除异常值
    if remove_outlier:
        for col in feature_cols:
            for group in [0, 1]:
                mask = results_df['组别'] == group
                group_data = results_df.loc[mask, col]  
                q1 = group_data.quantile(0.25)
                q3 = group_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                results_df.loc[mask & (group_data < lower_bound), col] = np.nanmedian(group_data)
                results_df.loc[mask & (group_data > upper_bound), col] = np.nanmedian(group_data)
    
    # 保存结果
    output_file = os.path.join(output_dir, f'{feature_type}_non_cartoon.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{feature_type}特征提取完成!")
    print(f"TD组: {len(results_df[results_df['组别'] == 0])}人")
    print(f"ASD组: {len(results_df[results_df['组别'] == 1])}人")
    print(f"结果已保存至: {output_file}")
    
    return results_df

def extract_all_emotion_features_non_cartoon(emotion_dir="result/Emotions_smooth2",
                                             output_dir="result/Emotion_Features_Non_Cartoon",
                                             remove_outlier=True):
    """
    提取所有类型的情绪特征（排除看动画片任务）
    
    Parameters:
    -----------
    emotion_dir : str
        情绪数据目录
    output_dir : str
        输出目录
    remove_outlier : bool
        是否移除异常值
    """
    feature_types = ['GEV', 'Frequency', 'Duration', 'Probability', 'TransWithSelf', 'TransWithOutSelf']
    
    print("="*80)
    print("开始提取排除看动画片任务后的所有情绪特征")
    print("="*80)
    
    all_results = {}
    
    for feature_type in feature_types:
        result_df = extract_emotion_features_non_cartoon(
            feature_type=feature_type,
            emotion_dir=emotion_dir,
            output_dir=output_dir,
            remove_outlier=remove_outlier
        )
        if result_df is not None:
            all_results[feature_type] = result_df
    
    print("\n" + "="*80)
    print("所有特征提取完成!")
    print("="*80)
    
    return all_results

def merge_emotion_features_non_cartoon(feature_dir="result/Emotion_Features_Non_Cartoon", 
                                        output_file="result/machine_learning/merged_emotion_features_non_cartoon.csv"):
    """
    合并排除看动画片任务后的所有情绪特征，用于模型分类
    参考 emotion_analysis.ipynb 中的 merge_emotion_features 方法
    
    Parameters:
    -----------
    feature_dir : str
        特征文件目录
    output_file : str
        输出文件路径
    """
    # 读取各种特征文件
    print("正在读取特征文件...")
    gev_df = pd.read_csv(os.path.join(feature_dir, 'GEV_non_cartoon.csv'), encoding='utf-8-sig')
    duration_df = pd.read_csv(os.path.join(feature_dir, 'Duration_non_cartoon.csv'), encoding='utf-8-sig')
    frequency_df = pd.read_csv(os.path.join(feature_dir, 'Frequency_non_cartoon.csv'), encoding='utf-8-sig')
    trans_df = pd.read_csv(os.path.join(feature_dir, 'TransWithOutSelf_non_cartoon.csv'), encoding='utf-8-sig')
    
    # 检查并统一列名（处理可能的Person/姓名列名不一致）
    for df_name, df in [('GEV', gev_df), ('Duration', duration_df), ('Frequency', frequency_df), ('TransWithOutSelf', trans_df)]:
        if 'Person' in df.columns and '姓名' not in df.columns:
            df.rename(columns={'Person': '姓名'}, inplace=True)
            print(f"  {df_name}: 已将'Person'列重命名为'姓名'")
        elif 'Person' in df.columns and '姓名' in df.columns:
            df.drop(columns=['Person'], inplace=True)
            print(f"  {df_name}: 已删除重复的'Person'列")
    
    # 使用基础信息列作为base（从duration_df获取）
    base_cols = ['组别', 'ABC', 'S1', 'R', 'B', 'L', 'S2', '克氏', 'Age', '姓名']
    # 只选择存在的列
    available_base_cols = [col for col in base_cols if col in duration_df.columns]
    result_df = duration_df[available_base_cols].copy()
    
    print(f"基础列: {available_base_cols}")
    
    # GEV特征
    print("合并GEV特征...")
    gev_feature_cols = [col for col in gev_df.columns if col not in available_base_cols]
    if '姓名' in gev_df.columns:
        result_df = result_df.merge(
            gev_df[['姓名'] + gev_feature_cols],
            on='姓名',
            validate='1:1'
        )
    else:
        raise KeyError(f"GEV文件中缺少'姓名'列，当前列名: {gev_df.columns.tolist()}")
    
    # Duration特征（已经在base中，但需要合并特征列）
    print("合并Duration特征...")
    duration_feature_cols = [col for col in duration_df.columns if col not in available_base_cols]
    if duration_feature_cols:
        result_df = result_df.merge(
            duration_df[['姓名'] + duration_feature_cols],
            on='姓名',
            validate='1:1'
        )
    
    # Frequency特征
    print("合并Frequency特征...")
    frequency_feature_cols = [col for col in frequency_df.columns if col not in available_base_cols]
    if '姓名' in frequency_df.columns:
        result_df = result_df.merge(
            frequency_df[['姓名'] + frequency_feature_cols],
            on='姓名',
            validate='1:1'
        )
    else:
        raise KeyError(f"Frequency文件中缺少'姓名'列，当前列名: {frequency_df.columns.tolist()}")
    
    # Transition概率特征
    print("合并TransWithOutSelf特征...")
    trans_feature_cols = [col for col in trans_df.columns if col not in available_base_cols]
    if '姓名' in trans_df.columns:
        result_df = result_df.merge(
            trans_df[['姓名'] + trans_feature_cols],
            on='姓名',
            validate='1:1'
        )
    else:
        raise KeyError(f"TransWithOutSelf文件中缺少'姓名'列，当前列名: {trans_df.columns.tolist()}")
    
    # 重命名列
    result_df = result_df.rename(columns={
        '组别': 'group'
    })
    
    # 删除不需要的列
    cols_to_drop = ['S1', 'R', 'B', 'L', 'S2', 'Age']
    cols_to_drop = [col for col in cols_to_drop if col in result_df.columns]
    result_df = result_df.drop(columns=cols_to_drop)
    
    # 重新排序列：基本信息列在前
    ordered_cols = ['姓名', 'group', 'ABC', '克氏']
    ordered_cols = [col for col in ordered_cols if col in result_df.columns]
    other_cols = [col for col in result_df.columns if col not in ordered_cols]
    result_df = result_df[ordered_cols + other_cols]
    
    # 创建输出目录
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存汇总结果
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"特征汇总完成，已保存至：{output_file}")
    
    # 检查数据丢失
    print(f"原始数据行数: {len(duration_df)}")
    print(f"合并后数据行数: {len(result_df)}")
    print(f"数据维度: {result_df.shape}")
    
    return result_df

if __name__ == "__main__":
    # 先提取所有特征
    print("="*80)
    print("第一步：提取情绪特征（排除看动画片任务）")
    print("="*80)
    all_results = extract_all_emotion_features_non_cartoon(
        emotion_dir="result/Emotions_smooth2",
        output_dir="result/Emotion_Features_Non_Cartoon",
        remove_outlier=True
    )
    
    # 然后合并所有特征用于模型分类
    print("\n" + "="*80)
    print("第二步：合并特征用于模型分类（排除看动画片任务）")
    print("="*80)
    merged_df = merge_emotion_features_non_cartoon(
        feature_dir="result/Emotion_Features_Non_Cartoon",
        output_file="result/machine_learning/merged_emotion_features_non_cartoon.csv"
    )
    
    print("\n" + "="*80)
    print("所有特征提取和合并完成！")
    print("="*80)

