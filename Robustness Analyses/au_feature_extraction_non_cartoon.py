"""
AU特征提取 - 排除看动画片任务后的特征提取
排除看动画片的片段，保留其他时间段，排除干扰片段
"""

import pandas as pd
import numpy as np
import os
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
        print(f"警告: 未找到{subject}的事件文件")
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

def replace_outliers(group_data, k=1.5):
    """
    使用IQR方法识别和处理异常值
    k=1.5为常用值，k=3.0为极端值标准
    """
    Q1 = group_data.quantile(0.25)
    Q3 = group_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    outlier_mask = (group_data < lower_bound) | (group_data > upper_bound)
    cleaned_data = group_data.copy()
    cleaned_data[outlier_mask] = group_data.median()  # 替换为中位数
    return cleaned_data

def extract_au_features_non_cartoon(emotion, output_dir="result/AU_Features_Non_Cartoon"):
    """
    提取指定情绪下排除看动画片任务后的AU特征
    
    Parameters:
    -----------
    emotion : str
        目标情绪（如'neutral', 'happy', 'sad', 'surprise', 'anger'）
    output_dir : str
        输出目录
    """
    # 读取数据
    au_dir = "result/AUs"
    emotion_dir = "result/Emotions_smooth2"
    demo_df = pd.read_csv("result/demographics.csv", encoding='utf-8-sig')
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 存储AU数据
    all_au_data = []
    total_subjects = len(demo_df)
    processed_subjects = {'TD': 0, 'ASD': 0}
    
    print(f"\n开始提取{emotion}情绪下排除看动画片任务后的AU强度数据...")
    
    # 遍历每个受试者
    for i, (_, row) in enumerate(demo_df.iterrows(), 1):
        subject = row['姓名']
        group = row['组别']
        
        print(f"\r处理进度：{i}/{total_subjects} ({i/total_subjects*100:.1f}%) - 当前处理：{subject}", 
              end='', flush=True)
        
        try:
            # 读取情绪数据
            emotion_file = os.path.join(emotion_dir, f"{subject}.csv")
            if not os.path.exists(emotion_file):
                continue
                
            with open(emotion_file, 'r', encoding='utf-8-sig') as f:
                next(f)
                next(f)
                emotion_df = pd.read_csv(f)
            
            # 读取AU数据
            au_file = os.path.join(au_dir, subject, "M1_2.csv")
            if not os.path.exists(au_file):
                continue
                
            au_df = pd.read_csv(au_file, encoding='utf-8-sig')
            au_df.columns = au_df.columns.str.strip()
            
            # 检查帧数是否匹配
            if not len(emotion_df) == len(au_df):
                print(f"\n警告: {subject} 的帧数不匹配!")
                continue
            
            # 获取干扰帧位置
            interference_frames = get_interference_frames(subject)
            
            # 创建干扰帧掩码（True表示非干扰帧）
            interference_mask = np.ones(len(au_df), dtype=bool)
            if interference_frames:  # 只有当存在干扰帧时才更新掩码
                for start_frame, end_frame in interference_frames:
                    start_frame = max(0, min(start_frame, len(au_df) - 1))
                    end_frame = max(0, min(end_frame, len(au_df) - 1))
                    interference_mask[start_frame:end_frame+1] = False
            
            # 获取AU强度列和存在列
            au_intensity_cols = [col for col in au_df.columns if '_r' in col and 'AU' in col]
            au_presence_cols = [col.replace('_r', '_c') for col in au_intensity_cols]
            
            # 获取目标情绪的帧
            emotion_mask = emotion_df['SmoothedEmotion'] == emotion
            
            # 有效掩码：目标情绪 + 成功检测 + 非干扰帧
            valid_mask = (emotion_mask) & (au_df['success'] == 1) & (interference_mask)
            
            if valid_mask.sum() == 0:
                continue
            
            # 计算每个AU的平均强度（只考虑存在的AUs）
            au_means = {}
            for intensity_col, presence_col in zip(au_intensity_cols, au_presence_cols):
                # 获取AU存在且情绪有效的帧
                au_valid_mask = valid_mask & (au_df[presence_col] == 1)
                if au_valid_mask.sum() > 0:
                    au_means[intensity_col] = au_df.loc[au_valid_mask, intensity_col].mean()
                else:
                    au_means[intensity_col] = np.nan  # 使用空值表示无效AU
            
            # 添加受试者信息
            au_means['姓名'] = subject
            au_means['group'] = group
            
            all_au_data.append(au_means)
            processed_subjects['TD' if group == 0 else 'ASD'] += 1
        
        except Exception as e:
            print(f"\n警告: 处理{subject}时出错 - {str(e)}")
            continue
    
    if not all_au_data:
        print(f"\n错误: 没有找到{emotion}情绪的有效数据")
        return None
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_au_data)
    
    # 获取所有AU列
    au_cols = [col for col in results_df.columns if '_r' in col]
    
    # 按组处理异常值
    for au_col in au_cols:
        for group_val in [0, 1]:  # 0: TD, 1: ASD
            group_mask = results_df['group'] == group_val
            if group_mask.sum() > 0:
                results_df.loc[group_mask, au_col] = replace_outliers(
                    results_df.loc[group_mask, au_col], k=1.5
                )
    
    # 保存结果
    output_file = output_path / f"au_intensities_{emotion}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n数据处理完成!")
    print(f"TD组: {processed_subjects['TD']}人")
    print(f"ASD组: {processed_subjects['ASD']}人")
    print(f"处理后的数据已保存至: {output_file}")
    
    return results_df

def extract_au_correlations_non_cartoon(emotion, output_dir="result/AU_Features_Non_Cartoon"):
    """
    提取指定情绪下排除看动画片任务后的AU相关性特征
    
    Parameters:
    -----------
    emotion : str
        目标情绪
    output_dir : str
        输出目录
    """
    # 读取数据
    au_dir = "result/AUs"
    emotion_dir = "result/Emotions_smooth2"
    demo_df = pd.read_csv("result/demographics.csv", encoding='utf-8-sig')
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = {}
    total_subjects = len(demo_df)
    processed_subjects = {'TD': 0, 'ASD': 0}
    
    print(f"\n开始提取{emotion}情绪下排除看动画片任务后的AU相关性数据...")
    
    # 遍历每个受试者
    for i, (_, row) in enumerate(demo_df.iterrows(), 1):
        subject = row['姓名']
        group = row['组别']
        
        print(f"\r处理进度：{i}/{total_subjects} ({i/total_subjects*100:.1f}%) - 当前处理：{subject}", 
              end='', flush=True)
        
        try:
            # 读取情绪数据
            emotion_file = os.path.join(emotion_dir, f"{subject}.csv")
            if not os.path.exists(emotion_file):
                continue
                
            with open(emotion_file, 'r', encoding='utf-8-sig') as f:
                next(f)
                next(f)
                emotion_df = pd.read_csv(f)
            
            # 读取AU数据
            au_file = os.path.join(au_dir, subject, "M1_2.csv")
            if not os.path.exists(au_file):
                continue
                
            au_df = pd.read_csv(au_file, encoding='utf-8-sig')
            au_df.columns = au_df.columns.str.strip()
            
            # 检查帧数是否匹配
            if not len(emotion_df) == len(au_df):
                continue
            
            # 获取干扰帧位置
            interference_frames = get_interference_frames(subject)
            
            # 创建干扰帧掩码（True表示非干扰帧）
            interference_mask = np.ones(len(au_df), dtype=bool)
            if interference_frames:  # 只有当存在干扰帧时才更新掩码
                for start_frame, end_frame in interference_frames:
                    start_frame = max(0, min(start_frame, len(au_df) - 1))
                    end_frame = max(0, min(end_frame, len(au_df) - 1))
                    interference_mask[start_frame:end_frame+1] = False
            
            # 获取所有AU强度列
            au_intensity_cols = [col for col in au_df.columns if '_r' in col and 'AU' in col]
            
            # 获取目标情绪的帧
            emotion_mask = emotion_df['SmoothedEmotion'] == emotion
            valid_mask = (emotion_mask) & (au_df['success'] == 1) & (interference_mask)
            
            if valid_mask.sum() == 0:
                continue
            
            # 识别连续的情绪片段
            segments = []
            start_idx = None
            for idx, val in enumerate(valid_mask):
                if val and start_idx is None:  # 片段开始
                    start_idx = idx
                elif not val and start_idx is not None:  # 片段结束
                    segments.append((start_idx, idx-1))
                    start_idx = None
            if start_idx is not None:  # 处理最后一个片段
                segments.append((start_idx, len(valid_mask)-1))
            
            # 存储每个片段的相关性系数
            segment_correlations = []
            
            # 计算每个片段的相关性矩阵
            for start, end in segments:
                segment_length = end - start + 1
                min_segment_length = 5 if emotion == 'anger' else 25
                if segment_length >= min_segment_length:
                    segment_data = au_df.loc[start:end, au_intensity_cols]
                    
                    # 检查是否有足够的非NaN值
                    if segment_data.notna().all().all():
                        try:
                            corr_matrix = segment_data.corr(method='spearman')
                            
                            # 提取上三角矩阵的相关性系数
                            correlations = {}
                            for i, au1 in enumerate(au_intensity_cols):
                                for j, au2 in enumerate(au_intensity_cols):
                                    if j > i:  # 只取上三角矩阵
                                        au1_name = au1.replace('_r', '')
                                        au2_name = au2.replace('_r', '')
                                        key = f"{au1_name}_{au2_name}_corr"
                                        correlations[key] = corr_matrix.loc[au1, au2]
                            
                            if correlations:
                                segment_correlations.append(correlations)
                        
                        except Exception as e:
                            continue
            
            if segment_correlations:
                # 计算所有片段的平均相关性系数
                mean_correlations = {}
                for key in segment_correlations[0].keys():
                    values = [seg[key] for seg in segment_correlations]
                    if values and not all(np.isnan(values)):
                        mean_correlations[key] = np.nanmedian(values)
                    else:
                        mean_correlations[key] = np.nan
                
                # 添加组信息
                mean_correlations['姓名'] = subject
                mean_correlations['group'] = group
                all_results[subject] = mean_correlations
                processed_subjects['TD' if group == 0 else 'ASD'] += 1
        
        except Exception as e:
            print(f"\n警告: 处理{subject}时出错 - {str(e)}")
            continue
    
    # 转换为DataFrame
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df = results_df.reset_index()  # 将index转换为列，确保'姓名'是普通列
    
    # 按组处理异常值
    results_clean = pd.DataFrame()
    for group_val in [0, 1]:  # 0: TD, 1: ASD
        group_data = results_df[results_df['group'] == group_val].copy()
        
        # 只处理相关性列，不包括组列和姓名列
        corr_cols = [col for col in group_data.columns if '_corr' in col]
        
        # 分别处理每个相关性列的异常值
        for col in corr_cols:
            group_data[col] = replace_outliers(group_data[col], k=1.5)
        
        results_clean = pd.concat([results_clean, group_data])
    
    # 保存结果
    output_file = output_path / f"au_intensity_correlations_{emotion}.csv"
    results_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n数据处理完成!")
    print(f"TD组: {processed_subjects['TD']}人")
    print(f"ASD组: {processed_subjects['ASD']}人")
    print(f"结果已保存至: {output_file}")
    print(f"\n成功处理的受试者数: {len(all_results)}")
    
    return results_clean

def merge_au_intensities_non_cartoon(feature_dir="result/AU_Features_Non_Cartoon",
                                     output_file="result/machine_learning/merged_au_intensities_non_cartoon.csv"):
    """
    合并排除看动画片任务后的所有情绪的AU强度特征
    参考 faceAU_analysis.ipynb 中的 merge_au_intensities 方法
    
    Parameters:
    -----------
    feature_dir : str
        特征文件目录
    output_file : str
        输出文件路径
    """
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    au_intensity_data = None
    
    print("=" * 60)
    print("合并AU强度特征（排除看动画片任务）")
    print("=" * 60)
    
    for emotion in emotions:
        file_path = os.path.join(feature_dir, f'au_intensities_{emotion}.csv')
        if os.path.exists(file_path):
            print(f"读取 {emotion} 情绪的特征...")
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 获取AU列（包含_r的列）
            au_cols = [col for col in df.columns if '_r' in col]
            
            # 为每个AU列添加情绪前缀
            rename_dict = {col: f'{emotion}_{col}' for col in au_cols}
            df = df.rename(columns=rename_dict)
            
            # 只保留重命名后的AU列和基本信息列
            keep_cols = list(rename_dict.values()) + ['姓名', 'group']
            df = df[keep_cols]
            
            if au_intensity_data is None:
                au_intensity_data = df
            else:
                au_intensity_data = au_intensity_data.merge(df, on=['姓名', 'group'], how='outer')
        else:
            print(f"警告: 未找到文件 {file_path}")
    
    if au_intensity_data is not None:
        # 重新排序列：姓名和group在前
        cols = ['姓名', 'group'] + [col for col in au_intensity_data.columns if col not in ['姓名', 'group']]
        au_intensity_data = au_intensity_data[cols]
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        au_intensity_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nAU强度数据合并完成，已保存至：{output_file}")
        print(f"数据维度：{au_intensity_data.shape}")
        
        return au_intensity_data
    else:
        print("错误: 未找到任何AU强度数据文件")
        return None

def merge_au_correlations_non_cartoon(feature_dir="result/AU_Features_Non_Cartoon",
                                      output_file="result/machine_learning/merged_au_correlations_non_cartoon.csv"):
    """
    合并排除看动画片任务后的所有情绪的AU相关性特征
    参考 faceAU_analysis.ipynb 中的 merge_au_correlations 方法
    
    Parameters:
    -----------
    feature_dir : str
        特征文件目录
    output_file : str
        输出文件路径
    """
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    au_correlation_data = None
    
    print("=" * 60)
    print("合并AU相关性特征（排除看动画片任务）")
    print("=" * 60)
    
    for emotion in emotions:
        file_path = os.path.join(feature_dir, f'au_intensity_correlations_{emotion}.csv')
        if os.path.exists(file_path):
            print(f"读取 {emotion} 情绪的特征...")
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 获取相关性列（包含_corr的列）
            corr_cols = [col for col in df.columns if '_corr' in col]
            
            # 为每个相关性列添加情绪前缀
            rename_dict = {col: f'{emotion}_{col}' for col in corr_cols}
            df = df.rename(columns=rename_dict)
            
            # 只保留重命名后的相关性列和基本信息列
            keep_cols = list(rename_dict.values()) + ['姓名', 'group']
            df = df[keep_cols]
            
            if au_correlation_data is None:
                au_correlation_data = df
            else:
                au_correlation_data = au_correlation_data.merge(df, on=['姓名', 'group'], how='outer')
        else:
            print(f"警告: 未找到文件 {file_path}")
    
    if au_correlation_data is not None:
        # 重新排序列：姓名和group在前
        cols = ['姓名', 'group'] + [col for col in au_correlation_data.columns if col not in ['姓名', 'group']]
        au_correlation_data = au_correlation_data[cols]
        
        # 按姓名首字母排序（可选）
        # au_correlation_data = au_correlation_data.sort_values(by='姓名', key=lambda x: x.str[0])
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        au_correlation_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nAU相关性数据合并完成，已保存至：{output_file}")
        print(f"数据维度：{au_correlation_data.shape}")
        
        return au_correlation_data
    else:
        print("错误: 未找到任何AU相关性数据文件")
        return None

def main():
    """主函数"""
    # 定义情绪列表
    emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    
    # 定义输出目录
    output_dir = "result/AU_Features_Non_Cartoon"
    
    print("=" * 60)
    print("AU特征提取 - 排除看动画片任务")
    print("排除干扰：口罩、自选动画、天鹅动画")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 提取AU强度特征
    print("\n" + "=" * 60)
    print("提取AU强度特征")
    print("=" * 60)
    for emotion in emotions:
        extract_au_features_non_cartoon(emotion, output_dir=output_dir)
    
    # 提取AU相关性特征
    print("\n" + "=" * 60)
    print("提取AU相关性特征")
    print("=" * 60)
    for emotion in emotions:
        extract_au_correlations_non_cartoon(emotion, output_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("所有特征提取完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 先提取所有特征
    print("=" * 80)
    print("第一步：提取AU特征（排除看动画片任务）")
    print("=" * 80)
    main()
    
    # 然后合并所有特征用于模型分类
    print("\n" + "=" * 80)
    print("第二步：合并AU特征用于模型分类（排除看动画片任务）")
    print("=" * 80)
    
    # 合并AU强度特征
    merged_intensities = merge_au_intensities_non_cartoon(
        feature_dir="result/AU_Features_Non_Cartoon",
        output_file="result/machine_learning/merged_au_intensities_non_cartoon.csv"
    )
    
    # 合并AU相关性特征
    merged_correlations = merge_au_correlations_non_cartoon(
        feature_dir="result/AU_Features_Non_Cartoon",
        output_file="result/machine_learning/merged_au_correlations_non_cartoon.csv"
    )
    
    print("\n" + "=" * 80)
    print("所有特征提取和合并完成！")
    print("=" * 80)

