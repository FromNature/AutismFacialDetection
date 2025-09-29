"""
操作EventTracking标签文件 V3.0版
更新于2024-11-20
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import Union, List, Tuple
from collections import defaultdict

class Event:
    TIME_FORMAT = "%H:%M:%S.%f"  # 将格式字符串作为类变量
    BASE_DATE = datetime(1900, 1, 1)  # 基准日期作为类变量
    
    def __init__(self, index, label, start_time, end_time, back_ground='#000000'):
        """初始化事件对象
        
        Args:
            index: 事件索引
            label (str): 事件标签名称
            start_time (Union[str, datetime]): 开始时间，可以是时间字符串或datetime对象
            end_time (Union[str, datetime]): 结束时间，可以是时间字符串或datetime对象
            back_ground (str, optional): 背景颜色. 默认为 '#000000'
        """
        self.Index = index
        self.Label = label
        self.back_ground = back_ground
        # 优化时间处理逻辑
        self.StartTime = (start_time if isinstance(start_time, datetime) 
                         else datetime.strptime(self._format_time(start_time), self.TIME_FORMAT))
        self.EndTime = (end_time if isinstance(end_time, datetime) 
                       else datetime.strptime(self._format_time(end_time), self.TIME_FORMAT))
        
    @staticmethod
    def _format_time(time_str):
        """格式化时间字符串"""
        if '.' not in time_str:
            return f"{time_str}.000000"
        return time_str[:-1] if time_str.count('.') == 1 else time_str
    
    def duration(self) -> float:
        """计算事件持续时间
        
        Returns:
            float: 事件持续时间（秒）
        """
        return (self.EndTime - self.StartTime).total_seconds()
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'BackGroundBrush': self.back_ground,
            'Label': self.Label,
            'StartTime': self.StartTime.strftime(self.TIME_FORMAT),
            'EndTime': self.EndTime.strftime(self.TIME_FORMAT),
            'Index': self.Index
        }

class Events:

    def __init__(self, file: str = None):
        self.events: List[Event] = []
        if file:
            self.load_json(file)

    def load_json(self, file: str):
        """从JSON文件加载事件数据
        
        Args:
            file (str): JSON文件路径
            
        Raises:
            Exception: 文件读取或解析错误时抛出异常
        """
        try:
            with open(file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
            self.events = []
            for record in json_data:
                try:
                    if self._is_valid_record(record):
                        self.events.append(Event(
                            record['Index'],
                            record['Label'],
                            record['StartTime'],
                            record['EndTime'],
                            record.get('BackGroundBrush', '#000000')
                        ))
                except (KeyError, ValueError) as e:
                    print(f"跳过无效记录: {e}")
                    
            if self.has_overlap('儿童发音'):
                print("警告: 标签'儿童发音'存在时间重叠")
                    
        except Exception as e:
            print(f"加载JSON文件出错: {e}")
            raise
    
    @staticmethod
    def _is_valid_record(record: dict) -> bool:
        """验证记录是否有效"""
        required_fields = ['Index', 'Label', 'StartTime', 'EndTime']
        return all(
            record.get(field) and record[field] != "00:00:00"
            for field in required_fields
        )
    
    def save_to_json(self, file: str):
        """将事件数据保存为JSON文件
        
        Args:
            file (str): 保存的文件路径
            
        Raises:
            Exception: 文件写入错误时抛出异常
        """
        try:
            json_data = [event.to_dict() for event in self.events]
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存JSON文件出错: {e}")
            raise

    # 判断某个标签是否有时间的重叠
    def has_overlap(self, label: str) -> bool:
        """检查指定标签的事件是否存在时间重叠
        
        Args:
            label (str): 要检查的事件标签
            
        Returns:
            bool: True表示存在重叠，False表示不存在重叠
        """
        time_sequence = []
        for event in self.events:
            if event.Label == label:
                time_sequence.extend([(event.StartTime - datetime(1900, 1, 1)).total_seconds(), (event.EndTime - datetime(1900, 1, 1)).total_seconds()])
            
        flag = False
        for i in range(1, len(time_sequence)):
            if time_sequence[i] < time_sequence[i - 1]:
                minutes, seconds = divmod(time_sequence[i-1], 60)
                formatted_time = f"{int(minutes)}:{int(seconds)}"
                print(formatted_time)
                flag = True
        return flag
       

    # 根据标签筛选事件
    def filter_by_label(self, range_labels=None, exclude_range_labels=None, 
                       stay_labels=None, exclude_labels=None, 
                       duration_min=None, duration_max=None) -> 'Events':
        """根据多个条件筛选事件
        
        Args:
            range_labels (List[str], optional): 时间范围内的标签列表
            exclude_range_labels (List[str], optional): 要排除的时间范围标签列表
            stay_labels (List[str], optional): 要保留的标签列表
            exclude_labels (List[str], optional): 要排除的标签列表
            duration_min (float, optional): 最小持续时间（秒）
            duration_max (float, optional): 最大持续时间（秒）
            
        Returns:
            Events: 新的Events对象，包含筛选后的事件
        """
        if not self.events:
            return Events()
            
        filtered = Events()
        events_to_process = self._filter_by_range(range_labels)
        
        # 应用范围排除
        if exclude_range_labels:
            events_to_process = self._exclude_range(events_to_process, exclude_range_labels)
            
        # 应用标签过滤
        if stay_labels or exclude_labels:
            events_to_process = self._apply_label_filters(events_to_process, 
                                                        stay_labels, exclude_labels)
            
        # 应用持续时间过滤
        filtered.events = self._filter_by_duration(events_to_process, 
                                                 duration_min, duration_max)
        
        return filtered
    
    def _filter_by_range(self, range_labels) -> List[Event]:
        """按范围过滤事件"""
        if not range_labels:
            return self.events.copy()
            
        range_events = self.filter_by_label(stay_labels=range_labels).events
        filtered = []
        for event in self.events:
            if any(not (event.EndTime <= r_event.StartTime or 
                       event.StartTime >= r_event.EndTime)
                   for r_event in range_events):
                filtered.append(event)
        return filtered
    
    # 返回事件的时间列表[[start_time, end_time]]
    def get_time_list(self) -> List[List[float]]:
        """获取所有事件的时间列表
        
        Returns:
            List[List[float]]: 二维列表，每个子列表包含[开始时间, 结束时间]（秒）
        """
        if not self.events:
            print("Events is empty.")
            return []
        
        time_list_seconds = []
        for event in self.events:
            start_time_seconds = (event.StartTime - datetime(1900, 1, 1)).total_seconds()
            end_time_seconds = (event.EndTime - datetime(1900, 1, 1)).total_seconds()
            time_list_seconds.append([start_time_seconds, end_time_seconds])

        return time_list_seconds
    
    # 返回事件的帧位置列表[[开始帧, 结束帧]]
    def get_frame_list(self, sr: float) -> List[List[int]]:
        """获取事件的帧位置列表
        
        Args:
            sr (float): 采样率（每秒的帧数）
            
        Returns:
            List[List[int]]: 二维列表，每个子列表包含[开始帧, 结束帧]
        """
        if not self.events:
            print("Events is empty.")
            return []

        def convert(sec, sr):
            return int(sec * sr)
        return [[convert(start_time, sr), convert(ent_time, sr)] for [start_time, ent_time] in self.get_time_list()]

    def has_label(self, label: str) -> bool:
        """检查是否存在指定标签的事件
        
        Args:
            label (str): 要检查的标签名称
            
        Returns:
            bool: True表示存在该标签，False表示不存在
        """
        return any(event.Label == label for event in self.events)

    # 打印所有事件
    def print_events(self):
        """打印所有事件的详细信息
        
        输出格式：Index: {索引}, Label: {标签}, StartTime: {开始时间}, EndTime: {结束时间}
        """
        if not self.events:
            print("Events is empty.")
            return
        
        for event in self.events:
            print(f"Index: {event.Index}, Label: {event.Label}, StartTime: {event.StartTime.strftime('%H:%M:%S.%f')[:-3]}, EndTime: {event.EndTime.strftime('%H:%M:%S.%f')[:-3]}")


    # 输出事件统计信息
    def events_statistics(self, is_print=True) -> pd.DataFrame:
        """生成事件统计信息
        
        Args:
            is_print (bool, optional): 是否打印统计结果. 默认为True
            
        Returns:
            pd.DataFrame: 包含统计信息的DataFrame，列包括：
                - Label: 标签名称
                - Count: 事件数量
                - Total-duration: 总持续时间
                - Mean-duration: 平均持续时间
                - Max-duration: 最大持续时间
                - Min-duration: 最小持续时间
        """
        if not self.events:
            print("Events为空")
            return pd.DataFrame()

        # 使用defaultdict简化数据收集
        stats = defaultdict(list)
        
        # 收集数据
        for event in self.events:
            stats[event.Label].append(event.duration())
            
        # 生成统计结果
        result_data = [
            {
                'Label': label,
                'Count': len(durations),
                'Total-duration': sum(durations),
                'Mean-duration': sum(durations) / len(durations),
                'Max-duration': max(durations),
                'Min-duration': min(durations)
            }
            for label, durations in stats.items()
        ]
        
        # 创建DataFrame
        result_df = pd.DataFrame(result_data)
        
        if is_print:
            for row in result_data:
                print(f"Label: {row['Label']}, "
                      f"Count: {row['Count']}, "
                      f"Total-duration: {row['Total-duration']:.3f} s, "
                      f"Mean-duration: {row['Mean-duration']:.3f}, "
                      f"Max-duration: {row['Max-duration']:.3f}, "
                      f"Min-duration: {row['Min-duration']:.3f}")
                
        return result_df

def extract_frames_by_positions(original_frames: Union[list, pd.DataFrame, np.ndarray], 
                                position_list: List[Tuple[int, int]], 
                                zero_length: int = 0):
    """根据位置列表提取帧数据
    
    Args:
        original_frames: 原始帧数据，支持list、DataFrame或ndarray格式
        position_list: 位置列表，每个元素为(start_frame, end_frame)元组
        zero_length: 在提取的片段之间插入的0的数量
        
    Returns:
        Union[list, pd.DataFrame]: 提取的帧数据，保持输入的数据类型
        
    Raises:
        ValueError: 当位置索引超出范围时
        TypeError: 当输入数据类型不支持时
    """
    if isinstance(original_frames, (list, np.ndarray)):
        extracted_frames = []
        for i, (start_frame, end_frame) in enumerate(position_list):
            # 检查索引是否合法
            if start_frame < 0 or end_frame >= len(original_frames):
                raise ValueError(f"索引 {start_frame}-{end_frame} 超出范围")
            # 提取帧
            extracted_frames.extend(original_frames[start_frame:end_frame + 1])
            # 如果不是最后一段，则插入 zero_length 个 0
            if i < len(position_list) - 1 and zero_length > 0:
                extracted_frames.extend([0] * int(zero_length))
        return extracted_frames
    
    elif isinstance(original_frames, pd.DataFrame):
        extracted_frames_list = []
        for i, (start_frame, end_frame) in enumerate(position_list):
            # 检查索引是否合法
            if start_frame < 0 or end_frame >= len(original_frames):
                raise ValueError(f"索引 {start_frame}-{end_frame} 超出范围")
            # 提取帧
            extracted_frames_list.append(original_frames.iloc[start_frame:end_frame + 1])
            # 如果不是最后一段，则插入 zero_length 行全 0 数据
            if i < len(position_list) - 1 and zero_length > 0:
                zero_df = pd.DataFrame(0, index=range(int(zero_length)), columns=original_frames.columns)
                extracted_frames_list.append(zero_df)
        # 使用 pd.concat 拼接
        extracted_frames = pd.concat(extracted_frames_list).reset_index(drop=True)
        return extracted_frames
    
    else:
        raise TypeError("original_frames 的类型必须是 list, np.ndarray 或 pd.DataFrame")


def merge_json_files(base_json_path: str, add_json_path: str, base_video_path: str, output_json_path: str):
    """合并两个JSON标注文件
    
    Args:
        base_json_path (str): 基准JSON文件路径
        add_json_path (str): 要添加的JSON文件路径
        base_video_path (str): 基准视频文件路径，用于获取视频时长
        output_json_path (str): 输出的合并后的JSON文件路径
        
    Raises:
        Exception: 文件操作或合并过程中的错误
    """
    base_events = Events(base_json_path)
    add_events = Events(add_json_path)

    def get_duration_from_moviepy(url: str) -> timedelta:
        clip = VideoFileClip(url)
        return timedelta(seconds=clip.duration)
    
    try:
        duration_timedelta = get_duration_from_moviepy(base_video_path)
        
        for event in add_events.events:
            event.StartTime += duration_timedelta
            event.EndTime += duration_timedelta
        
        base_events.events.extend(add_events.events)
        base_events.save_to_json(output_json_path)
        print('Merge json file successfully.')
    except Exception as e:
        print(f"Error merging JSON files: {e}")