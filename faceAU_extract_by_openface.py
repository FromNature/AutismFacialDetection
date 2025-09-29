'''
使用OpenFace提取面部动作单元 安装指南:F:\HelpDoc\OpenFace安装材料
环境: ASD_Face_env
日期: 2024-11-20
'''

import subprocess
from concurrent.futures import ThreadPoolExecutor
import os

class FaceAUProcessor:
    """
    使用OpenFace提取面部动作单元的类。
    
    Attributes:
        openface_exe_path (str): OpenFace可执行文件的路径。
        output_base_dir (str): 输出结果的基础目录。
    """

    def __init__(self, openface_exe_path, output_base_dir='result\\AUs'):
        """
        初始化FaceAUProcessor类的实例。

        Args:
            openface_exe_path (str): OpenFace可执行文件的路径。
            output_base_dir (str): 输出结果的基础目录。
        """
        self.openface_exe_path = openface_exe_path
        self.output_base_dir = output_base_dir

    def detect_au(self, video_path, index, total):
        """
        使用OpenFace检测视频中的面部动作单元。

        Args:
            video_path (str): 视频文件的路径。
            index (int): 当前处理的视频索引。
            total (int): 总视频文件数。
        """
        if not os.path.exists(video_path):
            print(f"错误: 输入文件 '{video_path}' 不存在。")
            return

        unique_id = os.path.basename(os.path.dirname(video_path))
        output_dir = os.path.join(self.output_base_dir, unique_id)
        os.makedirs(output_dir, exist_ok=True)

        command = [
            self.openface_exe_path,
            '-f', video_path,
            '-aus',
            '-pose',
            '-gaze',
            '-out_dir', output_dir
        ]

        try:
            print(f"Processing {index + 1}/{total}: {video_path}")
            subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Detection successful for {video_path}.")
        except subprocess.CalledProcessError as e:
            print(f"Error during detection for {video_path}: {e.stderr}")
        except FileNotFoundError:
            print("Executable not found. Please check the path to FeatureExtraction.exe.")
        except Exception as e:
            print(f"An unexpected error occurred for {video_path}: {e}")

    def process_videos(self, video_files):
        """
        处理多个视频文件以提取面部动作单元。

        Args:
            video_files (list): 视频文件路径的列表。
        """
        max_workers = 24  # 或者根据需要调整
        total_files = len(video_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda p: self.detect_au(p[1], p[0], total_files), enumerate(video_files))

def find_video_files(root_dir, video_name):
    """
    在指定目录下查找所有包含特定视频文件的路径。

    Args:
        root_dir (str): 要搜索的根目录。
        video_name (str): 要查找的视频文件名。

    Returns:
        list: 包含所有匹配视频文件完整路径的列表。
    """
    video_paths = []
    for root, dirs, files in os.walk(root_dir):
        if video_name in files:
            video_paths.append(os.path.join(root, video_name))
    return video_paths

# 使用示例
if __name__ == "__main__":
    openface_exe_path = 'F:\\Code\\OpenFace-OpenFace_2.2.0\\x64\\Release\\FeatureExtraction.exe'
    
    # 单人处理
    processor = FaceAUProcessor(openface_exe_path)
    processor.process_videos([r'K:\ZZY\M1_2.mp4']) 


    # 批处理
    # # 查找所有包含M1_2.mp4的视频文件路径
    # root_directory = r'K:'  # K盘根目录
    # video_name = 'M1_2.mp4'
    # video_files = find_video_files(root_directory, video_name)
    
    # if not video_files:
    #     print(f"未找到任何{video_name}文件")
    # else:
    #     print(f"找到{len(video_files)}个{video_name}文件:")
    #     for path in video_files:
    #         print(path)
    #     processor = FaceAUProcessor(openface_exe_path)
    #     processor.process_videos(video_files) 