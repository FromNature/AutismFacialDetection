'''
Extract facial action units using OpenFace Installation guide: F:\HelpDoc\OpenFace安装材料
Environment: ASD_Face_env
Date: 2024-11-20
'''

import subprocess
from concurrent.futures import ThreadPoolExecutor
import os

class FaceAUProcessor:
    """
    Class for extracting facial action units using OpenFace.
    
    Attributes:
        openface_exe_path (str): Path to the OpenFace executable file.
        output_base_dir (str): Base directory for output results.
    """

    def __init__(self, openface_exe_path, output_base_dir='result\\AUs'):
        """
        Initialize an instance of the FaceAUProcessor class.

        Args:
            openface_exe_path (str): Path to the OpenFace executable file.
            output_base_dir (str): Base directory for output results.
        """
        self.openface_exe_path = openface_exe_path
        self.output_base_dir = output_base_dir

    def detect_au(self, video_path, index, total):
        """
        Detect facial action units in video using OpenFace.

        Args:
            video_path (str): Path to the video file.
            index (int): Index of the current video being processed.
            total (int): Total number of video files.
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
        Process multiple video files to extract facial action units.

        Args:
            video_files (list): List of video file paths.
        """
        max_workers = 24  # Or adjust as needed
        total_files = len(video_files)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(lambda p: self.detect_au(p[1], p[0], total_files), enumerate(video_files))

def find_video_files(root_dir, video_name):
    """
    Find all paths containing specific video files under the specified directory.

    Args:
        root_dir (str): Root directory to search.
        video_name (str): Name of the video file to find.

    Returns:
        list: List containing all complete paths of matching video files.
    """
    video_paths = []
    for root, dirs, files in os.walk(root_dir):
        if video_name in files:
            video_paths.append(os.path.join(root, video_name))
    return video_paths

# Usage example
if __name__ == "__main__":
    openface_exe_path = 'F:\\Code\\OpenFace-OpenFace_2.2.0\\x64\\Release\\FeatureExtraction.exe'
    
    # Single person processing
    processor = FaceAUProcessor(openface_exe_path)
    processor.process_videos([r'K:\ZZY\M1_2.mp4']) 


    # Batch processing
    # # Find all video file paths containing M1_2.mp4
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