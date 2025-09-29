import os
import shutil
import json

def transfer_json_files():
    # 基础路径
    base_path = r'K:'
    
    # 确保目标目录存在
    target_dir = r'F:\Code\Face_extra\result\Events'
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历K盘下的所有文件夹
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        
        # 检查是否是文件夹
        if os.path.isdir(subject_path):
            # 构建M1.json的完整路径
            json_path = os.path.join(subject_path, 'M1.json')
            
            # 检查M1.json是否存在
            if os.path.exists(json_path):
                try:
                    # 读取json文件
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # 统计True的数量
                        true_count = sum(1 for x in data['interference_mask'] if x)
                        print(f'{subject}: interference_mask中True的数量为 {true_count}')
                    
                    # 构建新的文件名（subject.json）
                    new_filename = f'{subject}.json'
                    target_path = os.path.join(target_dir, new_filename)
                    
                    # 复制文件到新位置
                    shutil.copy2(json_path, target_path)
                    print(f'成功复制文件: {subject} -> {new_filename}')
                except Exception as e:
                    print(f'处理文件失败 {subject}: {str(e)}')
            else:
                print(f'{subject} 没有M1.json文件')

if __name__ == '__main__':
    transfer_json_files()
