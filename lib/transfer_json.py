import os
import shutil
import json

def transfer_json_files():
    # Base path
    base_path = r'K:'
    
    # Ensure target directory exists
    target_dir = r'F:\Code\Face_extra\result\Events'
    os.makedirs(target_dir, exist_ok=True)
    
    # Traverse all folders under K drive
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        
        # Check if it is a folder
        if os.path.isdir(subject_path):
            # Build complete path for M1.json
            json_path = os.path.join(subject_path, 'M1.json')
            
            # Check if M1.json exists
            if os.path.exists(json_path):
                try:
                    # Read json file
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Count the number of True
                        true_count = sum(1 for x in data['interference_mask'] if x)
                        print(f'{subject}: Number of True in interference_mask is {true_count}')
                    
                    # Build new filename (subject.json)
                    new_filename = f'{subject}.json'
                    target_path = os.path.join(target_dir, new_filename)
                    
                    # Copy file to new location
                    shutil.copy2(json_path, target_path)
                    print(f'Successfully copied file: {subject} -> {new_filename}')
                except Exception as e:
                    print(f'Failed to process file {subject}: {str(e)}')
            else:
                print(f'{subject} does not have M1.json file')

if __name__ == '__main__':
    transfer_json_files()
