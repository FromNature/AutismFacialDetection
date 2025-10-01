"""
openvino_env2
openvino==2023.1.0 (This version is required)
"""

import cv2 as cv
import numpy as np
from openvino.inference_engine import IECore
from datetime import datetime
import os

# Face detection model path
face_model_weights = "./intel/opencv_face_detector_uint8.pb"
face_model_config = "./intel/opencv_face_detector.pbtxt"

# Emotion recognition model path
emotion_model_xml = "./intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml"
emotion_model_bin = "./intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin"

# Emotion labels
labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# Initialize OpenVINO
ie = IECore()
emotion_net = ie.read_network(model=emotion_model_xml, weights=emotion_model_bin)
input_blob = next(iter(emotion_net.input_info))
exec_net = ie.load_network(network=emotion_net, device_name="GPU")

def detect_emotion(frame, frame_count, timestamp, output_file):
    emotion_label = 'undefined'
    probability = 0.0
    result_text = f"{frame_count},{timestamp:.2f},"
    
    # Convert to grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Find the largest face
        max_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = max_face
        
        # Check if face box almost occupies the entire frame (indicating possible false detection)
        frame_height, frame_width = frame.shape[:2]
        face_area = w * h
        frame_area = frame_width * frame_height
        face_ratio = face_area / frame_area
        
        if face_ratio > 0.9:  # If face box exceeds 90% of the frame, consider it false detection
            cv.putText(frame, "No Face Detected", (30, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            result_text += "undefined,0.00\n"
        else:
            # Extract face region and perform inference
            face = frame[y:y+h, x:x+w]
            face_blob = cv.resize(face, (64, 64))
            face_blob = face_blob.transpose((2, 0, 1))
            face_blob = face_blob.reshape(1, 3, 64, 64)

            res = exec_net.infer(inputs={input_blob: face_blob})
            emotion_scores = res[next(iter(res))]
            emotion_index = np.argmax(emotion_scores)
            emotion_label = labels[emotion_index]
            probability = float(emotion_scores[0][emotion_index] * 100)

            # Display results
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(frame, f"{emotion_label} ({probability:.2f}%)", 
                      (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            result_text += f"{emotion_label},{probability:.2f}\n"
    else:
        result_text += "undefined,0.00\n"
    
    # Write results in real-time
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(result_text)
        
    return frame

def process_video(video_path, output_file, display_output):
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    
    # Record start time
    start_time = datetime.now()
    
    # Write file header
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Emotion Analysis Results - {datetime.now()}\n")
        f.write(f"Video: {video_path}\n\n")
        f.write("Frame,Timestamp,Emotion,Probability\n")
    
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
            
        # Process current frame
        timestamp = frame_count / fps
        frame = detect_emotion(frame, frame_count, timestamp, output_file)
        
        # Display results
        if display_output:
            cv.imshow("Emotion Detection", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate progress and estimated remaining time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        progress = (frame_count + 1) / total_frames * 100
        if frame_count > 0:  # Avoid division by zero
            frames_per_second = (frame_count + 1) / elapsed_time
            remaining_frames = total_frames - (frame_count + 1)
            estimated_time = remaining_frames / frames_per_second
            # Convert seconds to minutes and seconds
            minutes = int(estimated_time // 60)
            seconds = int(estimated_time % 60)
            print(f"\r处理进度：{frame_count + 1}/{total_frames} ({progress:.1f}%) - 预计剩余时间：{minutes}分{seconds}秒", end='')
            
        frame_count += 1
    
    print()  # Print newline
    capture.release()
    cv.destroyAllWindows()
    print(f"结果已保存至: {output_file}")

def batch_process_videos():
    root_dir = r"C:\Users\dumin\Desktop\问题3"
    output_dir = r"result\Emotions"
    target_filename = "M1_2.mp4"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all person folders
    person_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    total_persons = len(person_folders)
    processed_count = 0
    
    print(f"找到{total_persons}个人的文件夹")
    
    for person in person_folders:
        video_path = os.path.join(root_dir, person, target_filename)
        if os.path.exists(video_path):
            print(f"\n正在处理 {person} 的视频...")
            output_file = os.path.join(output_dir, f"{person}.csv")
            process_video(video_path, output_file, False)  # Don't display video during batch processing
            processed_count += 1
            print(f"\n批处理进度：{processed_count}/{total_persons} ({processed_count/total_persons*100:.1f}%)")
        else:
            print(f"\n{person} 的文件夹中没有找到 {target_filename} 视频文件")
    
    print(f"\n批处理完成，共处理了 {processed_count} 个视频文件")

def main():
    # Keep the original single video processing functionality
    single_video_mode = True  # Can control whether to use single video mode through parameters
    
    if single_video_mode:
        video_path = r"test\LZY\M1_2.mp4"
        output_file = r"test\Openvino\LZY_emotions.csv"
        display_output = False
        process_video(video_path, output_file, display_output)
    else:
        batch_process_videos()

if __name__ == "__main__":
    main()
