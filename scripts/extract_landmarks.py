import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm

def extract_landmarks(video_path, output_path):
    mp_holistic = mp.solutions.holistic
    
    # Initialize Holistic model
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        frames_landmarks = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find landmarks.
            results = holistic.process(image)
            
            # Extract landmarks
            frame_data = {}
            
            # Pose
            if results.pose_landmarks:
                frame_data['pose'] = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            else:
                frame_data['pose'] = np.zeros((33, 4)) # 33 landmarks for pose

            # Left Hand
            if results.left_hand_landmarks:
                frame_data['left_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
            else:
                frame_data['left_hand'] = np.zeros((21, 3)) # 21 landmarks for hand

            # Right Hand
            if results.right_hand_landmarks:
                frame_data['right_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            else:
                frame_data['right_hand'] = np.zeros((21, 3))

            # Face (optional, can be heavy, usually 468 landmarks)
            # We will extract it but maybe filter later
            if results.face_landmarks:
                frame_data['face'] = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
            else:
                frame_data['face'] = np.zeros((468, 3))
                
            frames_landmarks.append(frame_data)
            
        cap.release()
        
        # Save as .npy
        # Structure: List of dicts is not efficient for npy. 
        # Better: Dict of arrays (T, N, C)
        
        T = len(frames_landmarks)
        if T == 0:
            return

        output_data = {
            'pose': np.array([f['pose'] for f in frames_landmarks]),
            'left_hand': np.array([f['left_hand'] for f in frames_landmarks]),
            'right_hand': np.array([f['right_hand'] for f in frames_landmarks]),
            'face': np.array([f['face'] for f in frames_landmarks])
        }
        
        np.save(output_path, output_data)

def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks from videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for landmarks')
    parser.add_argument('--subset', type=str, default='*', help='Subset to process (e.g., train, test, dev)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of videos to process')
    
    args = parser.parse_args()
    
    # Recursive search for video files
    search_pattern = os.path.join(args.input_dir, args.subset, '**', '*.mp4')
    # Phoenix videos might be .mp4 or image sequences. The user said "videos".
    # Let's check the file extension in the archive. 
    # Based on previous `ls`, it was `train/11August...`. It might be folders of images or video files.
    # I will assume files for now, but I should verify if they are folders of images.
    # The previous `ls` showed `train/11August_2010_Wednesday_tagesschau-1` as a name in the annotation.
    # Let's assume they are files. If not, I'll adjust.
    
    # Actually, let's look for any file type or specific extension if known.
    # I'll use glob to find files.
    
    video_files = glob.glob(search_pattern, recursive=True)
    if not video_files:
        # Try searching for files without extension or other common video formats
        video_files = glob.glob(os.path.join(args.input_dir, args.subset, '**', '*'), recursive=True)
        # Filter for likely video files or directories if it's image sequences
        video_files = [f for f in video_files if os.path.isfile(f) and not f.endswith('.DS_Store')]
        
    print(f"Found {len(video_files)} videos to process.")
    
    if args.limit:
        video_files = video_files[:args.limit]
        print(f"Limiting to {args.limit} videos.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for video_path in tqdm(video_files):
        # Create corresponding output path
        rel_path = os.path.relpath(video_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        output_path = os.path.splitext(output_path)[0] + '.npy'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            continue
            
        extract_landmarks(video_path, output_path)

if __name__ == "__main__":
    main()
