import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

correct_poses = {
    'Tree Pose': {
        'LEFT_HIP': [0.5, 0.5], 
        'RIGHT_HIP': [0.5, 0.5],
        'LEFT_KNEE': [0.45, 0.55],
        'RIGHT_KNEE': [0.55, 0.55],
        'LEFT_ANKLE': [0.4, 0.6],
        'RIGHT_ANKLE': [0.6, 0.6],
        
    },
    #these are not correct values first we run the code and capture the coordinates then after 
    'Warrior Pose': {
        'LEFT_HIP': [0.5, 0.5], 
        'RIGHT_HIP': [0.5, 0.5],
        'LEFT_KNEE': [0.5, 0.6],
        'RIGHT_KNEE': [0.5, 0.4],
        'LEFT_ANKLE': [0.5, 0.7],
        'RIGHT_ANKLE': [0.5, 0.3],
    },
}

pose_images = {
    'Tree Pose': 'tree_pose.png',
    'Warrior Pose': 'warrior_pose.png',
}

for pose, path in pose_images.items():
    if os.path.exists(path):
        pose_images[pose] = cv2.imread(path)
        pose_images[pose] = cv2.resize(pose_images[pose], (200, 200))
    else:
        print(f"Warning: Image for {pose} not found at {path}")
        pose_images[pose] = None

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def is_pose_correct(landmarks, correct_pose_landmarks, threshold=0.2):
    for landmark, correct_pos in correct_pose_landmarks.items():
        if landmark in landmarks:
            detected_pos = [landmarks[landmark].x, landmarks[landmark].y]
            distance = calculate_distance(detected_pos, correct_pos)
            print(f"{landmark}: Detected position {detected_pos}, Reference position {correct_pos}, Distance {distance}")
            if distance > threshold:
                return False
        else:
            return False
    return True

def get_pose_name(landmarks, correct_poses, threshold=0.2):
    for pose_name, correct_landmarks in correct_poses.items():
        if is_pose_correct(landmarks, correct_landmarks, threshold):
            return pose_name
    return "Incorrect Pose"

cap = cv2.VideoCapture(0)
cv2.namedWindow('Pose Suggestions', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = {mp_pose.PoseLandmark(i).name: results.pose_landmarks.landmark[i] for i in range(len(results.pose_landmarks.landmark))}
            print("Detected Landmarks:", landmarks.keys())  # Debug print statement

            pose_name = get_pose_name(landmarks, correct_poses)
            overlay_color = (0, 255, 0) if pose_name != "Incorrect Pose" else (0, 0, 255)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=overlay_color, thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=overlay_color, thickness=2, circle_radius=2))
            
            cv2.putText(frame, pose_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, overlay_color, 2, cv2.LINE_AA)
        
        cv2.imshow('Yoga Pose Detection', frame)

        suggestion_window = np.zeros((600, 300, 3), dtype=np.uint8)
        y_offset = 50

        for pose_name, pose_image in pose_images.items():
            if pose_image is not None:
                suggestion_window[y_offset:y_offset+200, 50:250] = pose_image
                cv2.putText(suggestion_window, pose_name, (50, y_offset+220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += 250

        cv2.imshow('Pose Suggestions', suggestion_window)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
