import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import numpy as np
import math as m

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    return int(theta * (180 / m.pi))

def main():
    st.title("Human Posture Detection using MediaPipe")
    st.sidebar.header("Options")

    mode = st.sidebar.selectbox("Choose Input Mode", ["Webcam", "Video File"])

    if mode == "Webcam":
        st.sidebar.write("Press 'Start' to begin posture detection from the webcam.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if st.sidebar.button("Start"):
        if mode == "Webcam":
            st.write("Starting webcam...")
            run_posture_detection_webcam(pose)
        elif uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            st.write("Processing uploaded video...")
            run_posture_detection_video(temp_file.name, pose)

def process_frame(frame, pose, good_frames, bad_frames):
    # Color definitions
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        lmPose = mp.solutions.pose.PoseLandmark

        # Extract key points
        l_shldr_x = int(lm[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)

        # Calculate offset for camera alignment
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        if offset < 100:
            cv2.putText(frame, f'{int(offset)} Aligned', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
        else:
            cv2.putText(frame, f'{int(offset)} Not Aligned', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)

        # Calculate angles
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Determine posture
        if neck_inclination < 30 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            color = light_green
        else:
            good_frames = 0
            bad_frames += 1
            color = red

        # Draw all landmarks
        cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(frame, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, pink, -1)

        # Draw vertical reference points
        vert_shoulder_point = (l_shldr_x, l_shldr_y - 100)
        vert_hip_point = (l_hip_x, l_hip_y - 100)
        cv2.circle(frame, vert_shoulder_point, 7, yellow, -1)
        cv2.circle(frame, vert_hip_point, 7, yellow, -1)

        # Draw all lines
        cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
        cv2.line(frame, (l_shldr_x, l_shldr_y), vert_shoulder_point, color, 4)
        cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)
        cv2.line(frame, (l_hip_x, l_hip_y), vert_hip_point, color, 4)

        # Add text annotations
        cv2.putText(frame, f'Neck: {int(neck_inclination)}  Torso: {int(torso_inclination)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Calculate and display time
        good_time = good_frames / 30  # Assuming 30 FPS
        bad_time = bad_frames / 30
        if good_time > 0:
            time_text = f'Good Posture Time: {round(good_time, 1)}s'
            cv2.putText(frame, time_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
        else:
            time_text = f'Bad Posture Time: {round(bad_time, 1)}s'
            cv2.putText(frame, time_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, red, 2)

    return frame, good_frames, bad_frames

def run_posture_detection_webcam(pose):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access webcam.")
        return

    good_frames = 0
    bad_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture frame from webcam.")
            break

        frame, good_frames, bad_frames = process_frame(frame, pose, good_frames, bad_frames)
        stframe.image(frame, channels="BGR")
     
    cap.release()

def run_posture_detection_video(video_path, pose):
    stframe = st.empty()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Unable to open video file.")
        return

    good_frames = 0
    bad_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame, good_frames, bad_frames = process_frame(frame, pose, good_frames, bad_frames)
        stframe.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()