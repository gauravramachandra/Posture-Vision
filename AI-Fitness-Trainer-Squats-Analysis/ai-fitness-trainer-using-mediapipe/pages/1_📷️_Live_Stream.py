import av
import os
import sys
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple

# Utilities
def get_mediapipe_pose():
    """Initialize MediaPipe Pose"""
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

@dataclass
class PoseThresholds:
    """Thresholds for pose analysis"""
    knee_angle_min: float
    knee_angle_max: float
    hip_angle_min: float
    hip_angle_max: float
    ankle_angle_min: float
    ankle_angle_max: float

def get_thresholds_beginner() -> PoseThresholds:
    return PoseThresholds(
        knee_angle_min=80,
        knee_angle_max=170,
        hip_angle_min=70,
        hip_angle_max=160,
        ankle_angle_min=70,
        ankle_angle_max=110
    )

def get_thresholds_pro() -> PoseThresholds:
    return PoseThresholds(
        knee_angle_min=70,
        knee_angle_max=170,
        hip_angle_min=60,
        hip_angle_max=160,
        ankle_angle_min=65,
        ankle_angle_max=110
    )

class ProcessFrame:
    def __init__(self, thresholds: PoseThresholds, flip_frame: bool = True):
        self.thresholds = thresholds
        self.flip_frame = flip_frame
        self.counter = 0
        self.stage = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def process(self, frame: np.ndarray, pose) -> Tuple[np.ndarray, dict]:
        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process pose
        results = pose.process(image)

        # Convert back to BGR and enable writing
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angles
                knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                hip_angle = self.calculate_angle([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                                               left_hip, left_knee)
                ankle_angle = self.calculate_angle(left_knee, left_ankle,
                                                 [landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                                  landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])

                # Squat counter logic
                if knee_angle > self.thresholds.knee_angle_max:
                    self.stage = "up"
                elif (self.thresholds.knee_angle_min <= knee_angle <= self.thresholds.knee_angle_max and
                      self.thresholds.hip_angle_min <= hip_angle <= self.thresholds.hip_angle_max and
                      self.thresholds.ankle_angle_min <= ankle_angle <= self.thresholds.ankle_angle_max):
                    if self.stage == "up":
                        self.counter += 1
                        self.stage = "down"

                # Visualization
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Status box
                cv2.rectangle(image, (0, 0), (250, 60), (0, 0, 0), -1)
                cv2.putText(image, f'Count: {self.counter}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Stage: {self.stage}', (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

        return image, {"counter": self.counter, "stage": self.stage}

def main():
    st.title('AI Fitness Trainer: Squats Analysis')

    # Mode selection
    mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)
    thresholds = get_thresholds_beginner() if mode == 'Beginner' else get_thresholds_pro()

    # Initialize processing
    live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
    pose = get_mediapipe_pose()

    # WebRTC configuration
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Callback for video frames
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_frame, _ = live_process_frame.process(img, pose)
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="squat-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Instructions
    if webrtc_ctx.state.playing:
        st.markdown("""
        ### Instructions:
        1. Stand in front of the camera
        2. Perform squats with proper form
        3. The counter will track your repetitions
        4. Watch the stage indicator for up/down positions
        """)
    
    # Display thresholds
    with st.expander("Current Thresholds"):
        st.write(f"Knee Angle: {thresholds.knee_angle_min}° - {thresholds.knee_angle_max}°")
        st.write(f"Hip Angle: {thresholds.hip_angle_min}° - {thresholds.hip_angle_max}°")
        st.write(f"Ankle Angle: {thresholds.ankle_angle_min}° - {thresholds.ankle_angle_max}°")

if __name__ == "__main__":
    main()