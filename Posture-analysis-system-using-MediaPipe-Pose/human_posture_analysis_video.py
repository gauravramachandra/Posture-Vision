import cv2
import math as m
import mediapipe as mp

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

# Placeholder for warning function
def sendWarning():
    print("Bad posture alert!")

def main():
    # Constants and color definitions
    font = cv2.FONT_HERSHEY_SIMPLEX
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    # Frame counters
    good_frames = 0
    bad_frames = 0

    while cap.isOpened():
        # Capture frame
        success, image = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Get frame dimensions
        h, w = image.shape[:2]

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = pose.process(rgb_image)

        # Convert back to BGR for drawing
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Extract landmarks
            lm = results.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark

            # Get key points
            l_shldr_x = int(lm[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm[lmPose.RIGHT_SHOULDER].y * h)
            l_ear_x = int(lm[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm[lmPose.LEFT_EAR].y * h)
            l_hip_x = int(lm[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm[lmPose.LEFT_HIP].y * h)

            # Calculate offset
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Camera alignment check
            if offset < 100:
                cv2.putText(image, f'{int(offset)} Aligned', (w - 150, 30), font, 0.9, green, 2)
            else:
                cv2.putText(image, f'{int(offset)} Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            # Calculate angles
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Posture determination
            if neck_inclination < 30 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1
                color = light_green
            else:
                good_frames = 0
                bad_frames += 1
                color = red

            # Angle text
            angle_text_string = f'Neck: {int(neck_inclination)}  Torso: {int(torso_inclination)}'
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, color, 2)

            # Draw landmarks
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)

            # Draw additional markers
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            # Draw lines
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), color, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), color, 4)

            # Posture time calculation
            good_time = good_frames / cap.get(cv2.CAP_PROP_FPS)
            bad_time = bad_frames / cap.get(cv2.CAP_PROP_FPS)

            # Display posture time
            if good_time > 0:
                time_text = f'Good Posture Time: {round(good_time, 1)}s'
                cv2.putText(image, time_text, (10, h - 20), font, 0.9, green, 2)
            else:
                time_text = f'Bad Posture Time: {round(bad_time, 1)}s'
                cv2.putText(image, time_text, (10, h - 20), font, 0.9, red, 2)

            # Send warning if bad posture persists
            if bad_time > 180:
                sendWarning()

        # Display the frame
        cv2.imshow('Posture Detection', image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    