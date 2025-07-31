# PostureVision: Real-Time Body Posture Analysis

### Overview
"PostureVision" is a mini project report on a real-time body posture analysis application. The primary objective of the project is to create an accessible and non-invasive solution for monitoring and correcting body posture using a standard webcam. The system is designed to provide real-time feedback for various activities, including general posture, exercises, and yoga poses.

### Features
* **Posture Analysis**: The application can capture both sitting and standing postures and provides real-time feedback to ensure proper body alignment.
* **Exercise Tracking**: It tracks joint movement, corrects form, and counts repetitions during exercises like arm curls and squats.
* **Yoga Pose Evaluation**: The system detects and corrects yoga poses using key-point detection and pre-trained models.
* **Instantaneous Feedback**: Users receive instant audio and visual cues to help them make immediate corrections to their posture or movements.
* **Spine Posture Detection**: The application successfully detects spine curvature and alignment, which is useful for activities involving prolonged sitting or standing.
* **Accessibility**: PostureVision is designed to work in various environments, such as homes, gyms, and therapy centers, and only requires a standard webcam.

### Technology Stack
The application is built using a combination of tools and frameworks:
* **Streamlit**: Used for building an interactive and user-friendly web interface.
* **Mediapipe**: A lightweight computer vision library that efficiently detects 17 body key points in real-time.
* **TensorFlow MoveNet**: A pre-trained deep learning model for high-accuracy pose estimation. It has two variants: Lightning (faster) and Thunder (more accurate).
* **OpenCV (cv2) and Keras**: Used for live video capture from the webcam and frame-by-frame image processing.
* **Numpy**: Essential for mathematical calculations, particularly for computing joint angles based on key-point coordinates.
* **Python**: The primary programming language used to develop the entire application.
* **pyttsx3**: A text-to-speech library used to provide non-blocking audio feedback to the user.

### Methodology
The proposed methodology for PostureVision involves several key steps:
1. **Input Acquisition**: Capturing live video frames from a user's webcam using the OpenCV library and option to upload video files too
2. **Pose Detection**: Using Mediapipe Pose and MoveNet models to detect and estimate 17 key points on the human body.
3. **Processing**: Computing angles between joints to assess exercise performance and checking the alignment of key points for posture analysis.
4. **Feedback**: Providing real-time visual and audio feedback, including overlays, repetition counts, and alerts for incorrect posture.

### Future Scope
The project's future development can focus on several key areas to enhance its functionality and performance:
* **Improved Accuracy**: Integrating advanced filtering and adaptive lighting adjustments to improve detection in low-light conditions or during rapid movements.
* **Expanded Library**: Expanding the range of supported physiotherapy exercises and yoga poses.
* **Data Storage**: Implementing a database system to track user progress and historical data for long-term use.
* **Advanced Feedback**: Adding more detailed visual and textual feedback to identify specific areas of misalignment.
* **Wearable Integration**: Synchronizing the application with wearable sensors to improve the accuracy of joint movement analysis.
* **Mobile Application**: Developing a mobile-friendly version to make the application more convenient and accessible.
