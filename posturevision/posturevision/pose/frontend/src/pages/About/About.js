import React from 'react';
import './About.css';

export default function About() {
    return (
        <div className="about-container">
            <div className="about-content-wrapper">
                <h1 className="about-heading"> About PostureVision</h1>
                <p className="about-content">
                    <strong>Overview:</strong><br />
                    PostureVision is a solution using computer vision and deep learning to monitor and correct posture in real time. Designed for fitness enthusiasts, physiotherapy patients, and everyday users, it addresses the widespread issue of poor posture in modern sedentary lifestyles.
                </p>
                <p className="about-content">
                    <strong>Key Features:</strong><br />
                    - Continuous Monitoring: Tracks body alignment during activities, workouts, and rehabilitation.<br />
                
                    - Real-Time Feedback: Provides instant posture alerts and suggestions to prevent injuries.<br />
                    
                    - Personalized Correction: Adapts to individual body types using machine learning.<br />
                    
                    - Seamless Integration: Works across different environments (desk, gym, therapy).<br />
                    
                    - Data Analytics & Tracking: Visualizes progress and improvements over time.<br />
                    
                </p>
                <p className="about-content">
                    PostureVision: Real-Time Body Posture Analysis addresses the gap by utilizing technology to provide an effective, real-time posture correction system using computer vision and deep neural networks.
                </p>
            </div>
        </div>
    );
}
