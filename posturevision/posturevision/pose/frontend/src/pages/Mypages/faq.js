import React, { useState } from "react";
import "./faq.css";

export default function FAQ() {
    const [activeIndex, setActiveIndex] = useState(null);

    const toggleFAQ = (index) => {
        setActiveIndex(index === activeIndex ? null : index);
    };

    const faqData = [
        { 
            question: "Who should use PostureVision?", 
            answer: "Anyone who wants to monitor and improve their posture using ML-based real-time analysis."
        },
        { 
            question: "What is required to use PostureVision?", 
            answer: "You only need a device with a camera and an internet connection. The tool works seamlessly without additional equipment." 
        },
        { 
            question: "Do I need technical skills to use PostureVision?", 
            answer: "No technical skills are required. The platform is user-friendly and easy to navigate for all users." 
        },
        { 
            question: "Is my data secure with PostureVision?", 
            answer: "Yes, we prioritize user privacy and ensure all data is  NOT stored." 
        },
        { 
            question: "Can PostureVision be used for rehabilitation?", 
            answer: "Yes, PostureVision is suitable for physiotherapy patients to track and improve recovery progress." 
        },
        { 
            question: "Is there a free trial available?", 
            answer: "Yes, we offer a free trial period so users can explore the features and benefits of PostureVision." 
        },
        { 
            question: "Can PostureVision help during workouts?", 
            answer: "Absolutely. PostureVision provides real-time feedback to ensure proper alignment during exercises, preventing injuries and optimizing performance." 
        },
        { 
            question: "What devices are compatible with PostureVision?", 
            answer: "PostureVision is compatible with smartphones, tablets, and laptops that have a working camera and internet connection." 
        },
        { 
            question: "Does PostureVision support multiple users?", 
            answer: "No, PostureVision does not supports multiple user profiles together on one screen." 
        },
        
    ];

    return (
        <div className="faq-container">
            <div className="faq-header">
                <h1>Frequently Asked Questions</h1>
                <p>Can't find what you're looking for? <span>Here are the answers for your queries.</span></p>
            </div>
            <div className="faq-content">
                {faqData.map((item, index) => (
                    <div
                        className={`faq-item ${activeIndex === index ? "active" : ""}`}
                        key={index}
                        onClick={() => toggleFAQ(index)}
                    >
                        <h2>{item.question}</h2>
                        <p>{item.answer}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
