import React, { useState } from "react";
import emailjs from "emailjs-com";
import "./Contact.css";

export default function Contact() {
  const [formData, setFormData] = useState({
    from_name: "",
    to_name: "Neha", // Static recipient name
    message: "",
  });

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    // Validation
    if (!formData.from_name || !formData.message) {
      setError("All fields are required.");
      return;
    }

    // Send email using EmailJS
    emailjs
      .send(
        "default_service", // Replace with your EmailJS service ID
        "template_rv7i6vm", // Replace with your EmailJS template ID
        formData,
        "A3MZZ6zLukmFLEPm0" // Replace with your EmailJS public key
      )
      .then(
        () => {
          setSuccess("Message sent successfully!");
          setFormData({ from_name: "", to_name: "Neha", message: "" });
        },
        () => {
          setError("Failed to send the message. Please try again later.");
        }
      );
  };

  return (
    <div className="contact-container">
      <h1>Contact Us</h1>
      <p>We’d love to hear from you! Please fill out the form below to get in touch.</p>
      <form className="contact-form" onSubmit={handleSubmit}>
        {error && <p className="error">{error}</p>}
        {success && <p className="success">{success}</p>}
        <div className="form-group">
          <label htmlFor="from_name">Your Name:</label>
          <input
            type="text"
            id="from_name"
            name="from_name"
            placeholder="Your Name"
            value={formData.from_name}
            onChange={handleChange}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="message">Message:</label>
          <textarea
            id="message"
            name="message"
            placeholder="Your Message"
            rows="5"
            value={formData.message}
            onChange={handleChange}
            required
          ></textarea>
        </div>
        <button type="submit">Send Message</button>
      </form>
    </div>
  );
}
