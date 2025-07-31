import React from "react";
import image from "./assets/image.png";
import image1 from "./assets/image1.png";
import end from "./assets/end.png";
import image2 from "./assets/image2.png";
import image3 from "./assets/image3.png";
import landing from "./assets/landing page.png";
import pose from "./assets/pose.png";
import standing from "./assets/standing.png";
import wellness from "./assets/wellness.png";
import Header from "./Header";
import { useNavigate } from "react-router-dom";
import FAQ from "./faq";

export default function Mypage() {
  const navigate = useNavigate();
  return (
    <div>
      <div>
        <Header />
        <div style={{ fontFamily: "Quattrocento, serif", lineHeight: "1.6" }}>
          {/* Hero Section */}
          <section
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "30px",
              backgroundColor: "#f5f5f5",
            }}
          >
            <div style={{ maxWidth: "800px" }}>
              <h1 style={{ fontSize: "6rem", marginBottom: "20px" }}>
                Introducing Our Posture Vision
              </h1>
              <p style={{ fontSize: "1.6rem", marginBottom: "25px" }}>
                Experience the revolutionary posture vision technology that
                combines fitness, yoga, and real-time analysis to help you
                achieve optimal posture.
              </p>
              <div>
                <button
                  style={{
                    marginRight: "10px",
                    padding: "12px 24px",
                    backgroundColor: "#007BFF",
                    color: "#fff",
                    border: "none",
                    borderRadius: "5px",
                    cursor: "pointer",
                  }}
                  onClick={() => navigate("/start")}
                >
                  Get Started
                </button>
                <button
                  style={{
                    padding: "12px 24px",
                    backgroundColor: "#28A745",
                    color: "#fff",
                    border: "none",
                    borderRadius: "5px",
                    cursor: "pointer",
                  }}
                  onClick={() => navigate("/faq")}
                >
                  Learn More
                </button>
              </div>
            </div>
            <img
              src={landing}
              alt="Posture Hero Image"
              style={{
                width: "650px",
                height: "650px",
                borderRadius: "10px",
                marginLeft: "25px",
                marginBottom: "20px",
                marginTop: "10px",
              }}
            />
          </section>

          {/* Features Section */}
          <section
            style={{
              textAlign: "center",
              padding: "40px 20px",
              backgroundColor: "#fff",
            }}
          >
            <h2 style={{ fontSize: "3rem", marginBottom: "20px" }}>
              Elevate Your Posture
            </h2>
            <p style={{ fontSize: "2rem", marginBottom: "40px" }}>
              Unlock the Power of Posture: Our innovative platform integrates
              advanced motion tracking.
            </p>
            <div
              style={{
                display: "flex",
                justifyContent: "space-around",
                flexWrap: "wrap",
                gap: "20px",
              }}
            >
              {[
                {
                  title: "Posture Perfection", //spinal backbone alignment
                  image: image1,
                  text: "Achieve a Balanced Posture: Our real-time posture analysis provides detailed insights.",
                  link: "http://localhost:8503/",
                },
                {
                  title: "Boost Your Potential", //ganashrees posturecorrect
                  image: image2,
                  text: "Transform Your Body and Mind with guidance and advanced tools.",
                  link: "http://localhost:8502/",
                },
                {
                  title: "Gym Exercises", //squats analysis
                  image: image3,
                  text: "Personalized Gym Coaching: Discover a better version of yourself.",
                  link: "http://localhost:8501/",
                },
              ].map((feature, index) => (
                <div
                  key={index}
                  style={{
                    padding: "20px",
                    border: "1px solid #ddd",
                    borderRadius: "10px",
                    maxWidth: "300px",
                    transition: "background-color 0.3s ease",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor = "#f0f0f0")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "white")
                  }
                >
                  <h3 style={{ fontSize: "1.5rem", marginBottom: "15px" }}>
                    {feature.title}
                  </h3>
                  <img
                    src={feature.image}
                    alt={feature.title}
                    style={{ width: "100px", marginBottom: "15px" }}
                  />
                  <p style={{ fontSize: "0.9rem", marginBottom: "15px" }}>
                    {feature.text}
                  </p>
                  <button
                    style={{
                      padding: "10px 20px",
                      backgroundColor: "#007BFF",
                      color: "#fff",
                      border: "none",
                      borderRadius: "5px",
                      cursor: "pointer",
                    }}
                    onClick={() => (window.location.href = feature.link)}
                  >
                    {feature.link === "posture.html"
                      ? "Try It Now"
                      : "Learn More"}
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Full-Screen Image Sections */}
          
          {[pose, standing, wellness].map((imageSrc, index) => (
            <section
              key={index}
              style={{
                display: "flex",
                justifyContent: "center",
                margin: "20px 0",
              }}
            >
              <img
                src={imageSrc}
                alt={`Full Screen Image ${index}`}
                style={{ width: "100%", borderRadius: "10px" }}
              />
            </section>
          ))}
          

          {/* Improve Your Posture Section */}
          
          <section
            style={{
              textAlign: "center",
              padding: "40px 20px",
              backgroundColor: "#ffe599",
              
            }}
          >
            <h2 style={{ fontSize: "2rem", marginBottom: "20px" }}>
              Improve Your Posture
            </h2>
            <button
              style={{
                padding: "10px 20px",
                backgroundColor: "#007BFF",
                color: "#fff",
                border: "none",
                borderRadius: "5px",
                cursor: "pointer",
              }}
              onClick={() => navigate("/start")}
            >
              Get Started
            </button>
          </section>

          {/* Add the "End" Image Section */}
          {/* 
          <section
            style={{
              display: "flex",
              justifyContent: "center",
              margin: "20px 0",
            }}
          >
            <img
              src={end}
              alt="End Section Image"
              style={{
                width: "100%",
                borderRadius: "10px",
              }}
            />
          </section>
          */}

          {/* Footer Section */}
          <footer
            style={{
              backgroundColor: "#222",
              color: "#fff",
              padding: "40px 20px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                flexWrap: "wrap",
              }}
            >
              <div style={{ maxWidth: "300px" }}>
                <div style={{ marginBottom: "15px" }}>
                  <span
                    style={{
                      width: "10px",
                      height: "10px",
                      backgroundColor: "#fff",
                      display: "inline-block",
                      borderRadius: "50%",
                    }}
                  ></span>
                </div>
                <p>© 2024 PostureVision, Inc. All rights reserved.</p>
              </div>
              {/* Footer Links */}
              {[
                {
                  title: "Company",
                  links: [
                    { name: "About", href: "about.html" },
                    { name: "Products", href: "products.html" },
                    { name: "Contact", href: "contact.html" },
                    { name: "FAQ", href: "faq.html" },
                  ],
                },
                {
                  title: "Resources",
                  links: [
                    { name: "Blog", href: "#" },
                    { name: "Videos", href: "#" },
                    { name: "Guides", href: "#" },
                    { name: "FAQs", href: "#" },
                  ],
                },
                {
                  title: "Support",
                  links: [
                    { name: "Help Center", href: "#" },
                    { name: "Live Chat", href: "#" },
                    { name: "Contact Us", href: "contact.html" },
                    { name: "Feedback", href: "#" },
                  ],
                },
              ].map((section, index) => (
                <div key={index}>
                  <h4>{section.title}</h4>
                  <ul style={{ listStyle: "none", padding: "0" }}>
                    {section.links.map((link, index) => (
                      <li
                        key={index}
                        style={{
                          marginBottom: "10px",
                        }}
                      >
                        <a
                          href={link.href}
                          style={{
                            color: "#fff",
                            textDecoration: "none",
                          }}
                        >
                          {link.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}
