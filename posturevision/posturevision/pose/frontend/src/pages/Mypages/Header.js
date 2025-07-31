import React from "react";
import "./Style.css";
import { Link } from "react-router-dom";

export default function Header() {
  return (
    <div>
      <header>
        <div className="logo-container">
          <div className="triangle-logo">
            <span className="dot"></span>
          </div>
          <div className="logo">PostureVision</div>
        </div>
        <nav>
          <Link to="/about">About</Link>
    
          <Link to="/contact">Contact</Link>
          <Link to="/faq">FAQ</Link>
        </nav>
      </header>
    </div>
  );
}
