import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import Home from "./pages/Home/Home";
import Yoga from "./pages/Yoga/Yoga";
import About from "./pages/About/About";
import Tutorials from "./pages/Tutorials/Tutorials";
import Mypage from "./pages/Mypages/Mypage";

import "./App.css";
import FAQ from "./pages/Mypages/faq";
import Contact from "./pages/Mypages/Contact";


export default function App() {
  return (
    <Router>
      <Routes>
        {/* <Route path='/' element={<Home />}/> */}
        <Route path="/" element={<Mypage />} />
        <Route path="/start" element={<Yoga />} />
        <Route path="/about" element={<About />} />
        <Route path="/tutorials" element={<Tutorials />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/Contact" element={<Contact />} />
        
      </Routes>
    </Router>
  );
}
