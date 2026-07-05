import '../../App.css'
import About from './About';
import Contact from './Contact';
import Features from './Features';
import Footer from './Footer';
import Home from './Home';
import React from 'react';

function LandingPage() {
  return (
    <div className="Landing-Page">
      <Home />
      <Features />
      <About />
      {/* <Contact />
      <Footer /> */}
    </div>
  );
}

export default LandingPage;