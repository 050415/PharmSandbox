import React from "react";
import BannerImage from "../../assets/drug-interaction.png";
import Navbar from "./Navbar";
import { FiArrowRight } from "react-icons/fi";

const Home = () => {
  return (
    <div className="home-container" id="home">
      <Navbar />
      <div className="home-banner-container">
        <div className="home-text-section">
          <h1 className="primary-heading">
            DDI Checker
          </h1>
          <p className="primary-text">
            Analyse Drug-Drug Interactions (DDIs) and get the results in seconds.
            Get the most accurate and reliable information on drug interactions.
          </p>
          <button className="secondary-button">
            Learn More <FiArrowRight />{" "}
          </button>
        </div>
        <div className="home-image-section">
          <img src={BannerImage} alt="" />
        </div>
      </div>
    </div>
  );
};

export default Home;