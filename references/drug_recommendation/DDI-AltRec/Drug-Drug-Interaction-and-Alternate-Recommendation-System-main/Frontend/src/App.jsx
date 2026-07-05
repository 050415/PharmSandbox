import React, { useEffect } from "react";
import "../node_modules/bootstrap/dist/css/bootstrap.min.css";
import "../node_modules/bootstrap/dist/js/bootstrap.bundle";
import LandingPage from './components/LandingPage/LandingPage';
import "./App.css";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import Login from "./components/Authentication/login";
import SignUp from "./components/Authentication/register";


import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import UserPreferences from "./components/Profile/Profile";
import { useState } from "react";
import { auth } from "./components/Authentication/firebase";
import Appointments from "./components/Appointments/Appointments";
import Medicines from "./components/Medicines/Medicines";
import MainPage from "./components/Dashboard/profile";
import MedicationManagement from "./components/InteractionTracker/InteractionTracker";

function App() {
  const [user, setUser] = useState();
  useEffect(() => {
    auth.onAuthStateChanged((user) => {
      setUser(user);
    });
  });
  return (
    <Router>
      <div className="App">
        <div className="auth-wrapper">
          <div className="auth-inner">
            <Routes>
              <Route path = "/landingpage" element = {<LandingPage />} />
              <Route
                path="/"
                element={user ? <Navigate to="/mainPage" /> : <Login />}
              />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<SignUp />} />
              <Route path="/mainPage" element={<MainPage />} />
              <Route path="/appointments" element={<Appointments />} />
              <Route path="/medicines" element={<Medicines />} />
              <Route path="/profile" element={<UserPreferences />} />
              <Route path="/interactions" element={<MedicationManagement />} />
            </Routes>
            <ToastContainer />
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;