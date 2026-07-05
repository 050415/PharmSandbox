import { signInWithEmailAndPassword } from "firebase/auth";
import React, { useState } from "react";
import { auth } from "./firebase";
import { toast } from "react-toastify";
import SignInwithGoogle from "./signInWIthGoogle";
import Navbar from "../LandingPage/Navbar";
import loginImg from "../../assets/loginPage2.png";

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await signInWithEmailAndPassword(auth, email, password);
      console.log("User logged in Successfully");
      window.location.href = "/mainPage";
      toast.success("User logged in Successfully", {
        position: "top-center",
      });
    } catch (error) {
      console.log(error.message);
      toast.error(error.message, {
        position: "bottom-center",
      });
    }
  };

  return (

    <div className="container d-flex justify-content-center align-items-center vh-100">
     <div>
        <img src={loginImg}/>
      </div>
      <div className="card shadow-sm container-fluid" style={{ width: '450px' }}>
        <div className="card-body p-4">
          <form onSubmit={handleSubmit}>
            <div className="text-center mb-5">
              <h3 className="card-title fw-bold text-login">Login</h3>
              <p className="text-muted">Enter your credentials</p>
            </div>

            <div className="mb-3">
              <label htmlFor="email" className="form-label">Email address</label>
              <div className="input-group">
                <input
                  id="email"
                  type="email"
                  className="form-control"
                  placeholder="Enter email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="mb-5">
              <label htmlFor="password" className="form-label">Password</label>
              <div className="input-group">
                <input
                  id="password"
                  type="password"
                  className="form-control"
                  placeholder="Enter password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="d-grid gap-2 mb-3">
              <button
                type="submit"
                className="btn btn-lg login-button"
              >
                Login
              </button>
            </div>

            <div className="text-center mb-3">
              <SignInwithGoogle />
            </div>

            <div className="text-center">
              <p className="small text-muted">
                New user? <a href="/register" className="text-primary fw-bold">Register Here</a>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>

  );
}

export default Login;