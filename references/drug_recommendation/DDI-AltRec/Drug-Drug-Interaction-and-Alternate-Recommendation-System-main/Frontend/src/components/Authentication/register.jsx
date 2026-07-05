import { createUserWithEmailAndPassword } from "firebase/auth";
import React, { useState } from "react";
import { auth, db } from "./firebase";
import { setDoc, doc } from "firebase/firestore";
import { toast } from "react-toastify";
import SignInwithGoogle from "./signInWIthGoogle";
import registerImg from "../../assets/loginPage2.png"; // Use the same image as the login page

function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fname, setFname] = useState("");
  const [lname, setLname] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");

  const handleRegister = async (e) => {
    e.preventDefault();

    // Password validation
    if (password !== passwordConfirm) {
      toast.error("Passwords do not match", {
        position: "bottom-center",
      });
      return;
    }

    try {
      await createUserWithEmailAndPassword(auth, email, password);
      const user = auth.currentUser;
      console.log(user);
      if (user) {
        await setDoc(doc(db, "Users", user.uid), {
          email: user.email,
          firstName: fname,
          lastName: lname,
          photo: "",
        });
      }
      console.log("User Registered Successfully!!");
      toast.success("User Registered Successfully!!", {
        position: "top-center",
      });
      window.location.href = "/profile"; // Redirect to profile page
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
        <img src={registerImg} alt="Register" />
      </div>
      <div className="card shadow-sm container-fluid" style={{ width: "450px" }}>
        <div className="card-body p-4">
          <form onSubmit={handleRegister}>
            <div className="text-center mb-5">
              <h3 className="card-title fw-bold text-login">Register</h3>
              <p className="text-muted">Create your account</p>
            </div>

            <div className="row mb-3">
              <div className="col-md-6">
                <label htmlFor="firstName" className="form-label">
                  First Name
                </label>
                <div className="input-group">
                  <input
                    id="firstName"
                    type="text"
                    className="form-control"
                    placeholder="First name"
                    onChange={(e) => setFname(e.target.value)}
                    required
                  />
                </div>
              </div>
              <div className="col-md-6">
                <label htmlFor="lastName" className="form-label">
                  Last Name
                </label>
                <div className="input-group">
                  <input
                    id="lastName"
                    type="text"
                    className="form-control"
                    placeholder="Last name"
                    onChange={(e) => setLname(e.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="mb-3">
              <label htmlFor="email" className="form-label">
                Email Address
              </label>
              <div className="input-group">
                <input
                  id="email"
                  type="email"
                  className="form-control"
                  placeholder="Enter email"
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="mb-3">
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <div className="input-group">
                <input
                  id="password"
                  type="password"
                  className="form-control"
                  placeholder="Enter password"
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength="6"
                />
              </div>
              <div className="form-text text-muted">
                Password must be at least 6 characters
              </div>
            </div>

            <div className="mb-3">
              <label htmlFor="confirmPassword" className="form-label">
                Confirm Password
              </label>
              <div className="input-group">
                <input
                  id="confirmPassword"
                  type="password"
                  className="form-control"
                  placeholder="Confirm password"
                  onChange={(e) => setPasswordConfirm(e.target.value)}
                  required
                  minLength="6"
                />
              </div>
            </div>

            <div className="d-grid gap-2 mb-3">
              <button type="submit" className="btn btn-lg login-button">
                Register
              </button>
            </div>

            <div className="text-center mb-3">
              <SignInwithGoogle />
            </div>

            <div className="text-center">
              <p className="small text-muted">
                Already have an account?{" "}
                <a href="/login" className="text-primary fw-bold">
                  Login Here
                </a>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Register;