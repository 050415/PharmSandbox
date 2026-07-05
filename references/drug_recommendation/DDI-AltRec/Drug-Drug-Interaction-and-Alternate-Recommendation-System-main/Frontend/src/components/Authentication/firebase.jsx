// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import {getAuth} from "firebase/auth";
import {getFirestore} from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBJsZVFFqbhTCo4bZYVyI7qbRGpVBOB_fk",
  authDomain: "major-project-5f739.firebaseapp.com",
  projectId: "major-project-5f739",
  storageBucket: "major-project-5f739.firebasestorage.app",
  messagingSenderId: "264696943617",
  appId: "1:264696943617:web:482b06a70e60efc2d05c00"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth=getAuth();
export const db=getFirestore(app);
export default app;