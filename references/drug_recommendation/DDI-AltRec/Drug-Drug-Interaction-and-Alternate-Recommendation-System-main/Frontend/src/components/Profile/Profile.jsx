import React, { useState, useEffect } from "react";
import { Container, Row, Col, Form, Button, Card, Alert, Tabs, Tab, Spinner } from "react-bootstrap";
import { doc, getDoc, setDoc, updateDoc } from "firebase/firestore";
import { auth, db } from "../Authentication/firebase";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "../Dashboard/dashboard/DashboardLayout";

export default function UserPreferences() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [activeKey, setActiveKey] = useState("personal");

  const [userProfile, setUserProfile] = useState({
    // Personal Information
    name: "",
    email: "",
    age: "",
    gender: "",
    height: "",
    weight: "",
    location: "",
    
    // Medical Information
    pregnancyStatus: "not_applicable",
    allergies: "",
    medicalConditions: "",
    kidneyFunction: "normal",
    liverFunction: "normal",
    
    // Lifestyle Information
    smokingStatus: "non_smoker",
    alcoholConsumption: "none",
    dietaryRestrictions: "",
    caffeineConsumption: "low",
    exerciseFrequency: "moderate",
    
    
    // Additional Information
    previousAdverseReactions: ""
  });

  // Fetch user data from Firestore
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const user = auth.currentUser;
        if (!user) {
          navigate("/login");
          return;
        }

        const docRef = doc(db, "Users", user.uid);
        const docSnap = await getDoc(docRef);

        if (docSnap.exists()) {
          // Merge existing data with default state structure
          const userData = docSnap.data();
          setUserProfile(prevState => ({
            ...prevState,
            ...userData
          }));
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching user data:", error);
        setSaveError("Failed to load user data");
        setLoading(false);
      }
    };

    fetchUserData();
  }, [navigate]);

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setUserProfile({
      ...userProfile,
      [name]: value
    });
  };

  // Save user profile data
  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setSaveSuccess(false);
    setSaveError(null);

    try {
      const user = auth.currentUser;
      if (!user) {
        navigate("/login");
        return;
      }

      const userRef = doc(db, "Users", user.uid);
      await updateDoc(userRef, userProfile);
      
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      console.error("Error updating user profile:", error);
      setSaveError("Failed to save changes. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Container className="d-flex justify-content-center align-items-center" style={{ minHeight: "60vh" }}>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </Container>
    );
  }

  return (
    <>
    <DashboardLayout />
    <Container className="py-4">
      <Card className="shadow-sm">
        <Card.Header as="h4" className="bg-dark text-white">
          User Preferences
        </Card.Header>
        <Card.Body>
          {saveSuccess && (
            <Alert variant="success" className="mb-3">
              Your profile has been updated successfully!
            </Alert>
          )}
          
          {saveError && (
            <Alert variant="danger" className="mb-3">
              {saveError}
            </Alert>
          )}

          <Form onSubmit={handleSubmit}>
            <Tabs
              activeKey={activeKey}
              onSelect={(k) => setActiveKey(k)}
              className="mb-4"
            >
              <Tab eventKey="personal" title="Personal Information">
                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Full Name</Form.Label>
                      <Form.Control
                        type="text"
                        name="name"
                        value={userProfile.name}
                        onChange={handleChange}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Email Address</Form.Label>
                      <Form.Control
                        type="email"
                        name="email"
                        value={userProfile.email}
                        onChange={handleChange}
                        disabled
                      />
                      <Form.Text className="text-muted">
                        Email cannot be changed.
                      </Form.Text>
                    </Form.Group>
                  </Col>
                </Row>

                <Row className="mb-3">
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Age</Form.Label>
                      <Form.Control
                        type="number"
                        name="age"
                        value={userProfile.age}
                        onChange={handleChange}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Gender</Form.Label>
                      <Form.Select
                        name="gender"
                        value={userProfile.gender}
                        onChange={handleChange}
                      >
                        <option value="">Select Gender</option>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                        <option value="other">Other</option>
                        <option value="prefer_not_to_say">Prefer not to say</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Location</Form.Label>
                      <Form.Control
                        type="text"
                        name="location"
                        placeholder="City, Country"
                        value={userProfile.location}
                        onChange={handleChange}
                      />
                    </Form.Group>
                  </Col>
                </Row>

                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Height (cm)</Form.Label>
                      <Form.Control
                        type="number"
                        name="height"
                        value={userProfile.height}
                        onChange={handleChange}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Weight (kg)</Form.Label>
                      <Form.Control
                        type="number"
                        name="weight"
                        value={userProfile.weight}
                        onChange={handleChange}
                      />
                    </Form.Group>
                  </Col>
                </Row>
              </Tab>

              <Tab eventKey="medical" title="Medical Information">
                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Pregnancy Status</Form.Label>
                      <Form.Select
                        name="pregnancyStatus"
                        value={userProfile.pregnancyStatus}
                        onChange={handleChange}
                      >
                        <option value="not_applicable">Not Applicable</option>
                        <option value="pregnant">Pregnant</option>
                        <option value="breastfeeding">Breastfeeding</option>
                        <option value="planning">Planning Pregnancy</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Kidney Function</Form.Label>
                      <Form.Select
                        name="kidneyFunction"
                        value={userProfile.kidneyFunction}
                        onChange={handleChange}
                      >
                        <option value="normal">Normal</option>
                        <option value="mild_impairment">Mild Impairment</option>
                        <option value="moderate_impairment">Moderate Impairment</option>
                        <option value="severe_impairment">Severe Impairment</option>
                        <option value="unknown">Unknown</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                </Row>

                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Liver Function</Form.Label>
                      <Form.Select
                        name="liverFunction"
                        value={userProfile.liverFunction}
                        onChange={handleChange}
                      >
                        <option value="normal">Normal</option>
                        <option value="mild_impairment">Mild Impairment</option>
                        <option value="moderate_impairment">Moderate Impairment</option>
                        <option value="severe_impairment">Severe Impairment</option>
                        <option value="unknown">Unknown</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                  </Col>
                </Row>

                <Form.Group className="mb-3">
                  <Form.Label>Known Allergies</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={2}
                    name="allergies"
                    placeholder="List any drug allergies and reactions"
                    value={userProfile.allergies}
                    onChange={handleChange}
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Medical Conditions</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={3}
                    name="medicalConditions"
                    placeholder="List any chronic conditions, diagnoses, or health concerns"
                    value={userProfile.medicalConditions}
                    onChange={handleChange}
                  />
                </Form.Group>

                <Form.Group className="mb-3">
                  <Form.Label>Previous Adverse Drug Reactions</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={2}
                    name="previousAdverseReactions"
                    placeholder="Describe any previous medication side effects or reactions"
                    value={userProfile.previousAdverseReactions}
                    onChange={handleChange}
                  />
                </Form.Group>
              </Tab>

              <Tab eventKey="lifestyle" title="Lifestyle">
                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Smoking Status</Form.Label>
                      <Form.Select
                        name="smokingStatus"
                        value={userProfile.smokingStatus}
                        onChange={handleChange}
                      >
                        <option value="non_smoker">Non-Smoker</option>
                        <option value="former_smoker">Former Smoker</option>
                        <option value="occasional">Occasional</option>
                        <option value="regular">Regular</option>
                        <option value="heavy">Heavy</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Alcohol Consumption</Form.Label>
                      <Form.Select
                        name="alcoholConsumption"
                        value={userProfile.alcoholConsumption}
                        onChange={handleChange}
                      >
                        <option value="none">None</option>
                        <option value="occasional">Occasional</option>
                        <option value="moderate">Moderate</option>
                        <option value="heavy">Heavy</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                </Row>

                <Row className="mb-3">
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Caffeine Consumption</Form.Label>
                      <Form.Select
                        name="caffeineConsumption"
                        value={userProfile.caffeineConsumption}
                        onChange={handleChange}
                      >
                        <option value="none">None</option>
                        <option value="low">Low (1 cup/day)</option>
                        <option value="moderate">Moderate (2-3 cups/day)</option>
                        <option value="high">High (4+ cups/day)</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group className="mb-3">
                      <Form.Label>Exercise Frequency</Form.Label>
                      <Form.Select
                        name="exerciseFrequency"
                        value={userProfile.exerciseFrequency}
                        onChange={handleChange}
                      >
                        <option value="sedentary">Sedentary</option>
                        <option value="light">Light (1-2 days/week)</option>
                        <option value="moderate">Moderate (3-4 days/week)</option>
                        <option value="active">Active (5+ days/week)</option>
                      </Form.Select>
                    </Form.Group>
                  </Col>
                </Row>
              </Tab>
            </Tabs>

            <div className="d-flex justify-content-between mt-4">
              <Button variant="secondary" onClick={() => navigate(-1)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                type="submit"
                disabled={saving}
              >
                {saving ? (
                  <>
                    <Spinner
                      as="span"
                      animation="border"
                      size="sm"
                      role="status"
                      aria-hidden="true"
                      className="me-2"
                    />
                    Saving...
                  </>
                ) : (
                  "Save Changes"
                )}
              </Button>
            </div>
          </Form>
        </Card.Body>
        <Card.Footer className="text-muted">
          <small>
            Your health information is used to provide personalized medication
            interaction alerts.
          </small>
        </Card.Footer>
      </Card>
    </Container>
    </>
  );
}