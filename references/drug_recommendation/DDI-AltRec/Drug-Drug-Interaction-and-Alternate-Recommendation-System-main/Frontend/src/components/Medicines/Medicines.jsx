import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Nav, Card } from 'react-bootstrap';
import { onAuthStateChanged } from 'firebase/auth';
import axios from 'axios';
import { auth } from '../Authentication/firebase';
import DashboardLayout from '../Dashboard/dashboard/DashboardLayout';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, parseISO } from 'date-fns';
import { Calendar } from 'react-bootstrap-icons';
import CalendarComponent from './medicines/Calender';
import { FaPills, FaClock, FaNotesMedical, FaCalendarAlt, FaInfoCircle } from "react-icons/fa";

const Medicines = () => {
    const [user, setUser] = useState(null);
    const [activeTab, setActiveTab] = useState('medicationList');
    const [selectedDate, setSelectedDate] = useState(new Date());
    const [medicationsOnSelectedDate, setMedicationsOnSelectedDate] = useState([]);
    const [medications, setMedications] = useState([]);
    const [newMedication, setNewMedication] = useState({
        name: '',
        dosage: '',
        times: [],
        medicalCondition: '',
        startDate: '',
        endDate: '',
        notes: ''
    });

    // Toast state and function (similar to Appointments component)
    const [toast, setToast] = useState({
        show: false,
        message: '',
        variant: 'success'
    });

    const showToast = (message, variant = 'success') => {
        setToast({ show: true, message, variant });
        setTimeout(() => {
            setToast({ show: false, message: '', variant: 'success' });
        }, 3000);
    };

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (currentUser) {
                setUser(currentUser);
                try {
                    const token = await currentUser.getIdToken();
                    const response = await axios.get('http://127.0.0.1:5000/medications', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        withCredentials: true,
                    });
                    setMedications(response.data);
                } catch (error) {
                    showToast(error.message, 'danger');
                }
            } else {
                setUser(null);
                setMedications([]);
            }
        });

        return () => unsubscribe();
    }, []);

    // Filter medications for selected date
    useEffect(() => {
        if (medications.length > 0) {
            const formattedSelectedDate = format(selectedDate, 'yyyy-MM-dd');
            const filteredMedications = medications.filter(med =>
                (med.startDate <= formattedSelectedDate) &&
                (!med.endDate || med.endDate >= formattedSelectedDate)
            );
            setMedicationsOnSelectedDate(filteredMedications);
        }
    }, [selectedDate, medications]);

    const handleAddMedication = async () => {
        if (!user) {
            showToast('Please log in first', 'danger');
            return;
        }

        try {
            const token = await user.getIdToken();
            await axios.post('http://127.0.0.1:5000/addmedication', newMedication, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            // Refresh medications list
            const response = await axios.get('http://127.0.0.1:5000/medications', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            setMedications(response.data);

            // Reset form and show success message
            setNewMedication({
                name: '',
                dosage: '',
                times: [],
                medicalCondition: '',
                startDate: '',
                endDate: '',
                notes: ''
            });
            setActiveTab('medicationList');
            showToast('Medication Added Successfully');
        } catch (error) {
            showToast(error.message, 'danger');
        }
    };

    return (
        <>
            <DashboardLayout />
            <Container className="mt-4">
                <h1 className="mb-4">Medication Tracker</h1>

                <Nav variant="tabs" activeKey={activeTab} onSelect={(selectedKey) => setActiveTab(selectedKey)}>
                    <Nav.Item>
                        <Nav.Link eventKey="medicationList">Medication List</Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                        <Nav.Link eventKey="addMedication">Add Medication</Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                        <Nav.Link eventKey="calendarView">Calendar View</Nav.Link>
                    </Nav.Item>
                </Nav>

                {activeTab === 'medicationList' && (
                    <div className="mt-3">
                        {medications.length === 0 ? (
                            <p className="text-center">No medications added yet</p>
                        ) : (
                            medications.map((med, index) => (
                                <Card key={index} className="mb-4 border-0 rounded-lg overflow-hidden shadow-sm">
                                    <Card.Body className="p-0">
                                        {/* Colored header section */}
                                        <div className="bg-gradient bg-primary p-3 text-white">
                                            <div className="d-flex justify-content-between align-items-center">
                                                <h5 className="mb-0 fw-bold d-flex align-items-center">
                                                    <FaPills className="me-2" />
                                                    {med.name}
                                                </h5>
                                            </div>
                                        </div>

                                        {/* Content section with improved spacing and layout */}
                                        <div className="p-3">
                                            <div className="row g-3">
                                                <div className="col-md-6">
                                                    <div className="d-flex align-items-center">
                                                        <div className="bg-light rounded p-2 me-3">
                                                            <FaPills className="text-primary" />
                                                        </div>
                                                        <div>
                                                            <small className="text-muted d-block">Dosage</small>
                                                            <span className="fw-medium">{med.dosage}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="col-md-6">
                                                    <div className="d-flex align-items-center">
                                                        <div className="bg-light rounded p-2 me-3">
                                                            <FaClock className="text-primary" />
                                                        </div>
                                                        <div>
                                                            <small className="text-muted d-block">Times</small>
                                                            <span className="fw-medium">{med.times.join(", ")}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="col-md-6">
                                                    <div className="d-flex align-items-center">
                                                        <div className="bg-light rounded p-2 me-3">
                                                            <FaCalendarAlt className="text-primary" />
                                                        </div>
                                                        <div>
                                                            <small className="text-muted d-block">Start Date</small>
                                                            <span className="fw-medium">{med.start_date}</span>
                                                        </div>
                                                    </div>
                                                </div>

                                                {med.end_date && (
                                                    <div className="col-md-6">
                                                        <div className="d-flex align-items-center">
                                                            <div className="bg-light rounded p-2 me-3">
                                                                <FaCalendarAlt className="text-primary" />
                                                            </div>
                                                            <div>
                                                                <small className="text-muted d-block">End Date</small>
                                                                <span className="fw-medium">{med.end_date}</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                {med.medical_condition && (
                                                    <div className="col-12">
                                                        <div className="d-flex align-items-center">
                                                            <div className="bg-light rounded p-2 me-3">
                                                                <FaNotesMedical className="text-primary" />
                                                            </div>
                                                            <div>
                                                                <small className="text-muted d-block">Condition</small>
                                                                <span className="fw-medium">{med.medical_condition}</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}

                                                {med.notes && (
                                                    <div className="mt-3 pt-3 border-top">
                                                        <div className="d-flex">
                                                            <div className="bg-light rounded p-2 me-3 align-self-start">
                                                                <FaInfoCircle className="text-primary" />
                                                            </div>
                                                            <div>
                                                                <small className="text-muted d-block">Notes</small>
                                                                <p className="mb-0">{med.notes}</p>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </Card.Body>
                                </Card>
                            ))
                        )}
                    </div>
                )}

                {activeTab === 'addMedication' && (
                    <Container className="mt-3">
                        <div className="p-4 border rounded shadow-sm">
                            <Form className="mt-3">
                                <Row>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>Medication Name</Form.Label>
                                            <Form.Control
                                                type="text"
                                                placeholder="e.g., Amoxicillin"
                                                value={newMedication.name}
                                                onChange={(e) => setNewMedication({
                                                    ...newMedication,
                                                    name: e.target.value
                                                })}
                                                required
                                            />
                                        </Form.Group>
                                    </Col>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>Dosage</Form.Label>
                                            <Form.Control
                                                type="text"
                                                placeholder="e.g., 500mg"
                                                value={newMedication.dosage}
                                                onChange={(e) => setNewMedication({
                                                    ...newMedication,
                                                    dosage: e.target.value
                                                })}
                                                required
                                            />
                                        </Form.Group>
                                    </Col>
                                </Row>

                                <Row>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>Medication Times</Form.Label>
                                            <div>
                                                <Button
                                                    variant="outline-primary"
                                                    size="sm"
                                                    className="mb-2"
                                                    onClick={() => {
                                                        // Add a new empty time to the times array
                                                        setNewMedication({
                                                            ...newMedication,
                                                            times: [...(newMedication.times || []), '']
                                                        });
                                                    }}
                                                >
                                                    + Add Time
                                                </Button>

                                                {(newMedication.times || []).map((time, index) => (
                                                    <Row key={index} className="mb-2 align-items-center">
                                                        <Col xs={9}>
                                                            <Form.Control
                                                                type="time"
                                                                value={time}
                                                                onChange={(e) => {
                                                                    const updatedTimes = [...newMedication.times];
                                                                    updatedTimes[index] = e.target.value;
                                                                    setNewMedication({
                                                                        ...newMedication,
                                                                        times: updatedTimes
                                                                    });
                                                                }}
                                                            />
                                                        </Col>
                                                        <Col xs={3}>
                                                            <Button
                                                                variant="outline-danger"
                                                                size="sm"
                                                                onClick={() => {
                                                                    const updatedTimes = newMedication.times.filter((_, i) => i !== index);
                                                                    setNewMedication({
                                                                        ...newMedication,
                                                                        times: updatedTimes
                                                                    });
                                                                }}
                                                            >
                                                                Remove
                                                            </Button>
                                                        </Col>
                                                    </Row>
                                                ))}

                                                {(!newMedication.times || newMedication.times.length === 0) && (
                                                    <div className="text-muted">No times added yet. Click '+ Add Time' to specify when to take this medication.</div>
                                                )}
                                            </div>
                                        </Form.Group>
                                    </Col>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>Medical Condition</Form.Label>
                                            <Form.Control
                                                type="text"
                                                placeholder="e.g., Bacterial Infection"
                                                value={newMedication.medicalCondition}
                                                onChange={(e) => setNewMedication({
                                                    ...newMedication,
                                                    medicalCondition: e.target.value
                                                })}
                                                required
                                            />
                                        </Form.Group>
                                    </Col>
                                </Row>

                                <Row>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>Start Date</Form.Label>
                                            <Form.Control
                                                type="date"
                                                value={newMedication.startDate}
                                                onChange={(e) => setNewMedication({
                                                    ...newMedication,
                                                    startDate: e.target.value
                                                })}
                                                required
                                            />
                                        </Form.Group>
                                    </Col>
                                    <Col md={6}>
                                        <Form.Group className="mb-3">
                                            <Form.Label>End Date</Form.Label>
                                            <Form.Control
                                                type="date"
                                                value={newMedication.endDate}
                                                onChange={(e) => setNewMedication({
                                                    ...newMedication,
                                                    endDate: e.target.value
                                                })}
                                            />
                                        </Form.Group>
                                    </Col>
                                </Row>

                                <Form.Group className="mb-3">
                                    <Form.Label>Notes (Optional)</Form.Label>
                                    <Form.Control
                                        as="textarea"
                                        rows={3}
                                        placeholder="Additional instructions or notes about this medication"
                                        value={newMedication.notes}
                                        onChange={(e) => setNewMedication({
                                            ...newMedication,
                                            notes: e.target.value
                                        })}
                                    />
                                </Form.Group>

                                <Button
                                    variant="dark"
                                    onClick={handleAddMedication}
                                    className="w-100"
                                >
                                    Add Medication
                                </Button>
                            </Form>
                        </div>
                    </Container>
                )}

                {activeTab === 'calendarView' && (
                    <CalendarComponent
                        selectedDate={selectedDate}
                        setSelectedDate={setSelectedDate}
                        medications={medications}
                    />
                )}
            </Container>
        </>
    );
};

export default Medicines;