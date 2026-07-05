import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Alert, Card, Badge, Button } from 'react-bootstrap';
import { onAuthStateChanged } from 'firebase/auth';
import axios from 'axios';
import { auth } from '../Authentication/firebase';
import DashboardLayout from '../Dashboard/dashboard/DashboardLayout';
import { ExclamationTriangleFill } from 'react-bootstrap-icons';

const MedicationManagement = () => {
    const [user, setUser] = useState(null);
    const [interactions, setInteractions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Toast state and function
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

    // Fetch medication interactions
    const fetchInteractions = async () => {
        try {
            setLoading(true);
            const token = await user.getIdToken();
            const response = await axios.get('http://127.0.0.1:5000/medication-interactions', {
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                withCredentials: true,
            });
            setInteractions(response.data.interactions);
            setLoading(false);
            console.log(response.data.interactions);
        } catch (error) {
            setError(error.message);
            showToast(error.message, 'danger');
            setLoading(false);
        }
    };

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (currentUser) {
                setUser(currentUser);
                try {
                    const token = await currentUser.getIdToken();
                    const response = await axios.get('http://127.0.0.1:5000/medication-interactions', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        withCredentials: true,
                    });
                    setInteractions(response.data.interactions);
                    setLoading(false);
                } catch (error) {
                    setError(error.message);
                    showToast(error.message, 'danger');
                    setLoading(false);
                }
            } else {
                setUser(null);
                setInteractions([]);
                setLoading(false);
            }
        });

        return () => unsubscribe();
    }, []);

    // Get badge color based on severity
    const getSeverityBadgeVariant = (severity) => {
        switch (severity.toLowerCase()) {
            case 'severe':
                return 'danger';
            case 'moderate':
                return 'warning';
            case 'mild':
                return 'info';
            default:
                return 'secondary';
        }
    };

    return (
        <>
            <DashboardLayout />
            <Container className="mt-4">
                <h1 className="mb-4">Medication Interactions</h1>

                {/* Toast notification */}
                {toast.show && (
                    <Alert variant={toast.variant} onClose={() => setToast({ show: false })} dismissible>
                        {toast.message}
                    </Alert>
                )}

                {/* Loading state */}
                {loading && <p>Loading medication interactions...</p>}

                {/* Error state */}
                {error && (
                    <Alert variant="danger">
                        Error loading medication interactions: {error}
                    </Alert>
                )}

                {/* No interactions found */}
                {!loading && !error && interactions.length === 0 && (
                    <Alert variant="success">
                        No medication interactions detected. Your current medications appear to be safe to take together.
                    </Alert>
                )}

                {/* Interactions list */}
                {!loading && !error && interactions.length > 0 && (
                    <>
                        <Alert variant="warning" className="d-flex align-items-center">
                            <ExclamationTriangleFill className="me-2" size={24} />
                            <div>
                                <strong>Important:</strong> {interactions.length} potential medication interaction{interactions.length !== 1 ? 's' : ''} detected. 
                                Please consult with your healthcare provider before making any changes to your medication regimen.
                            </div>
                        </Alert>

                        <Button 
                            variant="primary" 
                            className="mb-3"
                            onClick={fetchInteractions}
                        >
                            Refresh Interactions
                        </Button>

                        {interactions.map((interaction, index) => (
                            <Card key={index} className="mb-3 border-left-4 border-warning">
                                <Card.Header className="d-flex justify-content-between align-items-center">
                                    <h5 className="mb-0">
                                        {interaction.medication1.name} + {interaction.medication2.name}
                                    </h5>
                                    <Badge bg={getSeverityBadgeVariant(interaction.severity)}>
                                        {interaction.severity.charAt(0).toUpperCase() + interaction.severity.slice(1)} Interaction
                                    </Badge>
                                </Card.Header>
                                <Card.Body>
                                    <Row>
                                        <Col md={6}>
                                            <h6>{interaction.medication1.name}</h6>
                                            <p className="text-muted">Dosage: {interaction.medication1.dosage}</p>
                                        </Col>
                                        <Col md={6}>
                                            <h6>{interaction.medication2.name}</h6>
                                            <p className="text-muted">Dosage: {interaction.medication2.dosage}</p>
                                        </Col>
                                    </Row>
                                    <hr />
                                    <h6>Interaction:</h6>
                                    <p>{interaction.description}</p>
                                </Card.Body>
                                <Card.Footer className="text-muted">
                                    <small>Consult your doctor before making any changes to your medication regimen.</small>
                                </Card.Footer>
                            </Card>
                        ))}
                    </>
                )}
            </Container>
        </>
    );
};

export default MedicationManagement;