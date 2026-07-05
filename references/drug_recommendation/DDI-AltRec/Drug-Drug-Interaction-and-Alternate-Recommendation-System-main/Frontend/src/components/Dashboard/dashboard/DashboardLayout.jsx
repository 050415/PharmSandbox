import React, { useState, useEffect } from "react";
import { Navbar, Nav, Container, Offcanvas, Button, ListGroup} from "react-bootstrap";
import { List as MenuIcon, House, BoxArrowRight, Calendar, Repeat, HeartPulse, Gear } from "react-bootstrap-icons";
import { auth, db } from "../../Authentication/firebase";
import { doc, getDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";

export default function DashboardLayout() {
    const [userDetails, setUserDetails] = useState(null);
    const [showSidebar, setShowSidebar] = useState(false);
    const navigate = useNavigate();

    const fetchUserData = async () => {
        auth.onAuthStateChanged(async (user) => {
            if (user) {
                try {
                    const docRef = doc(db, "Users", user.uid);
                    const docSnap = await getDoc(docRef);
                    if (docSnap.exists()) {
                        setUserDetails(docSnap.data());
                    }
                } catch (error) {
                    console.error("Error fetching user details:", error);
                }
            }
        });
    };

    useEffect(() => {
        fetchUserData();
    }, []);

    async function handleLogout() {
        try {
            await auth.signOut();
            navigate("/landingpage");
        } catch (error) {
            console.error("Error logging out:", error.message);
        }
    }

    const navigateToProfile = () => {
        navigate("/profile");
        setShowSidebar(false);
    };

    const SidebarMenuItem = ({ icon, text, onClick }) => (
        <ListGroup.Item
            action
            onClick={onClick}
            className="d-flex align-items-center"
        >
            {React.cloneElement(icon, { className: "me-3" })}
            {text}
        </ListGroup.Item>
    );

    return (
        <>
            <Navbar
                bg="dark"
                variant="dark"
                expand="lg"
                className="mb-3"
            >
                <Container fluid>
                    <div className="d-flex align-items-center">
                        <Button
                            variant="link"
                            onClick={() => setShowSidebar(true)}
                            className="me-3"
                            size="lg"
                        >
                            <MenuIcon color="white" />
                        </Button>
                        <Navbar.Brand href="#home">Health Companion</Navbar.Brand>
                    </div>

                    <Nav className="ms-auto align-items-center">
                        {userDetails && (
                            <div
                                className="rounded-circle bg-light text-dark me-3 d-flex align-items-center justify-content-center"
                                style={{
                                    width: '35px',
                                    height: '35px',
                                    fontWeight: 'bold',
                                    cursor: 'pointer'
                                }}
                                onClick={navigateToProfile}
                            >
                                {userDetails.name ? userDetails.name.charAt(0).toUpperCase() : 'U'}
                            </div>
                        )}
                        <Nav.Link onClick={() => navigate("/mainPage")}>
                            <House className="me-1" /> Home
                        </Nav.Link>
                        <Nav.Link onClick={handleLogout} className="text-danger">
                            <BoxArrowRight className="me-1" /> Logout
                        </Nav.Link>
                    </Nav>
                </Container>
            </Navbar>

            <Offcanvas
                show={showSidebar}
                onHide={() => setShowSidebar(false)}
                placement="start"
            >
                <Offcanvas.Header closeButton>
                    <Offcanvas.Title>
                        <div 
                            className="d-flex align-items-center" 
                            style={{ cursor: 'pointer' }}
                            onClick={navigateToProfile}
                        >
                            <div
                                className="rounded-circle bg-primary text-white me-3 d-flex align-items-center justify-content-center"
                                style={{
                                    width: '50px',
                                    height: '50px',
                                    fontWeight: 'bold'
                                }}
                            >
                                {userDetails?.name ? userDetails.name.charAt(0).toUpperCase() : 'U'}
                            </div>
                            <div>
                                <h5 className="mb-0">{userDetails?.name || "User"}</h5>
                                <small className="text-muted">{userDetails?.email || ""}</small>
                            </div>
                        </div>
                    </Offcanvas.Title>
                </Offcanvas.Header>
                <Offcanvas.Body className="d-flex flex-column">
                    <div>
                        <ListGroup variant="flush">
                            <SidebarMenuItem
                                icon={<Calendar />}
                                text="Appointments"
                                onClick={() => {
                                    navigate("/appointments");
                                    setShowSidebar(false);
                                }}
                            />
                            <SidebarMenuItem
                                icon={<Calendar />}
                                text="Medicines"
                                onClick={() => {
                                    navigate("/medicines");
                                    setShowSidebar(false);
                                }}
                            />
                            <SidebarMenuItem
                                icon={<Repeat />}
                                text="Drug Interactions"
                                onClick={() => {
                                    navigate("/interactions");
                                    setShowSidebar(false);
                                }}
                            />
                        </ListGroup>
                    </div>
                    
                    <div className="mt-auto">
                        <hr className="my-3" />
                        <ListGroup variant="flush">
                            <SidebarMenuItem
                                icon={<Gear />}
                                text="Profile"
                                onClick={navigateToProfile}
                            />
                        </ListGroup>
                    </div>
                </Offcanvas.Body>
            </Offcanvas>
        </>
    );
}