import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Button, Modal, Form, Alert, Nav, Tab } from 'react-bootstrap';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from '../Authentication/firebase';
import DashboardLayout from '../Dashboard/dashboard/DashboardLayout';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, getDay, isSameDay } from 'date-fns';
import { FaCalendarAlt, FaClock, FaMapMarkerAlt, FaInfoCircle, FaTimesCircle, FaCheckCircle} from "react-icons/fa";

const Appointments = () => {
  const [appointments, setAppointments] = useState([]);
  const [filteredAppointments, setFilteredAppointments] = useState([]);
  const [currentAppointment, setCurrentAppointment] = useState({
    title: '',
    date: '',
    time: '',
    location: '',
    description: ''
  });
  const [user, setUser] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [activeListTab, setActiveListTab] = useState('upcoming');
  const [activeMainTab, setActiveMainTab] = useState('list');
  const [toast, setToast] = useState({
    show: false,
    message: '',
    variant: 'success'
  });

  // Custom toast function
  const showToast = (message, variant = 'success') => {
    setToast({ show: true, message, variant });
    setTimeout(() => {
      setToast({ show: false, message: '', variant: 'success' });
    }, 3000);
  };

  useEffect(() => {
    // Listen for authentication state changes
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        try {
          // Get the Firebase ID token
          const token = await currentUser.getIdToken();

          // Fetch appointments
          const response = await axios.get('http://127.0.0.1:5000/appointments', {
            headers: {
              'Authorization': `Bearer ${token}`
            },
            withCredentials: true,
          });
          console.log("appointments data: ", response.data);
          setAppointments(response.data);
        } catch (error) {
          showToast(error.message, 'danger');
        }
      } else {
        setUser(null);
        setAppointments([]);
      }
    });

    // Cleanup subscription
    return () => unsubscribe();
  }, []);

  // Filter appointments based on active tab
  useEffect(() => {
    const now = new Date();
    const filtered = appointments.filter(appointment => {
      const appointmentDate = new Date(appointment.date);

      if (activeListTab === 'upcoming') {
        return appointmentDate >= now;
      } else if (activeListTab === 'completed') {
        return appointmentDate < now;
      }

      return true;
    });

    setFilteredAppointments(filtered);
  }, [activeListTab, appointments]);

  const handleSubmitAppointment = async () => {
    if (!user) {
      showToast('Please log in first', 'danger');
      return;
    }

    try {
      // Get the Firebase ID token
      const token = await user.getIdToken();

      // Send appointment creation request
      await axios.post('http://127.0.0.1:5000/addappointments', currentAppointment, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      // Refresh appointments
      const response = await axios.get('http://127.0.0.1:5000/appointments', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      setAppointments(response.data);

      // Reset form
      setCurrentAppointment({
        title: '',
        date: '',
        time: '',
        location: '',
        description: ''
      });

      // Switch to appointment list tab
      setActiveMainTab('list');

      showToast('Appointment Created Successfully');
    } catch (error) {
      showToast(error.message, 'danger');
    }
  };

  // Calendar related functions
  const getAppointmentsForDate = (date) => {
    const formattedDate = format(date, 'yyyy-MM-dd');
    return appointments.filter(appointment =>
      appointment.date === formattedDate
    );
  };

  const generateCalendarDays = () => {
    const monthStart = startOfMonth(selectedDate);
    const monthEnd = endOfMonth(selectedDate);
    const startDate = monthStart;
    const endDate = monthEnd;

    const dateArray = eachDayOfInterval({ start: startDate, end: endDate });

    // Get what day of the week the month starts on (0 = Sunday, 1 = Monday, etc.)
    const startDay = getDay(monthStart);

    // Add empty spots for days before the start of the month
    const blanks = [];
    for (let i = 0; i < startDay; i++) {
      blanks.push(
        <div key={`blank-${i}`} className="calendar-day empty-day"></div>
      );
    }

    // Create array for the days in the month
    const days = dateArray.map(date => {
      const appsForDay = getAppointmentsForDate(date);
      const hasAppointments = appsForDay.length > 0;
      const isToday = isSameDay(date, new Date());

      return (
        <div
          key={date.toISOString()}
          className={`calendar-day 
            ${format(date, 'M') !== format(selectedDate, 'M') ? 'calendar-day-disabled' : ''} 
            ${hasAppointments ? 'calendar-day-has-appointment' : ''} 
            ${isToday ? 'calendar-day-today' : ''}
            ${isSameDay(date, selectedDate) ? 'calendar-day-selected' : ''}
          `}
          onClick={() => {
            setSelectedDate(date);
          }}
        >
          <span className="date-number">{format(date, 'd')}</span>
          {hasAppointments && (
            <div className="appointment-indicator">
              <span className="badge bg-primary rounded-pill">{appsForDay.length}</span>
            </div>
          )}
        </div>
      );
    });

    return [...blanks, ...days];
  };

  return (
    <>
      <DashboardLayout />
      <Container className="mt-4">
        {/* Toast Notification */}
        {toast.show && (
          <Alert
            variant={toast.variant}
            onClose={() => setToast({ show: false, message: '', variant: 'success' })}
            dismissible
            className="position-fixed top-0 end-0 m-3"
            style={{ zIndex: 1050 }}
          >
            {toast.message}
          </Alert>
        )}

        <Row>
          <Col>
            <h1 className="mb-4">My Appointments</h1>

            {/* Main Tabs Navigation */}
            <Tab.Container id="appointment-tabs" activeKey={activeMainTab} onSelect={(k) => setActiveMainTab(k)}>
              <Nav variant="tabs" className="mb-3">
                <Nav.Item>
                  <Nav.Link eventKey="list">Appointments</Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="add">Add Appointment</Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="calendar">Calendar</Nav.Link>
                </Nav.Item>
              </Nav>

              <Tab.Content>
                {/* LIST TAB */}
                <Tab.Pane eventKey="list">
                  <Card className="border-0">
                    <Card.Body>
                      {/* Appointment List Sub-Tabs */}
                      <Nav variant="pills" activeKey={activeListTab} onSelect={(selectedKey) => setActiveListTab(selectedKey)} className="mb-3">
                        <Nav.Item>
                          <Nav.Link eventKey="upcoming">
                            Upcoming ({appointments.filter(a => new Date(a.date) >= new Date()).length})
                          </Nav.Link>
                        </Nav.Item>
                        <Nav.Item>
                          <Nav.Link eventKey="completed">
                            Completed ({appointments.filter(a => new Date(a.date) < new Date()).length})
                          </Nav.Link>
                        </Nav.Item>
                      </Nav>

                      {filteredAppointments.length === 0 ? (
                        <div className="text-center py-5">
                          <h5>No {activeListTab} appointments</h5>
                          <p className="text-muted">
                            {activeListTab === 'upcoming' ?
                              'Schedule your next appointment by clicking the "Add Appointment" tab.' :
                              'Your completed appointments will appear here.'}
                          </p>

                        </div>
                      ) : (
                        filteredAppointments.map((appointment) => (
                          <Card key={appointment.id} className="mb-4 border-0 rounded-lg overflow-hidden shadow-sm">
                            <Card.Body className="p-0">
                              {/* Colored header section */}
                              <div className="bg-primary bg-gradient p-3 text-white">
                                <div className="d-flex justify-content-between align-items-center">
                                  <h5 className="mb-0 fw-bold d-flex align-items-center">
                                    <FaCalendarAlt className="me-2" />
                                    {appointment.title}
                                  </h5>
                                </div>
                              </div>

                              {/* Content section with improved spacing and layout */}
                              <div className="p-3">
                                <div className="row g-3">
                                  <div className="col-md-6">
                                    <div className="d-flex align-items-center">
                                      <div className="bg-light rounded p-2 me-3">
                                        <FaCalendarAlt className="text-primary" />
                                      </div>
                                      <div>
                                        <small className="text-muted d-block">Date</small>
                                        <span className="fw-medium">{format(new Date(appointment.date), "MMMM d, yyyy")}</span>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="col-md-6">
                                    <div className="d-flex align-items-center">
                                      <div className="bg-light rounded p-2 me-3">
                                        <FaClock className="text-primary" />
                                      </div>
                                      <div>
                                        <small className="text-muted d-block">Time</small>
                                        <span className="fw-medium">{appointment.time}</span>
                                      </div>
                                    </div>
                                  </div>

                                  {appointment.location && (
                                    <div className="col-12">
                                      <div className="d-flex align-items-center">
                                        <div className="bg-light rounded p-2 me-3">
                                          <FaMapMarkerAlt className="text-primary" />
                                        </div>
                                        <div>
                                          <small className="text-muted d-block">Location</small>
                                          <span className="fw-medium">{appointment.location}</span>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>

                                {appointment.description && (
                                  <div className="mt-3 pt-3 border-top">
                                    <div className="d-flex">
                                      <div className="bg-light rounded p-2 me-3 align-self-start">
                                        <FaInfoCircle className="text-primary" />
                                      </div>
                                      <div>
                                        <small className="text-muted d-block">Description</small>
                                        <p className="mb-0">{appointment.description}</p>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </Card.Body>
                          </Card>
                        ))
                      )}
                    </Card.Body>
                  </Card>
                </Tab.Pane>

                {/* ADD APPOINTMENT TAB */}
                <Tab.Pane eventKey="add">
                  <Card className="border-0">
                    <Card.Body>
                      <h4 className="mb-4">Add New Appointment</h4>
                      <Form>
                        <Row>
                          <Col md={6}>
                            <Form.Group className="mb-3">
                              <Form.Label>Title</Form.Label>
                              <Form.Control
                                type="text"
                                value={currentAppointment.title}
                                onChange={(e) => setCurrentAppointment({
                                  ...currentAppointment,
                                  title: e.target.value
                                })}
                                placeholder="Enter appointment title"
                                required
                              />
                            </Form.Group>
                          </Col>
                          <Col md={6}>
                            <Form.Group className="mb-3">
                              <Form.Label>Location (Optional)</Form.Label>
                              <Form.Control
                                type="text"
                                value={currentAppointment.location}
                                onChange={(e) => setCurrentAppointment({
                                  ...currentAppointment,
                                  location: e.target.value
                                })}
                                placeholder="Enter location"
                              />
                            </Form.Group>
                          </Col>
                        </Row>

                        <Row>
                          <Col md={6}>
                            <Form.Group className="mb-3">
                              <Form.Label>Date</Form.Label>
                              <Form.Control
                                type="date"
                                value={currentAppointment.date}
                                onChange={(e) => setCurrentAppointment({
                                  ...currentAppointment,
                                  date: e.target.value
                                })}
                                required
                              />
                            </Form.Group>
                          </Col>
                          <Col md={6}>
                            <Form.Group className="mb-3">
                              <Form.Label>Time</Form.Label>
                              <Form.Control
                                type="time"
                                value={currentAppointment.time}
                                onChange={(e) => setCurrentAppointment({
                                  ...currentAppointment,
                                  time: e.target.value
                                })}
                                required
                              />
                            </Form.Group>
                          </Col>
                        </Row>

                        <Form.Group className="mb-3">
                          <Form.Label>Description (Optional)</Form.Label>
                          <Form.Control
                            as="textarea"
                            rows={4}
                            value={currentAppointment.description}
                            onChange={(e) => setCurrentAppointment({
                              ...currentAppointment,
                              description: e.target.value
                            })}
                            placeholder="Enter any additional details"
                          />
                        </Form.Group>

                        <div className="d-flex justify-content-end">
                          <Button
                            variant="secondary"
                            className="me-2"
                            onClick={() => {
                              setActiveMainTab('list');
                              setCurrentAppointment({
                                title: '',
                                date: '',
                                time: '',
                                location: '',
                                description: ''
                              });
                            }}
                          >
                            Cancel
                          </Button>
                          <Button
                            variant="primary"
                            onClick={handleSubmitAppointment}
                            disabled={!currentAppointment.title || !currentAppointment.date || !currentAppointment.time}
                          >
                            Save Appointment
                          </Button>
                        </div>
                      </Form>
                    </Card.Body>
                  </Card>
                </Tab.Pane>

                {/* CALENDAR TAB */}
                <Tab.Pane eventKey="calendar">
                  <Row>
                    <Col md={8}>
                      <Card className="shadow-sm">
                        <Card.Header className="bg-primary text-white d-flex justify-content-between align-items-center">
                          <Button
                            variant="light"
                            size="sm"
                            onClick={() => setSelectedDate(new Date(selectedDate.getFullYear(), selectedDate.getMonth() - 1, 1))}
                          >
                            &lt; Prev
                          </Button>
                          <h4 className="mb-0">{format(selectedDate, 'MMMM yyyy')}</h4>
                          <Button
                            variant="light"
                            size="sm"
                            onClick={() => setSelectedDate(new Date(selectedDate.getFullYear(), selectedDate.getMonth() + 1, 1))}
                          >
                            Next &gt;
                          </Button>
                        </Card.Header>
                        <Card.Body>
                          <div className="calendar-container">
                            <div className="calendar-grid">
                              {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map(day => (
                                <div key={day} className="calendar-day-header">{day}</div>
                              ))}
                              {generateCalendarDays()}
                            </div>
                          </div>
                        </Card.Body>
                        <Card.Footer className="text-muted">
                          <div className="d-flex justify-content-between align-items-center">
                            <div>
                              <span className="legend-item"><div className="legend-color calendar-day-has-appointment-indicator"></div> Has appointment</span>
                              <span className="legend-item ms-3"><div className="legend-color calendar-day-today-indicator"></div> Today</span>
                            </div>
                            <Button
                              variant="primary"
                              size="sm"
                              onClick={() => {
                                setCurrentAppointment({
                                  ...currentAppointment,
                                  date: format(selectedDate, 'yyyy-MM-dd')
                                });
                                setActiveMainTab('add');
                              }}
                            >
                              Add for {format(selectedDate, 'MMM d')}
                            </Button>
                          </div>
                        </Card.Footer>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="shadow-sm">
                        <Card.Header className="bg-secondary text-white">
                          Appointments for {format(selectedDate, 'MMMM d, yyyy')}
                        </Card.Header>
                        <Card.Body>
                          {getAppointmentsForDate(selectedDate).length > 0 ? (
                            getAppointmentsForDate(selectedDate).map((appointment, index) => (
                              <div key={index} className="appointment-item mb-3">
                                <h6>{appointment.title}</h6>
                                <p className="mb-1"><strong>Time:</strong> {appointment.time}</p>
                                {appointment.location && <p className="mb-1"><strong>Location:</strong> {appointment.location}</p>}
                                {appointment.description && <p className="mb-0"><small>{appointment.description}</small></p>}
                              </div>
                            ))
                          ) : (
                            <div className="text-center py-4">
                              <p>No appointments for this date.</p>
                              <Button
                                variant="primary"
                                size="sm"
                                onClick={() => {
                                  setCurrentAppointment({
                                    ...currentAppointment,
                                    date: format(selectedDate, 'yyyy-MM-dd')
                                  });
                                  setActiveMainTab('add');
                                }}
                              >
                                Add Appointment
                              </Button>
                            </div>
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>
              </Tab.Content>
            </Tab.Container>
          </Col>
        </Row>
      </Container>

      {/* CSS styles */}
      <style jsx>{`
        .calendar-container {
          user-select: none;
        }
        
        .calendar-grid {
          display: grid;
          grid-template-columns: repeat(7, 1fr);
          gap: 2px;
        }
        
        .calendar-day-header {
          text-align: center;
          font-weight: bold;
          padding: 10px 0;
          background-color: #f8f9fa;
          border-radius: 4px;
        }
        
        .calendar-day {
          height: 80px;
          border: 1px solid #e9ecef;
          padding: 5px;
          cursor: pointer;
          transition: all 0.2s;
          border-radius: 4px;
          position: relative;
          display: flex;
          flex-direction: column;
        }
        
        .empty-day {
          background-color: #f8f9fa;
          cursor: default;
        }
        
        .date-number {
          font-weight: bold;
        }
        
        .calendar-day:hover:not(.empty-day) {
          background-color: #f1f3f5;
          transform: scale(1.02);
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
          z-index: 1;
        }
        
        .calendar-day-disabled {
          opacity: 0.5;
          pointer-events: none;
        }
        
        .calendar-day-has-appointment {
          background-color: #e7f5ff;
          border-color: #4dabf7;
        }
        
        .calendar-day-today {
          background-color: #fff9db;
          border: 2px solid #fcc419;
          font-weight: bold;
        }
        
        .calendar-day-selected {
          background-color: #e7f5ff;
          border: 2px solid #4dabf7;
        }
        
        .appointment-indicator {
          margin-top: auto;
          text-align: center;
        }
        
        .appointment-item {
          border-bottom: 1px solid #e9ecef;
          padding-bottom: 10px;
        }
        
        .appointment-item:last-child {
          border-bottom: none;
          margin-bottom: 0;
          padding-bottom: 0;
        }
        
        .legend-item {
          display: inline-flex;
          align-items: center;
          font-size: 0.8rem;
        }
        
        .legend-color {
          width: 15px;
          height: 15px;
          border-radius: 4px;
          margin-right: 5px;
          display: inline-block;
        }
        
        .calendar-day-has-appointment-indicator {
          background-color: #e7f5ff;
          border: 1px solid #4dabf7;
        }
        
        .calendar-day-today-indicator {
          background-color: #fff9db;
          border: 1px solid #fcc419;
        }
      `}</style>
    </>
  );
};

export default Appointments;