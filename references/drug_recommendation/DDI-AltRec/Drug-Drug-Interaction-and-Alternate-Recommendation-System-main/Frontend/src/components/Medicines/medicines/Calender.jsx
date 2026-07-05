import React, { useState } from 'react';
import { Card, Button, Row, Col, Modal, Badge } from 'react-bootstrap';
import { format, startOfMonth, endOfMonth, eachDayOfInterval, getDay, addDays, isSameDay } from 'date-fns';
import './Calendar.css'; // Import your CSS file for styling
import { parseISO } from 'date-fns';

const CalendarComponent = ({ selectedDate, setSelectedDate, medications }) => {
  const [showModal, setShowModal] = useState(false);
  const [modalDate, setModalDate] = useState(null);

  // Get medications for a specific date
  const getMedicationsForDate = (date) => {
    return medications.filter((med) => {
      const medStartDate = parseISO(med.start_date); // Parse start date
      const medEndDate = med.end_date ? parseISO(med.end_date) : null; // Parse end date if it exists
      return (
        medStartDate <= date &&
        (!medEndDate || medEndDate >= date)
      );
    });
  };

  // Generate month days with proper offset for first day of month
  const generateCalendarDays = () => {
    const monthStart = startOfMonth(selectedDate);
    const monthEnd = endOfMonth(selectedDate);
    const dateArray = eachDayOfInterval({ start: monthStart, end: monthEnd });

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
    const days = dateArray.map((date) => {
      const medsForDay = getMedicationsForDate(date); // Get medications for the current date
      const hasMedications = medsForDay.length > 0;
      const isToday = isSameDay(date, new Date());

      return (
        <div
          key={date.toISOString()}
          className={`calendar-day 
                    ${format(date, 'M') !== format(selectedDate, 'M') ? 'calendar-day-disabled' : ''} 
                    ${hasMedications ? 'calendar-day-has-medication' : ''} 
                    ${isToday ? 'calendar-day-today' : ''}
                    ${isSameDay(date, selectedDate) ? 'calendar-day-selected' : ''}
                `}
          onClick={() => {
            setSelectedDate(date);
            if (hasMedications) {
              setModalDate(date);
              setShowModal(true);
            }
          }}
        >
          <span className="date-number">{format(date, 'd')}</span>
          {hasMedications && (
            <div className="medication-indicator">
              <Badge pill bg="primary">{medsForDay.length}</Badge>
            </div>
          )}
        </div>
      );
    });

    return [...blanks, ...days];
  };

  return (
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
                <span className="legend-item"><div className="legend-color calendar-day-has-medication-indicator"></div> Has medication</span>
                <span className="legend-item ms-3"><div className="legend-color calendar-day-today-indicator"></div> Today</span>
              </div>
              <div>
                <small>Click on a date with medication to view details</small>
              </div>
            </div>
          </Card.Footer>
        </Card>
      </Col>
      <Col md={4}>
        <Card className="shadow-sm">
          <Card.Header className="bg-secondary text-white">
            Medication Summary
          </Card.Header>
          <Card.Body>
            <h5>{format(selectedDate, 'EEEE, MMMM d, yyyy')}</h5>
            {getMedicationsForDate(selectedDate).length > 0 ? (
              getMedicationsForDate(selectedDate).map((med, index) => (
                <div key={index} className="medication-item">
                  <h6>{med.name}</h6>
                  <p className="mb-1"><strong>Dosage:</strong> {med.dosage}</p>
                  {med.timing && <p className="mb-1"><strong>Time:</strong> {med.timing}</p>}
                  {med.instructions && <p className="mb-0"><small>{med.instructions}</small></p>}
                </div>
              ))
            ) : (
              <p>No medications scheduled for this date.</p>
            )}
          </Card.Body>
        </Card>
      </Col>

      {/* Medication Modal */}
      <Modal show={showModal} onHide={() => setShowModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>
            Medications for {modalDate ? format(modalDate, 'MMMM d, yyyy') : ''}
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {modalDate && getMedicationsForDate(modalDate).map((med, index) => (
            <Card key={index} className="mb-2">
              <Card.Body>
                <Card.Title>{med.name}</Card.Title>
                <Card.Subtitle className="mb-2 text-muted">{med.dosage}</Card.Subtitle>
                {med.timing && <p><strong>Time:</strong> {med.timing}</p>}
                {med.instructions && <p>{med.instructions}</p>}
                <p>
                  <small>
                    <strong>Period:</strong> {med.startDate}
                    {med.endDate ? ` to ${med.endDate}` : ' (ongoing)'}
                  </small>
                </p>
              </Card.Body>
            </Card>
          ))}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </Row>
  );
};

export default CalendarComponent;
