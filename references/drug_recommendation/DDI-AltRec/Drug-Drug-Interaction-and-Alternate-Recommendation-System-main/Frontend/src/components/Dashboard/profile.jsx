import React from "react";
import { Container, Row, Col, Card } from "react-bootstrap";
import DashboardLayout from "./dashboard/DashboardLayout";
import RecentInteractions from "./dashboard/RecentInteractions";
import Appointments from "./dashboard/Appointments";
import Reminders from "./dashboard/Reminders";

const CardWrapper = ({ children }) => (
  <Card className="shadow-md rounded-3">
    <Card.Body>{children}</Card.Body>
  </Card>
);

export default function MainPage() {
  return (
    <>
      <DashboardLayout />
      <Container className="py-3">
        <Row className="g-4">
          <Col xs={12} md={6}>
            <CardWrapper>
              <Appointments />
            </CardWrapper>
          </Col>
          <Col xs={12} md={6}>
            <CardWrapper>
              <Reminders />
            </CardWrapper>
          </Col>
          <Col xs={12}>
            <CardWrapper>
              <RecentInteractions />
            </CardWrapper>
          </Col>
        </Row>
      </Container>
    </>
  );
}
