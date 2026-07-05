import { useState, useEffect } from "react";
import { Box, Button, Flex, Heading, Text, VStack, Spinner } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "../../Authentication/firebase";

export default function Appointments() {
    const [appointments, setAppointments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [user, setUser] = useState(null);

    const navigate = useNavigate();

    useEffect(() => {
        // Listen for authentication state changes
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (currentUser) {
                console.log(user);
                setUser(currentUser);
                try {
                    setLoading(true);
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
                    setLoading(false);
                } catch (error) {
                    setError(error.message);
                    setLoading(false);
                }
            } else {
                setUser(null);
                setAppointments([]);
                setLoading(false);
            }
        });

        // Cleanup function to unsubscribe when component unmounts
        return () => unsubscribe();
    }, []);

    const latestTwoAppointments = appointments.slice(0, 2);

    return (
        <Box p={5} borderWidth="1px" borderRadius="lg" overflow="hidden" height= "500px">
            <Box p="4" borderBottomWidth="1px">
                <Heading size="md"> Upcoming Appointments </Heading>
                <Text color="gray.600" mt="1">
                    Stay on top of your schedule
                </Text>
            </Box>
            <Box p="4">
                <VStack spacing={4} align="stretch">
                    <Heading size="lg"></Heading>
                    {loading ? (
                        <Flex justify="center" p={5}>
                            <Spinner size="xl" />
                        </Flex>
                    ) : error ? (
                        <Box p={4} bg="red.100" color="red.800" borderRadius="md">
                            <Text>Error: {error}</Text>
                        </Box>
                    ) : appointments.length === 0 ? (
                        <Box p={4} bg="gray.100" borderRadius="md">
                            <Text>No upcoming appointments found.</Text>
                        </Box>
                    ) : (
                        <VStack spacing={4} align="stretch">
                            {latestTwoAppointments.map((appointment) => (
                                <Box
                                    key={appointment.id}
                                    p={4}
                                    borderWidth="1px"
                                    borderRadius="md"
                                    shadow="sm"
                                >
                                    <Text fontWeight="large">{appointment.title}</Text>
                                    {appointment.date && <Text fontSize="sm" color="gray.500">Date: {appointment.date}</Text>}
                                    {appointment.description && <Text fontSize="sm" color="gray.500">{appointment.description}</Text>}
                                </Box>
                            ))}
                        </VStack>
                    )}
                </VStack>
            </Box>
            <Box p="4" borderTopWidth="1px">
                <Button variant="outline" width="full" onClick={() => navigate('/appointments')}>
                    View All Appointments
                </Button>
            </Box>
        </Box>
    );
}