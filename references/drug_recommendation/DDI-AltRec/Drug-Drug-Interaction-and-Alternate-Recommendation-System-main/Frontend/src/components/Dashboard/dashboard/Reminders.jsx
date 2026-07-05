import { Box, Button, Flex, Heading, Icon, Text, VStack } from "@chakra-ui/react"
import { useNavigate } from "react-router-dom"
import { useState, useEffect } from "react"
import { Spinner } from "@chakra-ui/react"
import { onAuthStateChanged } from "firebase/auth"
import { auth } from "../../Authentication/firebase"
import axios from "axios"

export default function Reminders() {

    const [medicines, setMedicines] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [user, setUser] = useState(null);

    const navigate = useNavigate();

    useEffect(() => {
        // Listen for authentication state changes
        const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
            if (currentUser) {
                setUser(currentUser);
                try {
                    setLoading(true);
                    // Get the Firebase ID token
                    const token = await currentUser.getIdToken();

                    // Fetch appointments
                    const response = await axios.get('http://127.0.0.1:5000/medications', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        },
                        withCredentials: true,
                    });

                    setMedicines(response.data);
                    setLoading(false);
                } catch (error) {
                    setError(error.message);
                    setLoading(false);
                }
            } else {
                setUser(null);
                setMedicines([]);
                setLoading(false);
            }
        });

        // Cleanup function to unsubscribe when component unmounts
        return () => unsubscribe();
    }, []);
    const latestTwoMedicines = medicines.slice(0, 2);

    return (
        <Box p={5} borderWidth="1px" borderRadius="lg" overflow="hidden" height= "500px">
            <Box p="4" borderBottomWidth="1px">
                <Heading size="md"> Reminders </Heading>
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
                    ) : medicines.length === 0 ? (
                        <Box p={4} bg="gray.100" borderRadius="md">
                            <Text>No upcoming reminders found.</Text>
                        </Box>
                    ) : (
                        <VStack spacing={4} align="stretch">
                            {latestTwoMedicines.map((medicine) => (
                                <Box
                                    key={medicine.id}
                                    p={4}
                                    borderWidth="1px"
                                    borderRadius="md"
                                    shadow="sm"
                                >
                                    <Text fontWeight="large">{medicine.name}</Text>
                                    {medicine.dosage && <Text fontSize="sm" color="gray.500">Dosage: {medicine.dosage}</Text>}
                                    {medicine.times && <Text fontSize="sm" color="gray.500">Time: {medicine.times}</Text>}
                                </Box>
                            ))}
                        </VStack>
                    )}
                </VStack>
            </Box>
            <Box p="4" borderTopWidth="1px">
                <Button variant="outline" width="full" onClick={() => navigate('/medicines')}>
                    View All Medicines
                </Button>
            </Box>
        </Box>
    )
}

