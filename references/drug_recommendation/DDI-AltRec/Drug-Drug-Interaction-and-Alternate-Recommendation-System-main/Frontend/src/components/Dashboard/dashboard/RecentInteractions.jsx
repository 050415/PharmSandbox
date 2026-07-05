import { Box, Button, Flex, Heading, Icon, Text, VStack, Spinner } from "@chakra-ui/react";
import { WarningTwoIcon, CheckCircleIcon, ArrowRightIcon } from "@chakra-ui/icons";
import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { onAuthStateChanged } from "firebase/auth";
import { auth } from "../../Authentication/firebase";
import axios from "axios";

export default function RecentInteractions() {
  const [recentInteractions, setRecentInteractions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(null);

  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        try {
          setLoading(true);
          const token = await currentUser.getIdToken();
          const response = await axios.get('http://127.0.0.1:5000/medication-interactions', {
            headers: {
              'Authorization': `Bearer ${token}`,
            },
            withCredentials: true,
          });
          setRecentInteractions(response.data.interactions);
          setLoading(false);
          console.log(response.data.interactions);
        } catch (error) {
          setError(error.message);
          setLoading(false);
          console.error("Error fetching interactions:", error.message);
        }
      } else {
        setUser(null);
        setRecentInteractions([]);
        setLoading(false);
      }
    });

    return () => unsubscribe();
  }, []);

  return (
    <Box borderWidth="1px" borderRadius="lg" overflow="hidden" height="500px">
      <Box p="4" borderBottomWidth="1px">
        <Heading size="md">Recent Interactions</Heading>
        <Text color="gray.600" mt="1">
          Your most recent drug interaction checks
        </Text>
      </Box>
      <Box p="4" flex="1" overflowY="auto">
        <VStack spacing="4" align="stretch">
          {loading ? (
            <Flex justify="center" p={5}>
              <Spinner size="xl" />
            </Flex>
          ) : error ? (
            <Box p={4} bg="red.100" color="red.800" borderRadius="md">
              <Text>Error: {error}</Text>
            </Box>
          ) : recentInteractions.length === 0 ? (
            <Box p={4} bg="gray.100" borderRadius="md">
              <Text>No recent interactions found.</Text>
            </Box>
          ) : (
            recentInteractions.map((interaction, index) => (
              <Flex
                key={index}
                alignItems="center"
                justifyContent="space-between"
                p="3"
                borderWidth="1px"
                borderRadius="lg"
              >
                <Flex alignItems="center" gap="3">
                  {interaction.severity === "severe" ? (
                    <Icon as={WarningTwoIcon} boxSize="5" color="red.500" />
                  ) : interaction.severity === "moderate" ? (
                    <Icon as={WarningTwoIcon} boxSize="5" color="orange.500" />
                  ) : (
                    <Icon as={CheckCircleIcon} boxSize="5" color="green.500" />
                  )}
                  <Box>
                    <Text fontWeight="medium">
                      {interaction.medication1.name} + {interaction.medication2.name}
                    </Text>
                    <Text fontSize="sm" color="gray.500">
                      {interaction.date}
                    </Text>
                  </Box>
                </Flex>
                <Button variant="ghost" size="sm" rightIcon={<ArrowRightIcon />} />
              </Flex>
            ))
          )}
        </VStack>
      </Box>
      <Box p="4" borderTopWidth="1px">
        <Button variant="outline" width="full" onClick={() => navigate('/interactions')}>
          View All Interactions
        </Button>
      </Box>
    </Box>
  );
}

