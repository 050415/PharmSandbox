import { useState } from "react"
import {
  Box,
  Button,
  Flex,
  Heading,
  Input,
  IconButton,
  Text,
  VStack,
  Wrap,
  WrapItem,
} from "@chakra-ui/react"
import { SearchIcon } from "@chakra-ui/icons"

export default function DrugSearchPanel() {
  const [drugName, setDrugName] = useState("")

  const handleSearch = (e) => {
    e.preventDefault()
    console.log("Searching for:", drugName)
  }

  return (
    <Box borderWidth="1px" borderRadius="lg" overflow="hidden">
      <Box p="4" borderBottomWidth="1px">
        <Heading size="md">Drug Interaction Check</Heading>
        <Text color="gray.600" mt="1">
          Search for potential drug interactions and alternatives
        </Text>
      </Box>
      <Box p="4">
        <form onSubmit={handleSearch}>
          <VStack spacing="4" align="stretch">
            <VStack spacing="2" align="stretch">
              <Flex gap="2">
                <Input placeholder="Enter drug name" value={drugName} onChange={(e) => setDrugName(e.target.value)} p="2" />
                <IconButton type="submit" aria-label="Search" icon={<SearchIcon />} />
              </Flex>
            </VStack>
            <Box bg="gray.50" p="4" borderRadius="md">
              <Text fontWeight="medium" mb="2">
                Quick Search
              </Text>
              <Wrap spacing="2">
                <WrapItem>
                  <Button size="sm" variant="outline" p="2" onClick={() => setDrugName("Aspirin")}>
                    Aspirin
                  </Button>
                </WrapItem>
                <WrapItem>
                  <Button size="sm" variant="outline" p="2" onClick={() => setDrugName("Lisinopril")}>
                    Lisinopril
                  </Button>
                </WrapItem>
                <WrapItem>
                  <Button size="sm" variant="outline" p="2" onClick={() => setDrugName("Metformin")}>
                    Metformin
                  </Button>
                </WrapItem>
                <WrapItem>
                  <Button size="sm" variant="outline" p="2" onClick={() => setDrugName("Atorvastatin")}>
                    Atorvastatin
                  </Button>
                </WrapItem>
              </Wrap>
            </Box>
          </VStack>
        </form>
      </Box>
      <Box p="4" borderTopWidth="1px" fontSize="sm" color="gray.600">
        Enter a drug name to check for interactions or find alternatives
      </Box>
    </Box>
  )
}

