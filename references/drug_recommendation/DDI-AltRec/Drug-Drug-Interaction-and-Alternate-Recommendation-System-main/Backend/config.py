# Medication interaction database
MEDICATION_INTERACTIONS = {
    ('amoxicillin', 'ibuprofen'): {
        'severity': 'mild',
        'description': 'May increase the risk of bleeding.'
    },
    ('lisinopril', 'potassium'): {
        'severity': 'severe',
        'description': 'May cause dangerous increase in potassium levels.'
    },
    ('warfarin', 'aspirin'): {
        'severity': 'severe',
        'description': 'Increased risk of bleeding.'
    },
    ('simvastatin', 'grapefruit'): {
        'severity': 'moderate',
        'description': 'May increase statin side effects.'
    },
    ('metformin', 'alcohol'): {
        'severity': 'moderate',
        'description': 'Increased risk of lactic acidosis.'
    },
    ('levothyroxine', 'calcium'): {
        'severity': 'mild',
        'description': 'May decrease absorption of levothyroxine.'
    },
    ('sertraline', 'st. john\'s wort'): {
        'severity': 'moderate',
        'description': 'May reduce the effectiveness of sertraline.'
    },
    ('omeprazole', 'clopidogrel'): {
        'severity': 'moderate',
        'description': 'May reduce the effectiveness of clopidogrel.'
    }
}