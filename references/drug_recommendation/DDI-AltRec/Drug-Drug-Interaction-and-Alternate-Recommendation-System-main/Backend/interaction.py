import traceback
from database import get_db_connection
from config import MEDICATION_INTERACTIONS

# Date utilities for checking overlapping medications
def is_date_range_overlap(start1, end1, start2, end2):
    """Check if two date ranges overlap"""
    if not end1:
        end1 = '9999-12-31'  # Far future date
    if not end2:
        end2 = '9999-12-31'  # Far future date
    
    # Return true if there's overlap
    return start1 <= end2 and start2 <= end1

def find_overlapping_medications(user_id):
    """Find all medications with overlapping date ranges for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all medications for the user
        cursor.execute('SELECT * FROM medications WHERE user_id = ?', (user_id,))
        medications = [dict(medication) for medication in cursor.fetchall()]
        
        # Find overlapping pairs
        overlapping_pairs = []
        for i in range(len(medications)):
            for j in range(i+1, len(medications)):
                med1 = medications[i]
                med2 = medications[j]
                
                # Check if date ranges overlap
                if (med1['start_date'] and med2['start_date'] and 
                    is_date_range_overlap(
                        med1['start_date'], med1['end_date'],
                        med2['start_date'], med2['end_date']
                    )):
                    overlapping_pairs.append((med1, med2))
        
        return overlapping_pairs
    
    except Exception as e:
        print(f"Error finding overlapping medications: {e}")
        print(traceback.format_exc())
        return []
    
    finally:
        if 'conn' in locals():
            conn.close()

def check_medication_interaction(med1_name, med2_name):
    """Check if two medications have a known interaction"""
    med1_name_lower = med1_name.lower()
    med2_name_lower = med2_name.lower()
    
    if (med1_name_lower, med2_name_lower) in MEDICATION_INTERACTIONS:
        return MEDICATION_INTERACTIONS[(med1_name_lower, med2_name_lower)]
    elif (med2_name_lower, med1_name_lower) in MEDICATION_INTERACTIONS:
        return MEDICATION_INTERACTIONS[(med2_name_lower, med1_name_lower)]
    
    # No interaction found
    return None

def analyze_medication_interactions(user_id):
    """Analyze all medication interactions for a user"""
    try:
        # Find all overlapping medications
        overlapping_pairs = find_overlapping_medications(user_id)
        
        # Check each pair for interactions
        interactions = []
        for med1, med2 in overlapping_pairs:
            interaction = check_medication_interaction(med1['name'], med2['name'])
            if interaction:
                interactions.append({
                    'medication1': {
                        'id': med1['id'],
                        'name': med1['name'],
                        'dosage': med1['dosage']
                    },
                    'medication2': {
                        'id': med2['id'],
                        'name': med2['name'],
                        'dosage': med2['dosage']
                    },
                    'severity': interaction['severity'],
                    'description': interaction['description']
                })
        
        return interactions
    
    except Exception as e:
        print(f"Error analyzing medication interactions: {e}")
        print(traceback.format_exc())
        return []