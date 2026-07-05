import os
import sqlite3
import traceback
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, initialize_app, firestore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import initialize_database, get_db_connection
from interaction import analyze_medication_interactions, check_medication_interaction
from config import MEDICATION_INTERACTIONS
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import pytz
import uuid

# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyBJsZVFFqbhTCo4bZYVyI7qbRGpVBOB_fk",
    "authDomain": "major-project-5f739.firebaseapp.com",
    "projectId": "major-project-5f739",
    "storageBucket": "major-project-5f739.firebasestorage.app",
    "messagingSenderId": "264696943617",
    "appId": "1:264696943617:web:482b06a70e60efc2d05c00",
}

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate('./major-project.json')
    firebase_admin.initialize_app(cred)
except Exception as e:
    raise

# Initialize Flask app
app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app)
CORS(app, supports_credentials=True, origins="*")

# Initialize database on startup
initialize_database()

# Middleware to verify Firebase token
def verify_firebase_token(request):
    id_token = request.headers.get('Authorization')
    if not id_token:
        print("No Authorization token found")
        return None
    try:
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(id_token.split('Bearer ')[1])
        return decoded_token['uid']
    except Exception as e:
        print(f"Token verification error: {e}")
        print(traceback.format_exc())
        return None
   
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import pytz
import uuid
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_notification(recipient_email, appointment_data):
    # Email settings
    sender_email = "singhabhayjit07@gmail.com"  
    password = ""  
    
    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = f"Appointment: {appointment_data['title']}"
    
    # Email body
    body = f"""
    Dear User,
    
    Your appointment has been scheduled:
    
    Title: {appointment_data['title']}
    Date: {appointment_data['date']}
    Time: {appointment_data['time']}
    Location: {appointment_data.get('location', 'N/A')}
    Description: {appointment_data.get('description', 'N/A')}
    
    You'll find a calendar invitation attached that you can add to your calendar.
    
    Thank you for using our service.
    """
    
    message.attach(MIMEText(body, "plain"))
    
    # Create the calendar event
    cal = Calendar()
    cal.add('prodid', '-//My Medical App//mxm.dk//')
    cal.add('version', '2.0')
    
    event = Event()
    event.add('summary', appointment_data['title'])
    
    # Parse date and time
    date_str = appointment_data['date']
    time_str = appointment_data['time']
    
    # Assuming date format is YYYY-MM-DD and time format is HH:MM
    try:
        start_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        start_datetime = pytz.timezone('UTC').localize(start_datetime)  # Adjust for your timezone
        
        end_datetime = start_datetime + timedelta(hours=1)  # Default duration 1 hour
        
        event.add('dtstart', start_datetime)
        event.add('dtend', end_datetime)
        event.add('uid', str(uuid.uuid4()))
        event.add('dtstamp', datetime.now(pytz.utc))
        
        if appointment_data.get('location'):
            event.add('location', appointment_data['location'])
        if appointment_data.get('description'):
            event.add('description', appointment_data['description'])
        
        cal.add_component(event)  # Add event to calendar
        
        ics_attachment = MIMEText(cal.to_ical().decode('utf-8'), 'calendar')
        ics_attachment.add_header('Content-Disposition', 'attachment', filename='appointment.ics')
        message.attach(ics_attachment)
        
    except Exception as e:
        print(f"Error creating calendar event: {e}")
        return False
    
    # Send email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:  # Using 'with' for automatic resource cleanup
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False



@app.route('/appointments', methods=['GET'])
def get_appointments():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Attempt to get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute('SELECT * FROM appointments WHERE user_id = ?', (user_id,))
        appointments = cursor.fetchall()
        
        # Convert to list of dictionaries
        appointments_list = []
        for appointment in appointments:
            appointments_list.append(dict(zip([column[0] for column in cursor.description], appointment)))
        
        return jsonify(appointments_list), 200
    
    except sqlite3.Error as e:
        print(f"SQLite Error in get_appointments: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Database error", "details": str(e)}), 500
    
    except Exception as e:
        print(f"Unexpected error in get_appointments: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    finally:
        # Ensure connection is closed
        if 'conn' in locals():
            conn.close()

# Appointments creation route with detailed error handling
@app.route('/addappointments', methods=['POST'])
def create_appointment():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Get request data
        data = request.json
        print("Received appointment data:", data)
        
        # Validate required fields
        required_fields = ['title', 'date', 'time']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Attempt to get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert appointment
        cursor.execute('''
            INSERT INTO appointments 
            (user_id, title, date, time, location, description) 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            data['title'], 
            data['date'], 
            data['time'], 
            data.get('location'), 
            data.get('description')
        ))
        
        # Commit and get last row id
        conn.commit()
        appointment_id = cursor.lastrowid

        # Get user email from Firestore
        db = firestore.client()

        user = auth.get_user(user_id)
        print("User data:", user)
        user_email = user.email
        print("User email:", user_email)
        
        # Send email notification
        if user_email:
            send_email_notification(user_email, data)
        
        # Return response
        return jsonify({"id": appointment_id, "email_sent": bool(user_email)}), 201
        
        # return jsonify({"id": appointment_id}), 201
    
    except sqlite3.Error as e:
        print(f"SQLite Error in create_appointment: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Database error", "details": str(e)}), 500
    
    except Exception as e:
        print(f"Unexpected error in create_appointment: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    finally:
        # Ensure connection is closed
        if 'conn' in locals():
            conn.close()





# New endpoint to check medication interactions
@app.route('/medication-interactions', methods=['GET'])
def get_medication_interactions():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Analyze medication interactions
        interactions = analyze_medication_interactions(user_id)
        
        return jsonify({
            'interactions': interactions,
            'count': len(interactions)
        }), 200
    
    except Exception as e:
        print(f"Error in get_medication_interactions: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Endpoint to check interaction between two specific medications
@app.route('/check-interaction', methods=['POST'])
def check_specific_interaction():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Get request data
        data = request.json
        if 'medication1' not in data or 'medication2' not in data:
            return jsonify({"error": "Missing medication names"}), 400
        
        # Check for interaction
        interaction = check_medication_interaction(data['medication1'], data['medication2'])
        
        if interaction:
            return jsonify({
                'hasInteraction': True,
                'severity': interaction['severity'],
                'description': interaction['description']
            }), 200
        else:
            return jsonify({
                'hasInteraction': False
            }), 200
    
    except Exception as e:
        print(f"Error in check_specific_interaction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Medications route with detailed error handling
@app.route('/medications', methods=['GET'])
def get_medications():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Attempt to get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute('SELECT * FROM medications WHERE user_id = ?', (user_id,))
        medications = cursor.fetchall()
        
        # Convert to list of dictionaries
        medications_list = []
        for medication in medications:
            med_dict = dict(zip([column[0] for column in cursor.description], medication))
            
            # Parse the times JSON string back into an array
            if 'times' in med_dict and med_dict['times']:
                try:
                    med_dict['times'] = json.loads(med_dict['times'])
                except:
                    med_dict['times'] = []
            
            medications_list.append(med_dict)
        
        return jsonify(medications_list), 200
    
    except sqlite3.Error as e:
        print(f"SQLite Error in get_medications: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Database error", "details": str(e)}), 500
    
    except Exception as e:
        print(f"Unexpected error in get_medications: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    finally:
        # Ensure connection is closed
        if 'conn' in locals():
            conn.close()

# Medications creation route with detailed error handling
@app.route('/addmedication', methods=['POST'])
def create_medication():
    try:
        # Verify user token first
        user_id = verify_firebase_token(request)
        if not user_id:
            print("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Get request data
        data = request.json
        print("Received medication data:", data)
        
        # Validate required fields
        required_fields = ['name', 'dosage', 'times']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Convert times array to JSON string for storage
        times_json = json.dumps(data['times'])
        
        # Attempt to get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert medication with all the new fields
        cursor.execute('''
            INSERT INTO medications 
            (user_id, name, dosage, times, medical_condition, start_date, end_date, notes) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            data['name'], 
            data['dosage'], 
            times_json,
            data.get('medicalCondition'),
            data.get('startDate'),
            data.get('endDate'),
            data.get('notes')
        ))
        
        # Commit and get last row id
        conn.commit()
        medication_id = cursor.lastrowid
        
        # After adding a new medication, check for new interactions
        interactions = analyze_medication_interactions(user_id)
        
        return jsonify({
            "id": medication_id,
            "newInteractionsCount": len(interactions)
        }), 201
    
    except sqlite3.Error as e:
        print(f"SQLite Error in create_medication: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Database error", "details": str(e)}), 500
    
    except Exception as e:
        print(f"Unexpected error in create_medication: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    finally:
        # Ensure connection is closed
        if 'conn' in locals():
            conn.close()

@app.route('/', methods=['GET'])
def home():
    return "hello world"

if __name__ == '__main__':
    app.run(debug=True)