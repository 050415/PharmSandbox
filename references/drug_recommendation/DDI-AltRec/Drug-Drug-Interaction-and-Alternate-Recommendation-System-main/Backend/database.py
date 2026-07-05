import os
import sqlite3
import traceback
import json

def initialize_database(db_path='./Database/appointments.db'):
    """
    Comprehensive database initialization function
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create a new database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create appointments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            location TEXT,
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create updated medications table with times instead of frequency
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            times TEXT NOT NULL,
            medical_condition TEXT,
            start_date TEXT,
            end_date TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create medication interactions table to store detected interactions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS medication_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            medication1_id INTEGER NOT NULL,
            medication2_id INTEGER NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            FOREIGN KEY (medication1_id) REFERENCES medications (id),
            FOREIGN KEY (medication2_id) REFERENCES medications (id)
        )
        ''')
        
        # Create indexes for faster user-specific queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_appointments_user_id 
        ON appointments(user_id)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_medications_user_id 
        ON medications(user_id)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_medication_interactions_user_id 
        ON medication_interactions(user_id)
        ''')
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Database initialized successfully at {db_path}")
        return True
    
    except Exception as e:
        print(f"Database initialization error: {e}")
        print(traceback.format_exc())
        return False

def get_db_connection(db_path='./Database/appointments.db'):
    """
    Robust database connection function
    """
    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Attempt to connect to the database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    except Exception as e:
        print(f"Database connection error: {e}")
        print(traceback.format_exc())
        raise