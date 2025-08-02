import sqlite3
from datetime import datetime

DB_PATH = r"c:\Users\sriva\Desktop\AICUFLow\my_data_cleaning_app\audit.db"

def log_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO sessions (session_id, start_time)
        VALUES (?, ?)
    ''', (session_id, datetime.utcnow()))
    conn.commit()
    conn.close()

def log_file(session_id, filename, saved_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO files (session_id, filename, saved_path, upload_time)
        VALUES (?, ?, ?, ?)
    ''', (session_id, filename, saved_path, datetime.utcnow()))
    conn.commit()
    conn.close()

def log_event(session_id, event_type, event_detail=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (session_id, event_type, event_detail, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (session_id, event_type, event_detail, datetime.utcnow()))
    conn.commit()
    conn.close()
