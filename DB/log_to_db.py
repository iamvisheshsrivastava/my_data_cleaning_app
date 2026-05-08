import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            start_time TEXT,
            user_id TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            saved_path TEXT,
            upload_time TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            event_type TEXT,
            event_detail TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

def log_session(session_id):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO sessions (session_id, start_time)
        VALUES (?, ?)
    ''', (session_id, datetime.utcnow()))
    conn.commit()
    conn.close()

def log_file(session_id, filename, saved_path):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO files (session_id, filename, saved_path, upload_time)
        VALUES (?, ?, ?, ?)
    ''', (session_id, filename, saved_path, datetime.utcnow()))
    conn.commit()
    conn.close()

def log_event(session_id, event_type, event_detail=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (session_id, event_type, event_detail, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (session_id, event_type, event_detail, datetime.utcnow()))
    conn.commit()
    conn.close()
