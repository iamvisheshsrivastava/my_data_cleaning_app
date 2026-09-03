import logging
import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")

logger = logging.getLogger(__name__)

# Audit logging is a side-effect, not the primary feature — a DB error here
# (e.g. SQLITE_BUSY from concurrent Streamlit sessions writing the same
# file, a permissions/disk issue on the deployed instance) must never take
# down CSV cleaning itself. Every DB call below is wrapped in try/finally so
# the connection is always closed even on error, and try/except so a
# logging failure is reported to the server logs but doesn't propagate and
# crash the caller's Streamlit rerun.


def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
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
    except sqlite3.Error:
        logger.exception("Failed to initialize audit database at %s", DB_PATH)
        raise
    finally:
        conn.close()


def log_session(session_id):
    try:
        init_db()
    except sqlite3.Error:
        return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO sessions (session_id, start_time)
            VALUES (?, ?)
        ''', (session_id, datetime.utcnow()))
        conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to log session %s", session_id)
    finally:
        conn.close()


def log_file(session_id, filename, saved_path):
    try:
        init_db()
    except sqlite3.Error:
        return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO files (session_id, filename, saved_path, upload_time)
            VALUES (?, ?, ?, ?)
        ''', (session_id, filename, saved_path, datetime.utcnow()))
        conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to log file %s for session %s", filename, session_id)
    finally:
        conn.close()


def log_event(session_id, event_type, event_detail=None):
    try:
        init_db()
    except sqlite3.Error:
        return
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (session_id, event_type, event_detail, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (session_id, event_type, event_detail, datetime.utcnow()))
        conn.commit()
    except sqlite3.Error:
        logger.exception("Failed to log event %s for session %s", event_type, session_id)
    finally:
        conn.close()
