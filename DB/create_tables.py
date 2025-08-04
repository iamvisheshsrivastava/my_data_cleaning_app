import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")

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

print(f"Tables created successfully in {DB_PATH}")
