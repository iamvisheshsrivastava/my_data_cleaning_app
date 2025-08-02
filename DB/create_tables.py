import sqlite3

# Change this if you're using a different database file name or path
DB_PATH = r"c:\Users\sriva\Desktop\AICUFLow\my_data_cleaning_app\audit.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create sessions table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        start_time TEXT,
        user_id TEXT
    )
''')

# Create files table
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

# Create events table
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

print(f"✅ Tables created successfully in {DB_PATH}")
