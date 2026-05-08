from DB.log_to_db import DB_PATH, init_db

init_db()

print(f"Tables created successfully in {DB_PATH}")
