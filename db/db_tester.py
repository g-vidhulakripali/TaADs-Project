import sqlite3

try:
    conn = sqlite3.connect('D:/Coding Projects/Git_Hub_Projects/TaADs-Project/data/db/courses.sqlite')
    print("Database opened successfully")
    conn.close()
except sqlite3.Error as e:
    print(f"SQLite error: {e}")
