import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('traffic_predictions.db')
cursor = conn.cursor()

# Create a table for storing predictions
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        time TEXT,
        is_holiday INTEGER,
        temperature INTEGER,
        traffic_volume REAL,
        traffic_condition TEXT
    )
''')

# Commit and close
conn.commit()
conn.close()

print("Database setup complete!")
