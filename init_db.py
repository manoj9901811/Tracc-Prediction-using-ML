import sqlite3

# Connect to SQLite database (creates the file if it doesn’t exist)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the users table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')

# Commit changes and close connection
conn.commit()
conn.close()

print("Database initialized and table created successfully.")
