import sqlite3
import hashlib

def add_user(username, password):
    # Hash the password before storing
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Connect to database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    # Insert user
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        print("User added successfully.")
    except sqlite3.IntegrityError:
        print("Error: Username already exists.")
    
    conn.close()

# Add a test user
add_user('testuser', 'testpassword')



