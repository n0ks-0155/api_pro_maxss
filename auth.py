# В файле auth.py
from uuid import uuid4
import sqlite3


def generate_api_key(user_id: str):
    api_key = str(uuid4())
    conn = sqlite3.connect('strikegear.db')
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO users (user_id, api_key) VALUES (?, ?)",
        (user_id, api_key)
    )

    conn.commit()
    conn.close()
    return api_key


def verify_api_key(api_key: str):
    conn = sqlite3.connect('strikegear.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id FROM users WHERE api_key = ?",
        (api_key,)
    )

    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None