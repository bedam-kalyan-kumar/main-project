# database.py
import sqlite3
import json
from datetime import datetime
import os
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictor_history.db")

def init_db():
    """Initialize database with required table and indexes."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                language TEXT NOT NULL,
                model_used TEXT NOT NULL,
                predictions TEXT NOT NULL,
                continuations TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                user_session TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_history (timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON prediction_history (content_hash)')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB init error: {e}")

def add_history_entry(input_text, language, model_used, predictions, continuations, user_session="default"):
    """Add a new entry; prevents duplicates within last hour."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        content_hash = hashlib.md5(f"{input_text}_{language}_{user_session}".encode()).hexdigest()
        cursor.execute('SELECT id FROM prediction_history WHERE content_hash = ?', (content_hash,))
        if cursor.fetchone():
            conn.close()
            return None
        timestamp = datetime.now().isoformat()
        pred_json = json.dumps(predictions[:10])
        cont_json = json.dumps(continuations[:5])
        cursor.execute('''
            INSERT INTO prediction_history 
            (input_text, content_hash, language, model_used, predictions, continuations, timestamp, user_session)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (input_text[:500], content_hash, language, model_used, pred_json, cont_json, timestamp, user_session))
        conn.commit()
        history_id = cursor.lastrowid
        conn.close()
        return history_id
    except Exception:
        return None

def get_history(limit=50, user_session="default"):
    """Retrieve recent history entries."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, input_text, language, model_used, predictions, continuations, timestamp
            FROM prediction_history WHERE user_session = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (user_session, limit))
        rows = cursor.fetchall()
        conn.close()
        history = []
        for row in rows:
            try:
                history.append({
                    "id": row[0],
                    "input_text": row[1],
                    "language": row[2],
                    "model_used": row[3],
                    "predictions": json.loads(row[4]),
                    "continuations": json.loads(row[5]),
                    "timestamp": row[6]
                })
            except:
                continue
        return history
    except Exception:
        return []

def delete_history_entry(entry_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history WHERE id = ?', (entry_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted
    except:
        return False

def clear_all_history(user_session="default"):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prediction_history WHERE user_session = ?', (user_session,))
        conn.commit()
        count = cursor.rowcount
        conn.close()
        return count
    except:
        return 0

init_db()
