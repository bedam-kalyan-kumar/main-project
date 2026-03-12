# database.py
import sqlite3
import json
from datetime import datetime
import os
import hashlib

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictor_history.db")

def init_db():
    """Initialize the database with required tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_history'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if content_hash column exists
            cursor.execute("PRAGMA table_info(prediction_history)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'content_hash' not in columns:
                print("⚠️ Updating database schema...")
                
                # Create new table with correct schema
                cursor.execute('''
                    CREATE TABLE prediction_history_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_text TEXT NOT NULL,
                        content_hash TEXT,
                        language TEXT NOT NULL,
                        model_used TEXT NOT NULL,
                        predictions TEXT NOT NULL,
                        continuations TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        user_session TEXT
                    )
                ''')
                
                # Copy data from old table
                cursor.execute('''
                    INSERT INTO prediction_history_new 
                    (id, input_text, language, model_used, predictions, continuations, timestamp, user_session)
                    SELECT id, input_text, language, model_used, predictions, continuations, timestamp, user_session
                    FROM prediction_history
                ''')
                
                # Drop old table
                cursor.execute('DROP TABLE prediction_history')
                
                # Rename new table
                cursor.execute('ALTER TABLE prediction_history_new RENAME TO prediction_history')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_history (timestamp DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON prediction_history (content_hash)')
                
                print("✅ Database schema updated successfully")
            else:
                # Ensure indexes exist
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_history (timestamp DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON prediction_history (content_hash)')
        else:
            # Create new table from scratch
            cursor.execute('''
                CREATE TABLE prediction_history (
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
            
            # Create indexes
            cursor.execute('CREATE INDEX idx_timestamp ON prediction_history (timestamp DESC)')
            cursor.execute('CREATE INDEX idx_hash ON prediction_history (content_hash)')
            
            print("✅ New database created successfully")
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        # Don't raise the exception, just log it

def add_history_entry(input_text, language, model_used, predictions, continuations, user_session="default"):
    """Add a new entry to history (prevents duplicates)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create a hash of the input to detect duplicates
        content_hash = hashlib.md5(f"{input_text}_{language}_{user_session}".encode()).hexdigest()
        
        # Check for duplicates within last hour (instead of using UNIQUE constraint)
        one_hour_ago = (datetime.now().timestamp() - 3600) * 1000
        cursor.execute('''
            SELECT id FROM prediction_history 
            WHERE content_hash = ? AND julianday(timestamp) > julianday('now', '-1 hour')
        ''', (content_hash,))
        
        if cursor.fetchone():
            conn.close()
            print(f"⚠️ Duplicate history entry skipped: '{input_text[:30]}...'")
            return None
        
        timestamp = datetime.now().isoformat()
        
        # Convert lists to JSON strings for storage
        predictions_json = json.dumps(predictions[:10]) if predictions else json.dumps([])
        continuations_json = json.dumps(continuations[:5]) if continuations else json.dumps([])
        
        cursor.execute('''
            INSERT INTO prediction_history 
            (input_text, content_hash, language, model_used, predictions, continuations, timestamp, user_session)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            input_text[:500],  # Limit input text length
            content_hash,
            language, 
            model_used, 
            predictions_json, 
            continuations_json, 
            timestamp,
            user_session
        ))
        
        conn.commit()
        history_id = cursor.lastrowid
        conn.close()
        
        print(f"✅ History saved: '{input_text[:30]}...'")
        return history_id
        
    except sqlite3.IntegrityError as e:
        conn.close()
        print(f"⚠️ Duplicate history entry skipped: '{input_text[:30]}...'")
        return None
    except Exception as e:
        print(f"Error adding history entry: {e}")
        return None

def get_history(limit=50, user_session="default"):
    """Get recent history entries"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, input_text, language, model_used, predictions, continuations, timestamp
            FROM prediction_history
            WHERE user_session = ?
            ORDER BY timestamp DESC
            LIMIT ?
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
                    "predictions": json.loads(row[4]) if row[4] else [],
                    "continuations": json.loads(row[5]) if row[5] else [],
                    "timestamp": row[6]
                })
            except json.JSONDecodeError:
                continue
        
        return history
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

def delete_history_entry(entry_id):
    """Delete a specific history entry"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM prediction_history WHERE id = ?', (entry_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        
        return deleted
    except Exception as e:
        print(f"Error deleting history entry: {e}")
        return False

def clear_all_history(user_session="default"):
    """Clear all history for a user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM prediction_history WHERE user_session = ?', (user_session,))
        conn.commit()
        count = cursor.rowcount
        conn.close()
        
        return count
    except Exception as e:
        print(f"Error clearing history: {e}")
        return 0

def cleanup_old_entries(days=30):
    """Remove history entries older than specified days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM prediction_history 
            WHERE julianday('now') - julianday(timestamp) > ?
        ''', (days,))
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        if deleted > 0:
            print(f"🧹 Cleaned up {deleted} old history entries")
        return deleted
    except Exception as e:
        print(f"Error cleaning up old entries: {e}")
        return 0

# Initialize the database when module is imported
if __name__ != "__main__":
    init_db()
    # Optional: Clean up old entries on startup
    cleanup_old_entries(days=30)