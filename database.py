import sqlite3

def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tracking_records (
            track_id INTEGER PRIMARY KEY,
            label TEXT,
            first_seen TEXT,
            last_seen TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_new_tracking_record(track_id, label, first_seen):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO tracking_records (track_id, label, first_seen, last_seen) VALUES (?, ?, ?, ?)
    ''', (track_id, label, first_seen, first_seen))
    conn.commit()
    conn.close()

def update_tracking_record(track_id, last_seen):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''
        UPDATE tracking_records SET last_seen = ? WHERE track_id = ?
    ''', (last_seen, track_id))
    conn.commit()
    conn.close()

def check_if_track_exists(track_id):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM tracking_records WHERE track_id = ?", (track_id,))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_max_tracker_id(db_path='detections.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT MAX(track_id) FROM tracking_records")
    max_id = c.fetchone()[0]
    conn.close()
    return (max_id if max_id is not None else 0) + 1