import sqlite3
import time
import random
import json
import os
from datetime import datetime

class StreamingIngest:
    """Simulates Kafka / Streaming ingestion block from System Design."""
    
    def __init__(self, db_path="data/streaming_events.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                item_id INTEGER,
                timestamp DATETIME,
                event_type TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def produce_event(self, user_id, item_id, event_type="click"):
        """Produce an interaction event to Simulated Kafka Log."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO interactions (user_id, item_id, timestamp, event_type)
            VALUES (?, ?, ?, ?)
        ''', (user_id, item_id, now, event_type))
        conn.commit()
        conn.close()
        # print(f"📡 Kafka Sim: Produced Event {event_type} (u:{user_id}, i:{item_id})")

    def consume_recent(self, limit=10):
        """Consume most recent events from Simulated Streaming Log."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM interactions ORDER BY id DESC LIMIT ?', (limit,))
        events = cursor.fetchall()
        conn.close()
        return events

def run_simulation(duration_seconds=60):
    """Run a continuous stream of random interactions to simulate real traffic."""
    ingest = StreamingIngest()
    print(f"📡 Simulation Started: Ingesting events for {duration_seconds}s...")
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        # Simulate active users
        u_id = random.randint(0, 943) # ML-100k user range
        i_id = random.randint(0, 1681) # ML-100k item range
        event = random.choice(["click", "view", "like"])
        
        ingest.produce_event(u_id, i_id, event)
        time.sleep(random.uniform(0.5, 2.0)) # Random interval

if __name__ == "__main__":
    run_simulation()
