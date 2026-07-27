import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


DB_PATH = Path(__file__).resolve().parent / "god_eye.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Initialise les tables historiques et l'index local des visages."""
    with _connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                track_id TEXT,
                score REAL,
                timestamp TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL UNIQUE,
                name TEXT,
                embedding BLOB NOT NULL,
                image_path TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_faces_timestamp
            ON faces(timestamp)
            """
        )


def save_passage(name, track_id, score):
    """Conserve la compatibilité avec l'ancien historique des passages."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO passages (name, track_id, score, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                str(track_id),
                float(score),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def save_face_event(embedding, image_path, camera_id, track_id, confidence, name=None):
    """Enregistre un visage détecté, nommé ou non, dans l'index local."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    face_id = uuid.uuid4().hex[:12].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO faces (
                face_id, name, embedding, image_path, timestamp,
                camera_id, track_id, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face_id,
                name,
                vector.tobytes(),
                str(image_path),
                timestamp,
                str(camera_id),
                str(track_id),
                float(confidence),
            ),
        )

    return face_id, timestamp


def get_all_faces():
    """Retourne les enregistrements avec leur embedding reconstruit."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT face_id, name, embedding, image_path, timestamp,
                   camera_id, track_id, confidence
            FROM faces
            ORDER BY timestamp DESC
            """
        ).fetchall()

    return [
        {
            "face_id": row[0],
            "name": row[1],
            "embedding": np.frombuffer(row[2], dtype=np.float32),
            "image_path": row[3],
            "timestamp": row[4],
            "camera_id": row[5],
            "track_id": row[6],
            "confidence": row[7],
        }
        for row in rows
    ]
