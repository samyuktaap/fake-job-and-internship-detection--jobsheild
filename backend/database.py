"""
backend/database.py
SQLite persistence layer for JobShield.
Stores scan history, model risk predictions, and user feedback reports.
"""

import os
import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "jobshield.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row access."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite tables for scan history and feedback tracking."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL,
        company TEXT DEFAULT '',
        location TEXT DEFAULT '',
        risk_score_pct REAL NOT NULL,
        calibrated_probability REAL NOT NULL,
        prediction_label TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        badge_color TEXT NOT NULL,
        red_flags_json TEXT DEFAULT '[]',
        green_flags_json TEXT DEFAULT '[]'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        scan_id INTEGER,
        job_title TEXT DEFAULT '',
        reported_label TEXT NOT NULL,
        comments TEXT DEFAULT '',
        user_email TEXT DEFAULT '',
        FOREIGN KEY (scan_id) REFERENCES scan_history (id) ON DELETE SET NULL
    )
    """)

    conn.commit()
    conn.close()


def save_scan(data: Dict[str, Any]) -> int:
    """Persists an analyzed job posting scan and returns the new row id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    created_at = datetime.now(timezone.utc).isoformat()
    red_flags_str = json.dumps(data.get("red_flags", []))
    green_flags_str = json.dumps(data.get("green_flags", []))

    cursor.execute("""
    INSERT INTO scan_history (
        created_at, title, company, location,
        risk_score_pct, calibrated_probability,
        prediction_label, risk_level, badge_color,
        red_flags_json, green_flags_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        data.get("title", "Untitled Job"),
        data.get("company", ""),
        data.get("location", ""),
        float(data.get("risk_score_pct", 0.0)),
        float(data.get("calibrated_probability", 0.0)),
        data.get("prediction_label", "Unknown"),
        data.get("risk_level", "Unknown"),
        data.get("badge_color", "gray"),
        red_flags_str,
        green_flags_str
    ))

    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def get_recent_scans(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent scans ordered newest first."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, created_at, title, company, location,
           risk_score_pct, calibrated_probability,
           prediction_label, risk_level, badge_color,
           red_flags_json, green_flags_json
    FROM scan_history
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "title": r["title"],
            "company": r["company"],
            "location": r["location"],
            "risk_score_pct": r["risk_score_pct"],
            "calibrated_probability": r["calibrated_probability"],
            "prediction_label": r["prediction_label"],
            "risk_level": r["risk_level"],
            "badge_color": r["badge_color"],
            "red_flags": json.loads(r["red_flags_json"]),
            "green_flags": json.loads(r["green_flags_json"])
        })
    return results


def get_scan_by_id(scan_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a specific scan record by id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, created_at, title, company, location,
           risk_score_pct, calibrated_probability,
           prediction_label, risk_level, badge_color,
           red_flags_json, green_flags_json
    FROM scan_history
    WHERE id = ?
    """, (scan_id,))

    r = cursor.fetchone()
    conn.close()

    if not r:
        return None

    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "title": r["title"],
        "company": r["company"],
        "location": r["location"],
        "risk_score_pct": r["risk_score_pct"],
        "calibrated_probability": r["calibrated_probability"],
        "prediction_label": r["prediction_label"],
        "risk_level": r["risk_level"],
        "badge_color": r["badge_color"],
        "red_flags": json.loads(r["red_flags_json"]),
        "green_flags": json.loads(r["green_flags_json"])
    }


def save_feedback(data: Dict[str, Any]) -> int:
    """Stores user feedback/corrections on a scan."""
    conn = get_db_connection()
    cursor = conn.cursor()

    created_at = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
    INSERT INTO user_feedback (
        created_at, scan_id, job_title, reported_label, comments, user_email
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        created_at,
        data.get("scan_id"),
        data.get("job_title", ""),
        data.get("reported_label", "Unknown"),
        data.get("comments", ""),
        data.get("user_email", "")
    ))

    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return feedback_id


def get_recent_feedback(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent user feedback entries."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, created_at, scan_id, job_title, reported_label, comments, user_email
    FROM user_feedback
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(r) for r in rows]
