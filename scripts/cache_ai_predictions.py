import argparse
import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv

from cosmic_intelligence_model import get_cosmic_intelligence_model


def _resolve_db_path() -> str:
    explicit_path = os.getenv("COSMICWATCH_DB_PATH")
    if explicit_path:
        return explicit_path
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("sqlite:///"):
        sqlite_path = database_url[len("sqlite:///") :]
        if os.path.isabs(sqlite_path):
            return sqlite_path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, sqlite_path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "space_debris.db")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    db_path = args.db_path or _resolve_db_path()

    from utils.database import init_db, migrate_database_for_ai_caching

    init_db()
    migrate_database_for_ai_caching()

    model = get_cosmic_intelligence_model()

    where = ""
    if args.only_missing:
        where = "WHERE ai_enhanced = 0 OR ai_risk_level IS NULL OR ai_confidence IS NULL"

    query = f"""
        SELECT id, altitude, velocity, inclination, size, latitude, longitude, x, y, z, risk_score
        FROM space_debris
        {where}
        LIMIT ?
    """.strip()

    updated = 0
    errors = 0

    conn = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, (args.limit,))
        rows = cur.fetchall()
        total = len(rows)
        print(f"db_path={db_path}")
        print(f"rows_selected={total}")

        for i in range(0, total, args.batch_size):
            chunk = rows[i : i + args.batch_size]
            updates = []
            for row in chunk:
                debris_data = {
                    "altitude": row["altitude"] if row["altitude"] is not None else 400,
                    "velocity": row["velocity"] if row["velocity"] is not None else 7.5,
                    "inclination": row["inclination"] if row["inclination"] is not None else 51.6,
                    "size": row["size"] if row["size"] is not None else 1.0,
                    "latitude": row["latitude"] if row["latitude"] is not None else 0.0,
                    "longitude": row["longitude"] if row["longitude"] is not None else 0.0,
                    "x": row["x"] if row["x"] is not None else 0.0,
                    "y": row["y"] if row["y"] is not None else 0.0,
                    "z": row["z"] if row["z"] is not None else 0.0,
                }
                try:
                    pred = model.predict_debris_risk(debris_data)
                    risk_level = str(pred.get("risk_level", "MEDIUM")).upper()
                    confidence = float(pred.get("confidence", 0.0))
                    if confidence != confidence:
                        confidence = 0.0
                    confidence = max(0.0, min(confidence, 1.0))
                    updates.append((risk_level, confidence, _utc_now_iso(), 1, row["id"]))
                except Exception:
                    errors += 1

            conn.executemany(
                "UPDATE space_debris SET ai_risk_level = ?, ai_confidence = ?, ai_last_predicted = ?, ai_enhanced = ? WHERE id = ?",
                updates,
            )
            conn.commit()
            updated += len(updates)
            print(f"updated={updated}/{total} errors={errors}")
    finally:
        conn.close()

    print(f"done updated={updated} errors={errors}")


if __name__ == "__main__":
    main()

