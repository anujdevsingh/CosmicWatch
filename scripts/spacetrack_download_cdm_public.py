import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests
from dotenv import load_dotenv


def _env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def spacetrack_login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    url = f"{base_url.rstrip('/')}/ajaxauth/login"
    response = session.post(url, data={"identity": username, "password": password}, timeout=30)
    if response.status_code != 200:
        raise SystemExit(f"Space-Track login failed: HTTP {response.status_code}")
    if "Login failed" in response.text or "invalid" in response.text.lower():
        raise SystemExit("Space-Track login failed: invalid credentials or insufficient access")


def _build_query_url(base_url: str, parts: list[str]) -> str:
    base = base_url.rstrip("/")
    encoded = "/".join(quote(p, safe=">=<,.*()_:-") for p in parts)
    return f"{base}/{encoded}"


def _request_json(session: requests.Session, url: str, timeout: int, max_retries: int) -> list[dict]:
    attempt = 0
    backoff = 2.0
    while True:
        attempt += 1
        try:
            response = session.get(url, timeout=timeout)
        except Exception as e:
            if attempt > max_retries:
                raise SystemExit(f"CDM download failed: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 60.0)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as e:
                raise SystemExit(f"CDM download returned non-JSON response: {e}")
            if not isinstance(data, list):
                raise SystemExit(f"Unexpected response type: {type(data)}")
            return data

        if response.status_code in {429, 500, 502, 503, 504} and attempt <= max_retries:
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 60.0)
            continue

        preview = (response.text or "")[:500].replace("\n", " ").strip()
        raise SystemExit(f"CDM download failed: HTTP {response.status_code}{(': ' + preview) if preview else ''}")


def download_cdm_public_range(
    session: requests.Session,
    base_url: str,
    created_start: datetime,
    created_end: datetime,
    limit: int | None,
    orderby: str | None,
    timeout: int,
    max_retries: int,
) -> list[dict]:
    start_txt = created_start.strftime("%Y-%m-%d")
    end_txt = created_end.strftime("%Y-%m-%d")

    query_parts = [
        "basicspacedata",
        "query",
        "class",
        "cdm_public",
        "CREATED",
        f">{start_txt}",
        "CREATED",
        f"<{end_txt}",
    ]
    if orderby:
        query_parts.extend(["orderby", orderby])
    if limit is not None:
        query_parts.extend(["limit", str(limit)])
    query_parts.extend(["format", "json"])

    url = _build_query_url(base_url, query_parts)
    return _request_json(session, url, timeout=timeout, max_retries=max_retries)


def _dedupe_by_id(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        cdm_id = str(row.get("CDM_ID") or "").strip()
        if not cdm_id:
            key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        else:
            key = cdm_id
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _parse_yyyy_mm_dd(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _scan_local_summary(out_dir: str) -> dict:
    paths = []
    try:
        for name in os.listdir(out_dir):
            if name.endswith(".json") and name.startswith("cdm_public_"):
                paths.append(os.path.join(out_dir, name))
    except Exception:
        paths = []

    total_files = 0
    total_rows = 0
    unique_ids: set[str] = set()
    created_min: str | None = None
    created_max: str | None = None

    for p in paths:
        total_files += 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        total_rows += len(rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("CDM_ID") or "").strip()
            if cid:
                unique_ids.add(cid)
            created = str(row.get("CREATED") or "").strip()
            if created:
                created_min = created if created_min is None else min(created_min, created)
                created_max = created if created_max is None else max(created_max, created)

    return {
        "out_dir": out_dir,
        "files": total_files,
        "rows_total": total_rows,
        "unique_cdm_ids": len(unique_ids),
        "created_min": created_min,
        "created_max": created_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/cdm_public")
    parser.add_argument("--created-since-days", type=float, default=None)
    parser.add_argument("--created-start", default=None)
    parser.add_argument("--created-end", default=None)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--orderby", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--write-latest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--backfill-until-empty", action="store_true")
    parser.add_argument("--backfill-max-years", type=int, default=15)
    args = parser.parse_args()

    load_dotenv()
    base_url = (os.getenv("SPACETRACK_BASE_URL") or "https://www.space-track.org").strip()
    username = _env("SPACETRACK_USERNAME")
    password = _env("SPACETRACK_PASSWORD")

    _ensure_dir(args.out_dir)

    if args.summary:
        print(json.dumps(_scan_local_summary(args.out_dir), indent=2))
        return

    session = requests.Session()
    spacetrack_login(session, base_url, username, password)

    now = datetime.now(timezone.utc)
    if args.created_start:
        created_start = _parse_yyyy_mm_dd(args.created_start)
    elif args.created_since_days is not None:
        created_start = now - timedelta(days=float(args.created_since_days))
    else:
        created_start = now - timedelta(days=1.0)

    if args.created_end:
        created_end = _parse_yyyy_mm_dd(args.created_end)
    else:
        created_end = now

    if created_end <= created_start:
        raise SystemExit("created_end must be after created_start")

    chunk_days = max(1, int(args.chunk_days))
    ranges: list[tuple[datetime, datetime]] = []
    if args.backfill_until_empty:
        max_years = max(1, int(args.backfill_max_years))
        cursor_end = created_end
        for _ in range(int((365 * max_years) / chunk_days) + 1):
            cursor_start = cursor_end - timedelta(days=chunk_days)
            if cursor_start >= cursor_end:
                break
            ranges.append((cursor_start, cursor_end))
            cursor_end = cursor_start
            if cursor_end <= (now - timedelta(days=365 * max_years)):
                break
    else:
        cursor = created_start
        while cursor < created_end:
            chunk_end = min(created_end, cursor + timedelta(days=chunk_days))
            ranges.append((cursor, chunk_end))
            cursor = chunk_end

    if args.dry_run:
        planned = []
        for a, b in ranges[:50]:
            planned.append({"created_start": a.strftime("%Y-%m-%d"), "created_end": b.strftime("%Y-%m-%d")})
        if len(ranges) > 50:
            planned.append({"note": f"... {len(ranges)-50} more ranges"})
        print(json.dumps({"out_dir": args.out_dir, "ranges": planned, "orderby": args.orderby, "limit": args.limit}, indent=2))
        return

    total_collected: list[dict] = []
    empty_streak = 0
    for r_start, r_end in ranges:
        rows = download_cdm_public_range(
            session=session,
            base_url=base_url,
            created_start=r_start,
            created_end=r_end,
            limit=args.limit,
            orderby=args.orderby,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        if rows:
            empty_streak = 0
        else:
            empty_streak += 1

        stamp = f"{r_start.strftime('%Y%m%d')}_{r_end.strftime('%Y%m%d')}"
        chunk_path = os.path.join(args.out_dir, f"cdm_public_chunk_{stamp}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        total_collected.extend(rows)
        if args.max_records is not None and len(total_collected) >= int(args.max_records):
            total_collected = total_collected[: int(args.max_records)]
            break
        if args.backfill_until_empty and empty_streak >= 8:
            break

    if not args.no_dedupe:
        total_collected = _dedupe_by_id(total_collected)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"cdm_public_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(total_collected, f, indent=2)

    if args.write_latest:
        latest_path = os.path.join(args.out_dir, "latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(total_collected, f, indent=2)

    print(f"Downloaded {len(total_collected)} records -> {out_path}")
    print(json.dumps(_scan_local_summary(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
