"""Persistent, facility-scoped configuration and operational records for VoltFlow: Intelligent EV Charging & Scheduling Platform."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "gridcharge.sqlite3"
_TERMINAL_REQUEST_STATUSES = {"completed", "cancelled", "failed", "duplicate review"}


def _new_identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "facility"


@contextmanager
def connect(path: str | Path | None = None):
    db_path = Path(path) if path is not None else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 15000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize(reference_stations: pd.DataFrame, path: str | Path | None = None) -> None:
    """Create the persistent schema and seed demo facilities once from the bundled catalog."""
    with connect(path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS facilities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '',
            timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived')),
            grid_capacity_kw REAL NOT NULL CHECK(grid_capacity_kw > 0),
            operating_start TEXT NOT NULL DEFAULT '00:00',
            operating_end TEXT NOT NULL DEFAULT '23:59',
            tariff_per_kwh REAL NOT NULL DEFAULT 0 CHECK(tariff_per_kwh >= 0),
            currency TEXT NOT NULL DEFAULT 'INR',
            scheduling_algorithm TEXT NOT NULL DEFAULT 'HYBRID' CHECK(scheduling_algorithm IN ('HYBRID','LAFS','EDF','FCFS')),
            simulation_step_min REAL NOT NULL DEFAULT 1.0 CHECK(simulation_step_min > 0),
            data_origin TEXT NOT NULL DEFAULT 'Manually configured',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stations (
            facility_id TEXT NOT NULL,
            id TEXT NOT NULL,
            name TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '',
            power_cap_kw REAL NOT NULL CHECK(power_cap_kw > 0),
            status TEXT NOT NULL DEFAULT 'available' CHECK(status IN ('available','maintenance','offline')),
            PRIMARY KEY (facility_id, id),
            FOREIGN KEY (facility_id) REFERENCES facilities(id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS chargers (
            facility_id TEXT NOT NULL,
            station_id TEXT NOT NULL,
            id TEXT NOT NULL,
            connector_type TEXT NOT NULL,
            power_type TEXT NOT NULL CHECK(power_type IN ('AC','DC')),
            rated_power_kw REAL NOT NULL CHECK(rated_power_kw > 0),
            status TEXT NOT NULL DEFAULT 'available' CHECK(status IN ('available','maintenance','offline')),
            PRIMARY KEY (facility_id, id),
            FOREIGN KEY (facility_id, station_id) REFERENCES stations(facility_id, id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS requests (
            facility_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            request_id TEXT,
            vehicle_reg_no TEXT NOT NULL DEFAULT '',
            station_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Queued',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (facility_id, session_id),
            FOREIGN KEY (facility_id) REFERENCES facilities(id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            grid_capacity_kw REAL NOT NULL,
            epoch_anchor TEXT NOT NULL,
            dispatch_key TEXT,
            results_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (facility_id) REFERENCES facilities(id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            facility_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (facility_id) REFERENCES facilities(id) ON UPDATE CASCADE ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_requests_facility_status ON requests(facility_id, status);
        CREATE INDEX IF NOT EXISTS idx_schedules_facility_created ON schedules(facility_id, created_at DESC);
        """)
        request_columns = {r[1] for r in con.execute("PRAGMA table_info(requests)")}
        if "request_id" not in request_columns:
            con.execute("ALTER TABLE requests ADD COLUMN request_id TEXT")
        if "vehicle_reg_no" not in request_columns:
            con.execute("ALTER TABLE requests ADD COLUMN vehicle_reg_no TEXT NOT NULL DEFAULT ''")
        schedule_columns = {r[1] for r in con.execute("PRAGMA table_info(schedules)")}
        if "dispatch_key" not in schedule_columns:
            con.execute("ALTER TABLE schedules ADD COLUMN dispatch_key TEXT")

        # Older versions stored a request/ticket label in session_id. Give every
        # persisted request separate UUID-backed request and session identifiers,
        # and rewrite saved results with the corresponding session ID. Requests
        # and results remain intact; the old label is retained for traceability.
        migrate_ids = "request_id" not in request_columns
        rows = con.execute("SELECT facility_id,session_id,request_id,vehicle_reg_no,payload_json,status "
                           "FROM requests ORDER BY facility_id,created_at,session_id").fetchall()
        used_sessions: set[str] = set()
        used_requests: set[str] = set()
        session_map: dict[tuple[str, str], tuple[str, str]] = {}
        for row in rows:
            facility_id, old_sid = str(row[0]), str(row[1])
            try:
                payload = json.loads(row[4])
            except (TypeError, ValueError):
                payload = {}
            old_rid = str(row[2] or payload.get("request_id") or "").strip()
            sid = str(old_sid).strip()
            rid = old_rid
            if migrate_ids or not sid or sid in used_sessions or sid == rid:
                sid = _new_identifier("ses")
            if not rid or rid in used_requests or rid == sid or migrate_ids:
                rid = _new_identifier("req")
            used_sessions.add(sid)
            used_requests.add(rid)
            reg = str(row[3] or payload.get("vehicle_reg_no") or "").strip().upper()
            if migrate_ids and old_sid:
                payload.setdefault("legacy_ticket", old_sid)
            payload.update({"request_id": rid, "session_id": sid, "vehicle_reg_no": reg})
            session_map[(facility_id, old_sid)] = (sid, rid)
            if sid != old_sid or rid != old_rid or reg != str(row[3] or ""):
                con.execute("UPDATE requests SET session_id=?,request_id=?,vehicle_reg_no=?,payload_json=? "
                            "WHERE facility_id=? AND session_id=?",
                            (sid, rid, reg, json.dumps(payload, default=str), facility_id, old_sid))
            elif "request_id" not in payload:
                payload["request_id"] = rid
                con.execute("UPDATE requests SET request_id=?,vehicle_reg_no=?,payload_json=? "
                            "WHERE facility_id=? AND session_id=?",
                            (rid, reg, json.dumps(payload, default=str), facility_id, old_sid))

        # Saved schedule snapshots refer to the old session IDs. Keep the history
        # usable after the identifier migration instead of discarding old runs.
        for schedule in con.execute("SELECT id,facility_id,results_json FROM schedules").fetchall():
            try:
                results = json.loads(schedule[2])
            except (TypeError, ValueError):
                continue
            changed = False
            for result in results if isinstance(results, list) else []:
                mapped = session_map.get((str(schedule[1]), str(result.get("session_id", ""))))
                if mapped:
                    result["session_id"], result["request_id"] = mapped
                    changed = True
            if changed:
                con.execute("UPDATE schedules SET results_json=? WHERE id=?",
                            (json.dumps(results, default=str), schedule[0]))

        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_requests_session_id_global ON requests(session_id)")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_requests_request_id_global ON requests(request_id)")
        con.executescript("""
        CREATE TRIGGER IF NOT EXISTS trg_requests_unique_active_vehicle_insert
        BEFORE INSERT ON requests
        WHEN trim(NEW.vehicle_reg_no) <> ''
          AND lower(trim(NEW.status)) NOT IN ('completed','cancelled','failed','duplicate review')
          AND EXISTS (SELECT 1 FROM requests r WHERE r.facility_id=NEW.facility_id
            AND r.session_id<>NEW.session_id
            AND upper(trim(r.vehicle_reg_no))=upper(trim(NEW.vehicle_reg_no))
            AND lower(trim(r.status)) NOT IN ('completed','cancelled','failed','duplicate review'))
        BEGIN SELECT RAISE(ABORT, 'vehicle registration already has an active request'); END;
        CREATE TRIGGER IF NOT EXISTS trg_requests_unique_active_vehicle_update
        BEFORE UPDATE OF vehicle_reg_no,status ON requests
        WHEN trim(NEW.vehicle_reg_no) <> ''
          AND lower(trim(NEW.status)) NOT IN ('completed','cancelled','failed','duplicate review')
          AND (lower(trim(OLD.status)) IN ('completed','cancelled','failed','duplicate review')
               OR upper(trim(OLD.vehicle_reg_no))<>upper(trim(NEW.vehicle_reg_no)))
          AND EXISTS (SELECT 1 FROM requests r WHERE r.facility_id=NEW.facility_id
            AND r.session_id<>NEW.session_id
            AND upper(trim(r.vehicle_reg_no))=upper(trim(NEW.vehicle_reg_no))
            AND lower(trim(r.status)) NOT IN ('completed','cancelled','failed','duplicate review'))
        BEGIN SELECT RAISE(ABORT, 'vehicle registration already has an active request'); END;
        CREATE TRIGGER IF NOT EXISTS trg_requests_distinct_ids_insert
        BEFORE INSERT ON requests
        WHEN NEW.request_id=NEW.session_id
          OR EXISTS (SELECT 1 FROM requests r WHERE r.session_id=NEW.request_id OR r.request_id=NEW.session_id)
        BEGIN SELECT RAISE(ABORT, 'request_id and session_id must be distinct'); END;
        CREATE TRIGGER IF NOT EXISTS trg_requests_distinct_ids_update
        BEFORE UPDATE OF request_id,session_id ON requests
        WHEN NEW.request_id=NEW.session_id
          OR EXISTS (SELECT 1 FROM requests r WHERE r.session_id<>OLD.session_id
            AND (r.session_id=NEW.request_id OR r.request_id=NEW.session_id))
        BEGIN SELECT RAISE(ABORT, 'request_id and session_id must be distinct'); END;
        CREATE UNIQUE INDEX IF NOT EXISTS ux_schedules_facility_dispatch_key
            ON schedules(facility_id,dispatch_key) WHERE dispatch_key IS NOT NULL;
        """)
        # Keep local databases created by earlier VoltFlow builds usable.
        facility_columns = {r[1] for r in con.execute("PRAGMA table_info(facilities)")}
        if "scheduling_algorithm" not in facility_columns:
            con.execute("ALTER TABLE facilities ADD COLUMN scheduling_algorithm TEXT NOT NULL DEFAULT 'HYBRID'")
        if "simulation_step_min" not in facility_columns:
            con.execute("ALTER TABLE facilities ADD COLUMN simulation_step_min REAL NOT NULL DEFAULT 1.0")
        # Upgrade databases created before Hybrid was an operational default. Rebuild only
        # the parent table with foreign keys temporarily disabled; child rows are preserved.
        schema_sql = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='facilities'").fetchone()[0]
        if "HYBRID" not in schema_sql:
            con.execute("PRAGMA foreign_keys=OFF")
            con.execute("""CREATE TABLE facilities_hybrid_new (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, location TEXT NOT NULL DEFAULT '',
                timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived')),
                grid_capacity_kw REAL NOT NULL CHECK(grid_capacity_kw > 0),
                operating_start TEXT NOT NULL DEFAULT '00:00', operating_end TEXT NOT NULL DEFAULT '23:59',
                tariff_per_kwh REAL NOT NULL DEFAULT 0 CHECK(tariff_per_kwh >= 0), currency TEXT NOT NULL DEFAULT 'INR',
                scheduling_algorithm TEXT NOT NULL DEFAULT 'HYBRID' CHECK(scheduling_algorithm IN ('HYBRID','LAFS','EDF','FCFS')),
                simulation_step_min REAL NOT NULL DEFAULT 1.0 CHECK(simulation_step_min > 0),
                data_origin TEXT NOT NULL DEFAULT 'Manually configured', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            cols = [r[1] for r in con.execute("PRAGMA table_info(facilities_hybrid_new)")]
            old_cols = {r[1] for r in con.execute("PRAGMA table_info(facilities)")}
            copy_cols = [c for c in cols if c in old_cols]
            con.execute(f"INSERT INTO facilities_hybrid_new ({','.join(copy_cols)}) SELECT {','.join(copy_cols)} FROM facilities")
            con.execute("DROP TABLE facilities")
            con.execute("ALTER TABLE facilities_hybrid_new RENAME TO facilities")
            con.execute("PRAGMA foreign_keys=ON")
        count = con.execute("SELECT COUNT(*) FROM facilities").fetchone()[0]
        if count == 0 and not reference_stations.empty:
            for station_id, group in reference_stations.groupby("station_id", sort=False):
                station_id = str(station_id)
                environment = str(group["environment"].iloc[0])
                cap = float(group["station_aggregate_power_cap_kw_scenario"].iloc[0])
                fid = "demo-" + slugify(station_id)
                now = _now()
                con.execute("""INSERT INTO facilities
                    (id,name,location,timezone,status,grid_capacity_kw,operating_start,operating_end,
                     tariff_per_kwh,currency,scheduling_algorithm,simulation_step_min,data_origin,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (fid, f"{environment} Demonstration", "", "Asia/Kolkata", "active", cap,
                     "00:00", "23:59", 0.0, "INR", "HYBRID", 1.0, "Demonstration dataset", now, now))
                con.execute("INSERT INTO stations VALUES(?,?,?,?,?,?)",
                            (fid, station_id, station_id, environment, cap, "available"))
                for _, row in group.iterrows():
                    connector = str(row["connector_type"])
                    ptype = "DC" if "DC" in connector.upper() or "CCS" in connector.upper() else "AC"
                    con.execute("INSERT INTO chargers VALUES(?,?,?,?,?,?,?)",
                                (fid, station_id, str(row["charger_id"]), connector, ptype,
                                 float(row["rated_power_kw"]), "available"))


def list_facilities(path: str | Path | None = None, include_archived: bool = False) -> list[dict[str, Any]]:
    where = "" if include_archived else "WHERE f.status = 'active'"
    with connect(path) as con:
        rows = con.execute(f"""SELECT f.*,
            (SELECT COUNT(*) FROM stations s WHERE s.facility_id=f.id) AS station_count,
            (SELECT COUNT(*) FROM chargers c WHERE c.facility_id=f.id) AS charger_count
            FROM facilities f {where} ORDER BY f.name COLLATE NOCASE""").fetchall()
    return [dict(r) for r in rows]


def get_facility(facility_id: str, path: str | Path | None = None) -> dict[str, Any] | None:
    with connect(path) as con:
        row = con.execute("SELECT * FROM facilities WHERE id=?", (facility_id,)).fetchone()
    return dict(row) if row else None


def get_station_frame(facility_id: str, path: str | Path | None = None) -> pd.DataFrame:
    with connect(path) as con:
        rows = con.execute("""SELECT s.id AS station_id, s.name AS station_name,
            s.location AS station_location, s.power_cap_kw AS station_aggregate_power_cap_kw_scenario,
            s.status AS station_status, f.name AS environment,
            c.id AS charger_id, c.connector_type, c.power_type, c.rated_power_kw,
            c.status AS charger_status, 0 AS power_cap_is_measured,
            'Configured locally' AS profile_notes, 'Facility configuration' AS source_basis
            FROM stations s JOIN facilities f ON f.id=s.facility_id
            JOIN chargers c ON c.facility_id=s.facility_id AND c.station_id=s.id
            WHERE s.facility_id=? AND f.status='active'
            ORDER BY s.name, c.id""", (facility_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_configuration(facility: dict[str, Any], stations: list[dict[str, Any]],
                       chargers: list[dict[str, Any]], path: str | Path | None = None) -> None:
    """Replace one facility's editable configuration atomically and record the change."""
    fid = str(facility["id"]).strip()
    name = str(facility["name"]).strip()
    grid = float(facility["grid_capacity_kw"])
    if not fid or not name or grid <= 0:
        raise ValueError("Facility ID, name, and a positive grid capacity are required.")
    try:
        ZoneInfo(str(facility.get("timezone", "Asia/Kolkata")))
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Enter a valid IANA timezone, such as Asia/Kolkata or Europe/London.") from exc
    for time_key in ("operating_start", "operating_end"):
        value = str(facility.get(time_key, "00:00"))
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError as exc:
            raise ValueError("Operating hours must use 24-hour HH:MM format.") from exc
    if float(facility.get("tariff_per_kwh", 0)) < 0:
        raise ValueError("Energy tariff cannot be negative.")
    if str(facility.get("scheduling_algorithm", "HYBRID")) not in ("HYBRID", "LAFS", "EDF", "FCFS"):
        raise ValueError("Choose a supported scheduling policy.")
    station_ids: set[str] = set()
    for s in stations:
        sid = str(s["id"]).strip()
        cap = float(s["power_cap_kw"])
        if not sid or sid in station_ids or cap <= 0:
            raise ValueError("Station IDs must be unique and station limits must be positive.")
        if str(s.get("status", "available")) not in ("available", "maintenance", "offline"):
            raise ValueError("Station status must be Available, Maintenance, or Offline.")
        station_ids.add(sid)
    charger_ids: set[str] = set()
    for c in chargers:
        cid, sid = str(c["id"]).strip(), str(c["station_id"]).strip()
        if not cid or cid in charger_ids or sid not in station_ids:
            raise ValueError("Charger IDs must be unique and each charger must reference a configured station.")
        if not str(c["connector_type"]).strip() or float(c["rated_power_kw"]) <= 0:
            raise ValueError("Each charger requires a connector type and positive rated power.")
        if str(c.get("status", "available")) not in ("available", "maintenance", "offline"):
            raise ValueError("Charger status must be Available, Maintenance, or Offline.")
        charger_ids.add(cid)
    if not station_ids or not charger_ids:
        raise ValueError("Configure at least one station and one charger for an active facility.")
    if {str(c["station_id"]).strip() for c in chargers} != station_ids:
        raise ValueError("Every station must have at least one charger endpoint.")

    now = _now()
    with connect(path) as con:
        existing = con.execute("SELECT id FROM facilities WHERE id=?", (fid,)).fetchone()
        vals = (name, str(facility.get("location", "")), str(facility.get("timezone", "Asia/Kolkata")),
                str(facility.get("status", "active")), grid, str(facility.get("operating_start", "00:00")),
                str(facility.get("operating_end", "23:59")), float(facility.get("tariff_per_kwh", 0)),
                str(facility.get("currency", "INR")), str(facility.get("scheduling_algorithm", "LAFS")),
                float(facility.get("simulation_step_min", 1.0)),
                str(facility.get("data_origin", "Manually configured")), now)
        if existing:
            con.execute("""UPDATE facilities SET name=?,location=?,timezone=?,status=?,grid_capacity_kw=?,
                operating_start=?,operating_end=?,tariff_per_kwh=?,currency=?,scheduling_algorithm=?,
                simulation_step_min=?,data_origin=?,updated_at=? WHERE id=?""",
                (*vals, fid))
            event_type, summary = "facility.updated", f"Updated configuration for {name}"
        else:
            con.execute("""INSERT INTO facilities
                (name,location,timezone,status,grid_capacity_kw,operating_start,operating_end,
                 tariff_per_kwh,currency,scheduling_algorithm,simulation_step_min,data_origin,updated_at,id,created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (*vals[:-1], now, fid, now))
            event_type, summary = "facility.created", f"Created facility {name}"
        con.execute("DELETE FROM stations WHERE facility_id=?", (fid,))
        for s in stations:
            con.execute("INSERT INTO stations VALUES(?,?,?,?,?,?)",
                        (fid, str(s["id"]).strip(), str(s.get("name") or s["id"]).strip(),
                         str(s.get("location", "")), float(s["power_cap_kw"]), str(s.get("status", "available"))))
        for c in chargers:
            ptype = str(c.get("power_type") or ("DC" if "DC" in str(c["connector_type"]).upper() or "CCS" in str(c["connector_type"]).upper() else "AC")).upper()
            if ptype not in ("AC", "DC"):
                raise ValueError("Charger power type must be AC or DC.")
            con.execute("INSERT INTO chargers VALUES(?,?,?,?,?,?,?)",
                        (fid, str(c["station_id"]).strip(), str(c["id"]).strip(),
                         str(c["connector_type"]).strip(), ptype, float(c["rated_power_kw"]),
                         str(c.get("status", "available"))))
        con.execute("INSERT INTO audit_events(facility_id,event_type,summary,details_json,created_at) VALUES(?,?,?,?,?)",
                    (fid, event_type, summary, json.dumps({"stations": len(stations), "chargers": len(chargers)}), now))


def archive_facility(facility_id: str, path: str | Path | None = None) -> None:
    with connect(path) as con:
        active_count = con.execute("SELECT COUNT(*) FROM facilities WHERE status='active'").fetchone()[0]
        row = con.execute("SELECT name FROM facilities WHERE id=?", (facility_id,)).fetchone()
        if not row:
            raise ValueError("Facility does not exist.")
        if active_count <= 1:
            raise ValueError("At least one active facility must remain available.")
        now = _now()
        con.execute("UPDATE facilities SET status='archived',updated_at=? WHERE id=?", (now, facility_id))
        con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                    (facility_id, "facility.archived", f"Archived {row['name']}", now))


def normalize_request_records(requests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Repair legacy ID collisions without dropping records; flag duplicate active plates."""
    normalized = requests
    issues: list[str] = []
    seen_sessions: set[str] = set()
    seen_request_ids: set[str] = set()
    active_vehicles: set[str] = set()
    for request in normalized:
        old_sid = str(request.get("session_id") or "").strip()
        old_rid = str(request.get("request_id") or "").strip()
        sid, rid = old_sid, old_rid
        if not sid or sid in seen_sessions or sid in seen_request_ids or sid == rid:
            if sid:
                request.setdefault("legacy_session_id", sid)
                issues.append(f"Reassigned duplicate or conflated session ID {sid}.")
            sid = _new_identifier("ses")
        if not rid or rid in seen_request_ids or rid in seen_sessions or rid == sid:
            rid = _new_identifier("req")
            issues.append("Assigned a new unique request ID to a retained request.")
        request["session_id"] = sid
        request["request_id"] = rid
        seen_sessions.add(sid)
        seen_request_ids.add(rid)

        reg = str(request.get("vehicle_reg_no") or "").strip().upper()
        if reg:
            request["vehicle_reg_no"] = reg
        status = str(request.get("status", "Queued")).strip().lower()
        if reg and status not in _TERMINAL_REQUEST_STATUSES:
            if reg in active_vehicles:
                request["status"] = "Duplicate review"
                request["duplicate_reason"] = (
                    "A separate active request for this vehicle already exists. This record was retained for review and excluded from dispatch."
                )
                issues.append(f"Retained duplicate active registration {reg} for review; no request was deleted.")
            else:
                active_vehicles.add(reg)
    return normalized, issues


def _save_requests_in_transaction(con: sqlite3.Connection, facility_id: str,
                                  requests: list[dict[str, Any]], now: str) -> None:
    current_ids = {str(q["session_id"]) for q in requests}
    stored_ids = {r[0] for r in con.execute("SELECT session_id FROM requests WHERE facility_id=?", (facility_id,))}
    removed = stored_ids - current_ids
    if removed:
        con.executemany("DELETE FROM requests WHERE facility_id=? AND session_id=?",
                        [(facility_id, sid) for sid in removed])
    # Review/terminal rows first so a legacy duplicate plate is safely moved out
    # of the active set before its retained primary request is refreshed.
    ordered_requests = sorted(requests, key=lambda q: str(q.get("status", "Queued")).strip().lower()
                              not in _TERMINAL_REQUEST_STATUSES)
    for q in ordered_requests:
        sid, rid = str(q["session_id"]), str(q["request_id"])
        reg = str(q.get("vehicle_reg_no", "")).strip().upper()
        q["vehicle_reg_no"] = reg
        con.execute("""INSERT INTO requests
            (facility_id,session_id,request_id,vehicle_reg_no,station_id,payload_json,status,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(facility_id,session_id) DO UPDATE SET
            request_id=excluded.request_id,vehicle_reg_no=excluded.vehicle_reg_no,
            station_id=excluded.station_id,payload_json=excluded.payload_json,
            status=excluded.status,updated_at=excluded.updated_at""",
            (facility_id, sid, rid, reg, str(q.get("station_id", "")), json.dumps(q, default=str),
             str(q.get("status", "Queued")), now, now))


def save_requests(facility_id: str, requests: list[dict[str, Any]], path: str | Path | None = None) -> None:
    normalized, issues = normalize_request_records(requests)
    now = _now()
    with connect(path) as con:
        _save_requests_in_transaction(con, facility_id, normalized, now)
        for issue in issues:
            con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                        (facility_id, "request.identity_repaired", issue, now))


def save_simulation_snapshot(facility_id: str, requests: list[dict[str, Any]],
                             path: str | Path | None = None, *,
                             allow_rewind: bool = False) -> bool:
    """Atomically persist clock-owned fields without replacing concurrent queue edits.

    Returns False if a newer browser session has already persisted a later clock
    snapshot. Reset is the only caller allowed to intentionally move time back.
    """
    simulation_fields = (
        "status", "waiting_reason", "energy_delivered_kwh", "current_soc_pct",
        "assigned_charger_id", "allocated_power_kw", "simulation_minute",
    )
    now = _now()
    with connect(path) as con:
        con.execute("BEGIN IMMEDIATE")
        updates = []
        seen = set()
        for request in requests:
            session_id = str(request.get("session_id", "")).strip()
            if not session_id or session_id in seen:
                raise ValueError("Simulation snapshot contains a missing or duplicate session ID.")
            seen.add(session_id)
            row = con.execute(
                "SELECT payload_json FROM requests WHERE facility_id=? AND session_id=?",
                (facility_id, session_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Session {session_id} is no longer in this facility's request store.")
            try:
                payload = json.loads(row[0])
            except (TypeError, ValueError):
                payload = {}
            saved_minute = payload.get("simulation_minute")
            new_minute = request.get("simulation_minute")
            if not allow_rewind and saved_minute is not None and new_minute is not None:
                try:
                    if float(new_minute) + 1e-6 < float(saved_minute):
                        return False
                except (TypeError, ValueError):
                    pass
            for field in simulation_fields:
                if field in request:
                    payload[field] = request[field]
            updates.append((json.dumps(payload, default=str), str(payload.get("status", "Queued")),
                            now, facility_id, session_id))
        con.executemany(
            "UPDATE requests SET payload_json=?,status=?,updated_at=? WHERE facility_id=? AND session_id=?",
            updates,
        )
    return True


def load_requests(facility_id: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as con:
        rows = con.execute("SELECT session_id,request_id,vehicle_reg_no,payload_json FROM requests "
                           "WHERE facility_id=? ORDER BY created_at,session_id", (facility_id,)).fetchall()
    requests = []
    for row in rows:
        try:
            request = json.loads(row[3])
        except (TypeError, ValueError):
            request = {}
        request.update({"session_id": row[0], "request_id": row[1],
                        "vehicle_reg_no": row[2] or request.get("vehicle_reg_no", "")})
        requests.append(request)
    before = json.dumps(requests, sort_keys=True, default=str)
    requests, issues = normalize_request_records(requests)
    if issues or before != json.dumps(requests, sort_keys=True, default=str):
        save_requests(facility_id, requests, path)
    return requests


def reset_operational_data(facility_id: str, path: str | Path | None = None) -> None:
    """Clear a facility's request queue and saved schedules while retaining configuration."""
    now = _now()
    with connect(path) as con:
        if not con.execute("SELECT 1 FROM facilities WHERE id=?", (facility_id,)).fetchone():
            raise ValueError("Facility does not exist.")
        con.execute("DELETE FROM requests WHERE facility_id=?", (facility_id,))
        con.execute("DELETE FROM schedules WHERE facility_id=?", (facility_id,))
        con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                    (facility_id, "operational_data.reset", "Reset local vehicle queue and dispatch records; facility configuration retained", now))


def save_schedule(facility_id: str, algorithm: str, grid_capacity_kw: float,
                  epoch_anchor: Any, results: list[Any], path: str | Path | None = None,
                  dispatch_key: str | None = None) -> int:
    rows = [asdict(r) for r in results]
    now = _now()
    with connect(path) as con:
        if dispatch_key:
            existing = con.execute("SELECT id FROM schedules WHERE facility_id=? AND dispatch_key=?",
                                   (facility_id, dispatch_key)).fetchone()
            if existing:
                return int(existing[0])
        cur = con.execute("INSERT INTO schedules(facility_id,algorithm,grid_capacity_kw,epoch_anchor,dispatch_key,results_json,created_at) VALUES(?,?,?,?,?,?,?)",
                          (facility_id, algorithm, float(grid_capacity_kw), str(epoch_anchor), dispatch_key,
                           json.dumps(rows, default=str), now))
        con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                    (facility_id, "schedule.generated", f"Generated {algorithm} schedule for {len(rows)} requests", now))
        return int(cur.lastrowid)


def save_dispatch_state(facility_id: str, requests: list[dict[str, Any]], algorithm: str,
                        grid_capacity_kw: float, epoch_anchor: Any, results: list[Any],
                        dispatch_key: str, path: str | Path | None = None) -> int:
    """Atomically persist request progress and its schedule, keyed idempotently."""
    normalized, issues = normalize_request_records(requests)
    rows = [asdict(r) for r in results]
    now = _now()
    with connect(path) as con:
        _save_requests_in_transaction(con, facility_id, normalized, now)
        existing = con.execute("SELECT id FROM schedules WHERE facility_id=? AND dispatch_key=?",
                               (facility_id, dispatch_key)).fetchone()
        if existing:
            schedule_id = int(existing[0])
        else:
            cur = con.execute("INSERT INTO schedules(facility_id,algorithm,grid_capacity_kw,epoch_anchor,dispatch_key,results_json,created_at) VALUES(?,?,?,?,?,?,?)",
                              (facility_id, algorithm, float(grid_capacity_kw), str(epoch_anchor), dispatch_key,
                               json.dumps(rows, default=str), now))
            schedule_id = int(cur.lastrowid)
            con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                        (facility_id, "schedule.generated", f"Generated {algorithm} schedule for {len(rows)} requests", now))
        for issue in issues:
            con.execute("INSERT INTO audit_events(facility_id,event_type,summary,created_at) VALUES(?,?,?,?)",
                        (facility_id, "request.identity_repaired", issue, now))
        return schedule_id


def load_latest_schedule(facility_id: str, path: str | Path | None = None) -> dict[str, Any] | None:
    from scheduler import SimResult
    with connect(path) as con:
        row = con.execute("SELECT * FROM schedules WHERE facility_id=? ORDER BY id DESC LIMIT 1", (facility_id,)).fetchone()
        facility = con.execute("SELECT updated_at FROM facilities WHERE id=?", (facility_id,)).fetchone()
    if not row:
        return None
    # Do not present a dispatch as current after facility or charger configuration changed.
    if facility and str(row["created_at"]) <= str(facility["updated_at"]):
        return None
    return {"id": row["id"], "algorithm": row["algorithm"], "grid_capacity_kw": row["grid_capacity_kw"],
            "dispatch_key": row["dispatch_key"],
            "epoch_anchor": row["epoch_anchor"], "created_at": row["created_at"],
            "results": [SimResult(**r) for r in json.loads(row["results_json"])]}


def audit_history(facility_id: str, path: str | Path | None = None, limit: int = 100) -> pd.DataFrame:
    with connect(path) as con:
        rows = con.execute("SELECT event_type,summary,details_json,created_at FROM audit_events WHERE facility_id=? ORDER BY id DESC LIMIT ?",
                           (facility_id, int(limit))).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def facility_data_summary(facility_id: str, path: str | Path | None = None) -> dict[str, int]:
    with connect(path) as con:
        return {
            "requests": con.execute("SELECT COUNT(*) FROM requests WHERE facility_id=?", (facility_id,)).fetchone()[0],
            "schedules": con.execute("SELECT COUNT(*) FROM schedules WHERE facility_id=?", (facility_id,)).fetchone()[0],
        }
