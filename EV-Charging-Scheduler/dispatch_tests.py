"""Automatic dispatch, identifier integrity, and persistence regression suite."""
from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

import data_loader as dl
from dispatch_service import calculate_live_dispatch, dispatch_key
from facility_store import (
    connect, facility_data_summary, initialize, load_latest_schedule,
    load_requests, save_configuration, save_dispatch_state, save_requests,
    save_simulation_snapshot,
)
from scheduler import ALGORITHMS, snapshot_at


def infrastructure(count=4, ratings=None, stations=("S1",), cap=250.0):
    if ratings is None:
        defaults = [100.0, 100.0, 75.0, 75.0]
        ratings = [defaults[index] if index < len(defaults) else 50.0
                   for index in range(count)]
    if len(ratings) < count:
        raise ValueError("Provide one charger rating for every configured endpoint")
    rows = []
    for index in range(count):
        station_id = stations[index % len(stations)]
        rows.append({
            "station_id": station_id, "station_name": station_id,
            "station_location": "Test site", "station_aggregate_power_cap_kw_scenario": cap,
            "station_status": "available", "environment": "Test facility",
            "charger_id": f"{station_id}-CP{index + 1}", "connector_type": "CCS2 DC",
            "power_type": "DC", "rated_power_kw": float(ratings[index]),
            "charger_status": "available", "power_cap_is_measured": 0,
            "profile_notes": "Configured test endpoint", "source_basis": "Unit test",
        })
    return pd.DataFrame(rows)


def vehicles(count, station_id="S1", arrivals=None, registration_prefix="TEST"):
    arrivals = arrivals or [0.0] * count
    base = pd.Timestamp("2026-04-01 09:00:00")
    result = []
    for index in range(count):
        arrival = base + pd.Timedelta(minutes=float(arrivals[index]))
        request_id, session_id = dl.new_live_request_ids()
        result.append({
            "request_id": request_id, "session_id": session_id,
            "vehicle_reg_no": f"{registration_prefix}-{index + 1:03d}",
            "manufacturer": "Test EV", "model_name": "Test EV", "model_variant": "Test",
            "vehicle_model_id": "TEST-EV", "battery_capacity_kwh": 80.0,
            "soc_arrival_pct": 25.0, "soc_target_pct": 50.0,
            "energy_required_kwh": 20.0, "max_charge_power_kw": 100.0,
            "veh_max_ac_kw": 11.0, "veh_max_dc_kw": 100.0,
            "connector_type": "CCS2 DC", "station_id": station_id,
            "arrival_time_IST": str(arrival),
            "departure_time_IST": str(arrival + pd.Timedelta(minutes=90)),
            "status": "Queued", "record_type": "registered",
        })
    return result


def facility_config(db: Path, stations_df: pd.DataFrame, facility_id="dispatch-test"):
    initialize(pd.DataFrame(), db)
    station_records = []
    for station_id, group in stations_df.groupby("station_id"):
        station_records.append({
            "id": str(station_id), "name": str(station_id), "location": "Test site",
            "power_cap_kw": float(group["station_aggregate_power_cap_kw_scenario"].iloc[0]),
            "status": "available",
        })
    charger_records = [{
        "station_id": row.station_id, "id": row.charger_id,
        "connector_type": row.connector_type, "power_type": row.power_type,
        "rated_power_kw": row.rated_power_kw, "status": row.charger_status,
    } for row in stations_df.itertuples(index=False)]
    save_configuration({
        "id": facility_id, "name": "Dispatch Test", "location": "",
        "timezone": "Asia/Kolkata", "status": "active", "grid_capacity_kw": 250.0,
        "operating_start": "00:00", "operating_end": "23:59", "tariff_per_kwh": 0,
        "currency": "INR", "scheduling_algorithm": "HYBRID", "simulation_step_min": 1.0,
        "data_origin": "Unit test",
    }, station_records, charger_records, db)
    return facility_id


def legacy_database_migration(db: Path):
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE facilities (
      id TEXT PRIMARY KEY,name TEXT NOT NULL,location TEXT NOT NULL DEFAULT '',timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
      status TEXT NOT NULL DEFAULT 'active',grid_capacity_kw REAL NOT NULL,operating_start TEXT NOT NULL DEFAULT '00:00',
      operating_end TEXT NOT NULL DEFAULT '23:59',tariff_per_kwh REAL NOT NULL DEFAULT 0,currency TEXT NOT NULL DEFAULT 'INR',
      scheduling_algorithm TEXT NOT NULL DEFAULT 'HYBRID',simulation_step_min REAL NOT NULL DEFAULT 1,
      data_origin TEXT NOT NULL DEFAULT 'Test',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE requests (
      facility_id TEXT NOT NULL,session_id TEXT NOT NULL,station_id TEXT NOT NULL,payload_json TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'Queued',created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
      PRIMARY KEY(facility_id,session_id));
    CREATE TABLE schedules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,facility_id TEXT NOT NULL,algorithm TEXT NOT NULL,
      grid_capacity_kw REAL NOT NULL,epoch_anchor TEXT NOT NULL,results_json TEXT NOT NULL,created_at TEXT NOT NULL);
    """)
    now = "2026-01-01T00:00:00+00:00"
    for facility_id in ("old-a", "old-b"):
        con.execute("INSERT INTO facilities(id,name,grid_capacity_kw,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (facility_id, facility_id, 250.0, now, now))
        payload = {"session_id": "LIVE-REQ-001", "vehicle_reg_no": f"{facility_id}-EV",
                   "station_id": "S1", "status": "Queued"}
        con.execute("INSERT INTO requests VALUES(?,?,?,?,?,?,?)",
                    (facility_id, "LIVE-REQ-001", "S1", json.dumps(payload), "Queued", now, now))
        con.execute("INSERT INTO schedules(facility_id,algorithm,grid_capacity_kw,epoch_anchor,results_json,created_at) VALUES(?,?,?,?,?,?)",
                    (facility_id, "HYBRID", 250.0, now,
                     json.dumps([{"session_id": "LIVE-REQ-001", "timeline": [], "started": False}]), now))
    con.commit(); con.close()
    initialize(pd.DataFrame(), db)
    with connect(db) as migrated:
        requests = migrated.execute("SELECT facility_id,session_id,request_id,payload_json FROM requests ORDER BY facility_id").fetchall()
        schedules = migrated.execute("SELECT facility_id,results_json FROM schedules ORDER BY facility_id").fetchall()
        assert len(requests) == 2
        assert len({row[1] for row in requests}) == len({row[2] for row in requests}) == 2
        for row in requests:
            payload = json.loads(row[3])
            assert payload["request_id"] == row[2] and payload["session_id"] == row[1]
        for row in schedules:
            result = json.loads(row[1])[0]
            matching = next(req for req in requests if req[0] == row[0])
            assert result["session_id"] == matching[1] and result["request_id"] == matching[2]


def run():
    one_port_fleet = infrastructure()
    one = vehicles(1)
    plan_one = calculate_live_dispatch(one, one_port_fleet, 250.0)
    assert len(plan_one["results"]) == 1 and plan_one["results"][0].started
    assert plan_one["results"][0].charger_id_assigned in set(one_port_fleet.charger_id)
    assert one[0]["request_id"] != one[0]["session_id"]
    print("PASS 1. One registration dispatches to one of four available compatible chargers")

    one_endpoint = infrastructure(count=1, ratings=[50.0])
    one_vehicle = vehicles(1, registration_prefix="A")
    plan_a = calculate_live_dispatch(one_vehicle, one_endpoint, 250.0)
    result_a = plan_a["results"][0]
    assert len(snapshot_at(plan_a["results"], 0.0)["charging"]) == 1
    assert result_a.completed_by_deadline and result_a.energy_delivered_kwh >= result_a.requested_energy_kwh - 1e-6
    print("PASS A. One vehicle automatically charges and completes on one configured charger")

    three_endpoint = infrastructure(count=3, ratings=[50.0] * 3)
    three_vehicle = vehicles(3, registration_prefix="B")
    plan_b = calculate_live_dispatch(three_vehicle, three_endpoint, 250.0)
    assert len(snapshot_at(plan_b["results"], 0.0)["charging"]) == 3
    print("PASS B. Three compatible vehicles charge simultaneously on three endpoints")

    six_endpoint = infrastructure(count=6, ratings=[50.0] * 6)
    six_vehicle = vehicles(6, registration_prefix="C")
    plan_c = calculate_live_dispatch(six_vehicle, six_endpoint, 250.0)
    assert len(snapshot_at(plan_c["results"], 0.0)["charging"]) == 6
    print("PASS C. Six vehicles charge independently and simultaneously on six endpoints")

    eight_vehicle = vehicles(8, registration_prefix="D")
    plan_d = calculate_live_dispatch(eight_vehicle, six_endpoint, 250.0)
    first_d = snapshot_at(plan_d["results"], 0.0)
    assert len(first_d["charging"]) == 6 and len(first_d["waiting"]) == 2
    first_release = min(result.charge_end_min for result, _ in first_d["charging"].values())
    deferred_d = [result for result in plan_d["results"]
                  if result.charge_start_min is not None and result.charge_start_min > 1e-6]
    assert len(deferred_d) == 2 and all(result.charge_start_min >= first_release - 1e-6
                                        for result in deferred_d)
    after_handoff = snapshot_at(plan_d["results"], min(r.charge_start_min for r in deferred_d) + 0.01)
    assert len(after_handoff["charging"]) == 2
    print("PASS D. Eight vehicles on six endpoints start six, queue two, then hand off released ports automatically")

    degraded = infrastructure(count=3, ratings=[50.0] * 3)
    degraded.loc[degraded.index[:2], "charger_status"] = "offline"
    plan_e = calculate_live_dispatch(vehicles(2, registration_prefix="E"), degraded, 250.0)
    first_e = snapshot_at(plan_e["results"], 0.0)
    assert len(first_e["charging"]) == 1 and len(first_e["waiting"]) == 1
    assert all(result.charger_id_assigned == degraded.loc[degraded.charger_status.eq("available"), "charger_id"].iloc[0]
               for result, _ in first_e["charging"].values())
    assert len(plan_e["results"]) == 2
    print("PASS E. Offline chargers are skipped; the unserved request remains safely queued")

    two = vehicles(2)
    plan_two = calculate_live_dispatch(two, one_port_fleet, 250.0)
    initial_two = snapshot_at(plan_two["results"], 0.0)
    assert len(initial_two["charging"]) == 2 and not initial_two["waiting"]
    print("PASS 1a. Two simultaneous arrivals start in parallel on separate available chargers")

    four = vehicles(4)
    plan_four = calculate_live_dispatch(four, one_port_fleet, 250.0)
    initial_four = snapshot_at(plan_four["results"], 0.0)
    assert len(initial_four["charging"]) == 4 and not initial_four["waiting"]
    print("PASS 1b. Four simultaneous arrivals use all four configured compatible chargers")

    # Endpoint count must come from the configured infrastructure, while the
    # independent facility cap still limits concurrent delivered power.
    for endpoint_count in (4, 6, 10):
        configured = infrastructure(count=endpoint_count, ratings=[50.0] * endpoint_count,
                                    cap=250.0)
        fleet = vehicles(endpoint_count, registration_prefix=f"PORT-{endpoint_count}")
        configured_plan = calculate_live_dispatch(fleet, configured, 250.0)
        first_snapshot = snapshot_at(configured_plan["results"], 0.0)
        active = list(first_snapshot["charging"].values())
        expected_active = endpoint_count
        assert len(active) == expected_active
        assert sum(power for _, power in active) <= 250.0 + 1e-6
        assert len({result.charger_id_assigned for result, _ in active}) == len(active)
        ratings_by_id = dict(zip(configured.charger_id.astype(str), configured.rated_power_kw.astype(float)))
        assert all(power <= ratings_by_id[result.charger_id_assigned] + 1e-6
                   for result, power in active)
        assert all(result.charger_id_assigned is None or result.charger_id_assigned in ratings_by_id
                   for result in configured_plan["results"])
        for minute in sorted({point for result in configured_plan["results"]
                              for start, end, _ in result.timeline for point in (start, end)}):
            power_at_minute = [(result, power) for result in configured_plan["results"]
                               for start, end, power in result.timeline
                               if start <= minute < end]
            assert sum(power for _, power in power_at_minute) <= 250.0 + 1e-6
            assert len({result.charger_id_assigned for result, _ in power_at_minute}) == len(power_at_minute)
            assert all(power <= ratings_by_id[result.charger_id_assigned] + 1e-6
                       for result, power in power_at_minute)
        waiting = len(first_snapshot["waiting"])
        assert waiting == 0
        print(f"PASS 1c. {endpoint_count} configured endpoints allocate {expected_active} in parallel, "
              f"leave {waiting} waiting, and respect the 250 kW and per-port limits")

    five = vehicles(5)
    plan_five = calculate_live_dispatch(five, one_port_fleet, 250.0)
    initial = snapshot_at(plan_five["results"], 0.0)
    assert len(initial["charging"]) == 4 and len(initial["waiting"]) == 1
    print("PASS 2. Five simultaneous vehicles start on four chargers; one remains waiting")

    existing_four = vehicles(4, registration_prefix="G")
    newly_registered = vehicles(1, registration_prefix="G-NEW")
    plan_g = calculate_live_dispatch(existing_four + newly_registered, one_port_fleet, 250.0)
    first_g = snapshot_at(plan_g["results"], 0.0)
    assert len(first_g["charging"]) == 4 and len(first_g["waiting"]) == 1
    assert any(result.vehicle_reg_no == "G-NEW-001" for result in plan_g["results"])
    assert len({result.charger_id_assigned for result, _ in first_g["charging"].values()}) == 4
    print("PASS G. New registration is dispatched in the active-fleet plan without double-booking full endpoints")

    six = vehicles(6)
    plan_six = calculate_live_dispatch(six, one_port_fleet, 250.0)
    initial_six = snapshot_at(plan_six["results"], 0.0)
    assert len(initial_six["charging"]) == 4 and len(initial_six["waiting"]) == 2
    print("PASS 3. Six simultaneous arrivals use four chargers and leave two waiting")

    starts = sorted(r.charge_start_min for r in plan_six["results"] if r.charge_start_min is not None)
    ends = sorted(r.charge_end_min for r in plan_six["results"] if r.charge_end_min is not None)
    first_end = min(r.charge_end_min for r in plan_six["results"] if r.started)
    next_start = min(r.charge_start_min for r in plan_six["results"]
                     if r.charge_start_min is not None and r.charge_start_min > 0)
    assert next_start >= first_end - 1e-6
    after_release = snapshot_at(plan_six["results"], next_start + 0.01)
    assert len(after_release["charging"]) >= 1
    assert len(starts) >= 5 and len(ends) >= 4
    print("PASS 4. Waiting vehicles auto-start in the precomputed event schedule when a session completes")

    future = vehicles(2, arrivals=[0, 30])
    future_plan = calculate_live_dispatch(future, one_port_fleet, 250.0)
    later = next(r for r in future_plan["results"] if r.vehicle_reg_no.endswith("002"))
    assert later.charge_start_min is None or later.charge_start_min >= later.arrival_min - 1e-8
    assert all(start >= later.arrival_min - 1e-8 for start, _, _ in later.timeline)
    before_arrival = snapshot_at(future_plan["results"], later.arrival_min - 0.1)
    assert all(result.session_id != later.session_id for result, _ in before_arrival["charging"].values())
    print("PASS 5. Future arrivals are neither assigned nor charged before arrival")

    two_stations = infrastructure(count=2, ratings=[75, 75], stations=("S1", "S2"), cap=100.0)
    at_s1 = vehicles(1, station_id="S1")
    station_plan = calculate_live_dispatch(at_s1, two_stations, 150.0)
    assert station_plan["results"][0].charger_id_assigned.startswith("S1-")
    blocked = calculate_live_dispatch(at_s1, two_stations[two_stations.station_id.eq("S2")], 150.0)
    assert blocked["results"][0].charger_id_assigned is None and not blocked["results"][0].timeline
    print("PASS 6. Vehicles use only compatible endpoints at their assigned station")

    peak = 0.0
    rating_by_charger = dict(zip(one_port_fleet.charger_id, one_port_fleet.rated_power_kw))
    for left, right in zip(sorted({t for r in plan_five["results"] for a, b, _ in r.timeline for t in (a, b)}),
                           sorted({t for r in plan_five["results"] for a, b, _ in r.timeline for t in (a, b)})[1:]):
        moment = (left + right) / 2.0
        active = [(r, power) for r in plan_five["results"] for start, end, power in r.timeline
                  if start <= moment < end]
        peak = max(peak, sum(power for _, power in active))
    assert peak <= 250.0 + 1e-6
    for result in plan_five["results"]:
        if result.charger_id_assigned:
            assert result.allocated_power_peak_kw <= rating_by_charger[result.charger_id_assigned] + 1e-6
    print(f"PASS 7–8. Facility power peaks at {peak:.2f} kW; no endpoint exceeds its rating")

    reference = dl.load_stations()
    demo_station = str(reference.station_id.iloc[0])
    demo_infrastructure = reference[reference.station_id.astype(str).eq(demo_station)]
    demo_once = dl.get_default_live_fleet(demo_station, demo_infrastructure)
    demo_queue, added_once = dl.merge_live_demo_requests([], demo_once)
    demo_twice = dl.get_default_live_fleet(demo_station, demo_infrastructure)
    same_queue, added_twice = dl.merge_live_demo_requests(demo_queue, demo_twice)
    assert added_once == len(demo_once) and added_twice == 0 and len(same_queue) == len(demo_queue)
    assert len({q["session_id"] for q in same_queue}) == len(same_queue)
    assert len({q["request_id"] for q in same_queue}) == len(same_queue)
    assert len({q["vehicle_reg_no"] for q in same_queue}) == len(same_queue)
    print("PASS 9. Demo loading twice is idempotent with unique request/session IDs and no repeated active vehicles")

    repeated = calculate_live_dispatch(five, one_port_fleet, 250.0)
    assert [r.session_id for r in repeated["results"]] == [r.session_id for r in plan_five["results"]]
    assert repeated["dispatch_key"] == plan_five["dispatch_key"]
    print("PASS 10. Repeated dispatch creates no duplicate session IDs and has a stable idempotency key")

    with tempfile.TemporaryDirectory(prefix="voltflow-dispatch-") as folder:
        db = Path(folder) / "test.sqlite3"
        fid = facility_config(db, one_port_fleet)
        save_requests(fid, five, db)
        key = dispatch_key(five, one_port_fleet, 250.0, "HYBRID", 1.0)
        schedule_id = save_dispatch_state(fid, five, "HYBRID", 250.0,
                                          plan_five["epoch_anchor"], plan_five["results"], key, db)
        repeated_id = save_dispatch_state(fid, five, "HYBRID", 250.0,
                                          plan_five["epoch_anchor"], plan_five["results"], key, db)
        restored_queue = load_requests(fid, db)
        restored_schedule = load_latest_schedule(fid, db)
        assert len(restored_queue) == 5 and schedule_id == repeated_id
        assert len({q["request_id"] for q in restored_queue}) == 5
        assert len({q["session_id"] for q in restored_queue}) == 5
        assert restored_schedule and restored_schedule["dispatch_key"] == key
        assert facility_data_summary(fid, db)["schedules"] == 1
        print("PASS 11. Persisted request/session state and active schedule recover after reload; repeated save is idempotent")
        print("PASS F. Active sessions recover from SQLite after simulated application restart")

        clock_update = dict(restored_queue[0], status="Charging", simulation_minute=10.0,
                            energy_delivered_kwh=1.5, current_soc_pct=27.0)
        stale_clock_update = dict(clock_update, status="Waiting", simulation_minute=4.0,
                                  energy_delivered_kwh=0.0, current_soc_pct=25.0)
        assert save_simulation_snapshot(fid, [clock_update], db)
        assert not save_simulation_snapshot(fid, [stale_clock_update], db)
        clock_restored = next(q for q in load_requests(fid, db)
                              if q["session_id"] == clock_update["session_id"])
        assert clock_restored["simulation_minute"] == 10.0
        assert clock_restored["status"] == "Charging"
        assert clock_restored["energy_delivered_kwh"] == 1.5
        print("PASS 11a. Clock writes are transactional, merge only simulation fields, and reject stale snapshots")

        # SQLite rejects a second active request for the same plate even if callers bypass the UI.
        try:
            with connect(db) as con:
                con.execute("INSERT INTO requests(facility_id,session_id,request_id,vehicle_reg_no,station_id,payload_json,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                            (fid, "ses_raw_duplicate", "req_raw_duplicate", "TEST-001", "S1", "{}", "Queued", "now", "now"))
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("Database trigger allowed a duplicate active vehicle registration")

        duplicate_ids = [dict(five[0]), dict(five[1])]
        duplicate_ids[1]["session_id"] = duplicate_ids[0]["session_id"]
        duplicate_ids[1]["request_id"] = duplicate_ids[0]["request_id"]
        duplicate_ids[1]["vehicle_reg_no"] = "REPAIRED-002"
        save_requests(fid, duplicate_ids, db)
        retained = load_requests(fid, db)
        assert len(retained) == 2 and len({q["session_id"] for q in retained}) == 2
        assert len({q["request_id"] for q in retained}) == 2
        print("PASS 12. Legacy ID collisions are re-keyed without deleting either vehicle request")

        duplicate_plates = [dict(five[0]), dict(five[1])]
        duplicate_plates[1]["session_id"] = dl.new_live_request_ids()[1]
        duplicate_plates[1]["request_id"] = dl.new_live_request_ids()[0]
        duplicate_plates[1]["vehicle_reg_no"] = duplicate_plates[0]["vehicle_reg_no"]
        save_requests(fid, duplicate_plates, db)
        reviewed = load_requests(fid, db)
        assert len(reviewed) == 2
        assert sum(q.get("status") == "Duplicate review" for q in reviewed) == 1
        print("PASS 13. Existing duplicate registrations remain visible and are excluded pending review")

    with tempfile.TemporaryDirectory(prefix="voltflow-legacy-migration-") as folder:
        legacy_database_migration(Path(folder) / "legacy.sqlite3")
    print("PASS 14. Legacy database migration preserves requests and remaps saved schedule session IDs")

    for algorithm in ALGORITHMS:
        policy_plan = calculate_live_dispatch(five, one_port_fleet, 250.0, algorithm=algorithm)
        assert len(policy_plan["results"]) == 5
        assert len({r.session_id for r in policy_plan["results"]}) == 5
    print("PASS 15. FCFS, EDF, LAFS and Hybrid all run through the automatic dispatch service")
    print("All automatic dispatch regression checks passed.")


if __name__ == "__main__":
    run()
