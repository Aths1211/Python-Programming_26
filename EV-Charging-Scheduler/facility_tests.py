"""Facility persistence and multi-vehicle dispatch regression checks."""
from __future__ import annotations

import tempfile
from pathlib import Path
import math

import pandas as pd
import numpy as np

from facility_store import (
    initialize, list_facilities, get_facility, get_station_frame,
    save_configuration, save_requests, load_requests, save_schedule,
    load_latest_schedule, archive_facility, audit_history,
    reset_operational_data,
)
from scheduler import (Session, run_single_algorithm, snapshot_at, _hybrid_priority_rows,
                       validate_hybrid_weights, results_to_dataframe, compute_summary)
from visualizations import (fig_energy_comparison, fig_completion_rate, fig_waiting_time,
                            fig_station_load, fig_missed_energy, fig_radar_summary)


def _configuration(fid: str, name: str, grid: float, station_id: str,
                   charger_ids: list[str], status="available"):
    facility = {"id": fid, "name": name, "location": "Test", "timezone": "Asia/Kolkata",
        "status": "active", "grid_capacity_kw": grid, "operating_start": "06:00",
        "operating_end": "23:00", "tariff_per_kwh": 8.5, "currency": "INR",
        "scheduling_algorithm": "LAFS", "simulation_step_min": 1.0,
        "data_origin": "Test fixture"}
    stations = [{"id": station_id, "name": station_id, "location": "Test",
        "power_cap_kw": grid, "status": "available"}]
    chargers = [{"station_id": station_id, "id": cid, "connector_type": "Type-2 AC",
        "power_type": "AC", "rated_power_kw": 7.2, "status": status}
        for cid in charger_ids]
    return facility, stations, chargers


def run():
    with tempfile.TemporaryDirectory(prefix="voltflow-tests-") as td:
        db = Path(td) / "gridcharge.sqlite3"
        initialize(pd.DataFrame(), db)

        f1, s1, c1 = _configuration("west-yard", "West Yard", 100, "W1", ["W-1", "W-2", "W-3"])
        f2, s2, c2 = _configuration("east-depot", "East Depot", 35, "E1", ["E-1"])
        save_configuration(f1, s1, c1, db)
        save_configuration(f2, s2, c2, db)
        assert get_facility("west-yard", db)["grid_capacity_kw"] == 100
        assert get_facility("east-depot", db)["grid_capacity_kw"] == 35
        assert len(get_station_frame("west-yard", db)) == 3
        assert len(get_station_frame("east-depot", db)) == 1

        save_requests("west-yard", [{"session_id":"W-REQ-1","station_id":"W1","status":"Queued"},
            {"session_id":"W-REQ-2","station_id":"W1","status":"Queued"}], db)
        save_requests("east-depot", [{"session_id":"E-REQ-1","station_id":"E1","status":"Queued"}], db)
        # Saving a queue snapshot must remove items deleted by the operator.
        save_requests("west-yard", [{"session_id":"W-REQ-2","station_id":"W1","status":"Queued"}], db)
        assert [q["session_id"] for q in load_requests("west-yard", db)] == ["W-REQ-2"]
        assert [q["session_id"] for q in load_requests("east-depot", db)] == ["E-REQ-1"]

        # Simulate independently configured facilities with enough ports and a shared site cap.
        station_rows = []
        for station_idx in range(2):
            for port_idx in range(5):
                station_rows.append({"station_id": f"S{station_idx+1}", "charger_id": f"S{station_idx+1}-CP{port_idx+1}",
                    "connector_type":"Type-2 AC", "rated_power_kw":7.2, "charger_status":"available",
                    "station_status":"available", "station_aggregate_power_cap_kw_scenario":25.0})
        infrastructure = pd.DataFrame(station_rows)
        station_caps = {"S1":25.0, "S2":25.0}
        for count in (3, 5, 10):
            sessions = pd.DataFrame([{
                "session_id": f"V{count}-{i}", "arrival_epoch_min":0.0, "deadline_epoch_min":60.0,
                "energy_required_kwh":3.0, "sim_power_kw":7.2, "connector_type":"Type-2 AC",
                "station_id":f"S{(i % 2)+1}", "charger_id":f"preferred-{i}", "environment":"Test",
                "vehicle_reg_no":f"TEST-{i}", "max_ac_charge_power_kw":7.2, "max_dc_charge_power_kw":0.0,
            } for i in range(count)])
            for policy in ("LAFS", "EDF", "FCFS", "HYBRID"):
                results = run_single_algorithm(policy, sessions, infrastructure, station_caps,
                    step_min=1.0, facility_power_cap_kw=40.0)
                assert len({r.charger_id_assigned for r in results if r.timeline and r.timeline[0][0] == 0}) == count
                assert all(r.charge_start_min == 0 for r in results), f"{count}/{policy} serialized a free charger"
                by_minute: dict[float, float] = {}
                for result in results:
                    for minute, _, kw in result.timeline:
                        by_minute[minute] = by_minute.get(minute, 0) + kw
                assert max(by_minute.values()) <= 40.0 + 1e-6
                for result in results:
                    assert result.charger_id_assigned in set(infrastructure["charger_id"])
                snapshot = snapshot_at(results, 0)
                assert len(snapshot["charging"]) == count
                assert snapshot["demand_kw"] <= 40.0 + 1e-6
            print(f"PASS {count} vehicles: concurrent assignments under 40 kW facility cap (LAFS/EDF/FCFS/HYBRID)")

        # Unavailable/incompatible points must be ignored and leave the request unassigned.
        blocked = infrastructure.iloc[[0]].copy()
        blocked["charger_status"] = "maintenance"
        blocked_result = run_single_algorithm("LAFS", pd.DataFrame([{
            "session_id":"BLOCKED", "arrival_epoch_min":0, "deadline_epoch_min":30,
            "energy_required_kwh":1, "sim_power_kw":7.2, "connector_type":"Type-2 AC",
            "station_id":"S1", "charger_id":"", "environment":"Test"}]), blocked,
            {"S1":25}, facility_power_cap_kw=25)[0]
        assert blocked_result.charger_id_assigned is None and blocked_result.energy_delivered_kwh == 0
        print("PASS unavailable charger remains unassigned")

        # Verify that a tight station cap does not strand the facility's shared headroom.
        redistribution = pd.DataFrame([
            {"station_id":"TIGHT", "charger_id":"T1", "connector_type":"Type-2 AC", "rated_power_kw":50,
             "charger_status":"available", "station_status":"available", "station_aggregate_power_cap_kw_scenario":1.0},
            {"station_id":"OPEN", "charger_id":"O1", "connector_type":"Type-2 AC", "rated_power_kw":50,
             "charger_status":"available", "station_status":"available", "station_aggregate_power_cap_kw_scenario":50.0},
        ])
        mixed_sessions = pd.DataFrame([
            {"session_id":"TIGHT-REQ", "arrival_epoch_min":0, "deadline_epoch_min":60, "energy_required_kwh":20,
             "sim_power_kw":50, "connector_type":"Type-2 AC", "station_id":"TIGHT", "charger_id":"", "environment":"Test"},
            {"session_id":"OPEN-REQ", "arrival_epoch_min":0, "deadline_epoch_min":60, "energy_required_kwh":20,
             "sim_power_kw":50, "connector_type":"Type-2 AC", "station_id":"OPEN", "charger_id":"", "environment":"Test"},
        ])
        for policy in ("LAFS", "EDF", "FCFS", "HYBRID"):
            mixed = run_single_algorithm(policy, mixed_sessions, redistribution,
                {"TIGHT":1.0, "OPEN":50.0}, facility_power_cap_kw=40.0)
            first_minute_power = sum(next(p for t0, _, p in row.timeline if t0 == 0) for row in mixed)
            assert abs(first_minute_power - 40.0) < 1e-6, f"{policy} stranded site headroom: {first_minute_power} kW"
        print("PASS site headroom redistributes around a binding station cap")

        # Sensitivity: pure arrival-age emphasis favors the older request; pure EDF
        # emphasis favors the earlier deadline. Both are valid normalized weight sets.
        sensitivity = [Session("OLDER", 0, 200, 10, 7, "Type-2 AC", "Test", "S1", ""),
                       Session("URGENT", 10, 50, 10, 7, "Type-2 AC", "Test", "S1", "")]
        fcfs_rank = [x["session"].session_id for x in _hybrid_priority_rows(
            sensitivity, 20, {}, {"FCFS": 1.0, "EDF": 0.0, "FAIRNESS": 0.0})]
        edf_rank = [x["session"].session_id for x in _hybrid_priority_rows(
            sensitivity, 20, {}, {"FCFS": 0.0, "EDF": 1.0, "FAIRNESS": 0.0})]
        assert fcfs_rank[0] == "OLDER" and edf_rank[0] == "URGENT"
        assert validate_hybrid_weights({"FCFS": .2, "EDF": .3, "FAIRNESS": .5})
        print("PASS Hybrid weight sensitivity: FCFS-heavy favors wait age; EDF-heavy favors deadline")

        # Academic demo configuration: five vehicles, four endpoint ratings, 250 kW grid.
        ratings = [150.0, 150.0, 75.0, 75.0]
        five_four = pd.DataFrame([{
            "station_id":"HUB", "charger_id":f"CP0{i+1}", "connector_type":"CCS2 DC",
            "rated_power_kw":rating, "charger_status":"available", "station_status":"available",
            "station_aggregate_power_cap_kw_scenario":450.0,
        } for i, rating in enumerate(ratings)])
        five = pd.DataFrame([{
            "session_id":f"HUB-EV-{i+1}", "arrival_epoch_min":0.0, "deadline_epoch_min":120.0,
            "energy_required_kwh":60.0, "sim_power_kw":200.0, "connector_type":"CCS2 DC",
            "station_id":"HUB", "charger_id":"", "environment":"Demo",
            "vehicle_reg_no":f"DEMO-{i+1}", "battery_capacity_kwh_verified":100.0,
            "soc_arrival_pct":20.0, "soc_target_pct":80.0,
            "max_ac_charge_power_kw":11.0, "max_dc_charge_power_kw":200.0,
        } for i in range(5)])
        five_results = run_single_algorithm("HYBRID", five, five_four, {"HUB":450.0},
                                            step_min=1.0, facility_power_cap_kw=250.0)
        started_at_zero = [r for r in five_results if r.charge_start_min == 0]
        assert len(started_at_zero) == 4
        assert len({r.charger_id_assigned for r in started_at_zero}) == 4
        assert all(r.charger_id_assigned is not None for r in five_results)
        ratings_by_id = dict(zip(five_four.charger_id, five_four.rated_power_kw))
        endpoint_events: dict[str, list[tuple[float,float]]] = {}
        site_events: dict[float, float] = {}
        for result in five_results:
            endpoint_events.setdefault(result.charger_id_assigned, [])
            assert result.allocated_power_peak_kw <= ratings_by_id[result.charger_id_assigned] + 1e-8
            assert result.allocated_power_peak_kw <= result.sim_power_kw + 1e-8
            computed_energy = sum(p * (t1-t0) / 60 * 0.90 for t0,t1,p in result.timeline)
            assert abs(computed_energy-result.energy_delivered_kwh) < 0.0002
            assert result.energy_delivered_kwh <= result.requested_energy_kwh + 1e-8
            assert abs(result.soc_end_pct - min(80.0,20.0+result.energy_delivered_kwh)) < 0.02
            for t0,t1,p in result.timeline:
                assert result.arrival_min <= t0 + 1e-9 and t1 <= result.deadline_min + 1e-9
                endpoint_events[result.charger_id_assigned].append((t0,t1))
                site_events[t0] = site_events.get(t0,0)+p
                site_events[t1] = site_events.get(t1,0)-p
        for intervals in endpoint_events.values():
            ordered = sorted(intervals)
            assert all(ordered[i][1] <= ordered[i+1][0]+1e-9 for i in range(len(ordered)-1))
        running = peak = 0.0
        for time in sorted(site_events):
            running += site_events[time]
            peak = max(peak,running)
        assert peak <= 250.0 + 1e-8
        print(f"PASS 5 vehicles / 4 chargers / 250 kW grid: peak {peak:.2f} kW, endpoint and energy limits hold")

        # Off-grid arrival/departure boundaries are honored exactly within a nominal 1-minute step.
        off_grid = pd.DataFrame([{"session_id":"OFF-GRID", "arrival_epoch_min":0.5,
            "deadline_epoch_min":1.25, "energy_required_kwh":10.0, "sim_power_kw":7.2,
            "connector_type":"Type-2 AC", "station_id":"S1", "charger_id":"CP1",
            "environment":"Test", "max_ac_charge_power_kw":7.2}])
        off_result = run_single_algorithm("HYBRID", off_grid, infrastructure.iloc[[0]],
            {"S1":25.0}, step_min=1.0, facility_power_cap_kw=25.0)[0]
        assert off_result.timeline and off_result.timeline[0][0] >= 0.5
        assert off_result.timeline[-1][1] <= 1.25 + 1e-9
        assert abs(off_result.charging_duration_min - 0.75) < 1e-9
        print("PASS off-grid arrival/departure are clipped to exact simulation times")

        # Empty requests and an empty endpoint inventory are honest zero-result cases.
        empty_results = run_single_algorithm("HYBRID", pd.DataFrame(), five_four)
        assert empty_results == []
        no_ports = five_four.iloc[0:0].copy()
        no_port_result = run_single_algorithm("HYBRID", five.iloc[:1], no_ports)[0]
        assert no_port_result.charger_id_assigned is None and no_port_result.energy_delivered_kwh == 0
        empty_summary = compute_summary(results_to_dataframe({}), {}, stations_df=five_four)
        assert empty_summary.empty and not empty_summary.isna().any().any()
        print("PASS empty queue, zero available endpoints and empty analytics return no NaN/runtime errors")

        # The vehicle request power is not the allocated power: actual allocation is derived from timeline.
        five_df = results_to_dataframe({"HYBRID": five_results})
        assert (five_df["Peak Allocated Power (kW)"] <= five_df["Vehicle Max Power (kW)"]).all()
        assert all(row["Peak Allocated Power (kW)"] <= ratings_by_id[row["Charger (Assigned)"]] + 1e-8
                   for _, row in five_df.iterrows())
        print("PASS reported allocated power is the actual peak timeline allocation, not vehicle request limit")

        one_policy_summary = compute_summary(five_df, {"HYBRID": five_results}, stations_df=five_four)
        assert one_policy_summary["Algorithm"].tolist() == ["HYBRID"]
        assert abs(float(one_policy_summary.iloc[0]["Peak Facility Demand (kW)"]) - 250.0) < 1e-6
        energy_fig = fig_energy_comparison(five_df)
        complete_fig = fig_completion_rate(one_policy_summary)
        wait_fig = fig_waiting_time(five_df)
        load_fig = fig_station_load({"HYBRID": five_results}, power_cap_kw=250.0)
        assert list(energy_fig.data[0].x) == ["HYBRID"]
        assert list(complete_fig.data[0].x) == ["HYBRID"]
        assert set(wait_fig.data[0].x) == {"HYBRID"}
        assert len(load_fig.data) == 1 and max(load_fig.data[0].y) <= 250.0 + 1e-6
        print("PASS operational charts show only executed policy; analytics demand matches schedule timeline")

        invalid_analytics = pd.DataFrame([
            {"Algorithm":"FCFS", "Completion (%)":0.0, "Completed":False, "Feasible":True,
             "Started":True, "Requested (kWh)":10.0, "Delivered (kWh)":5.0, "Missed (kWh)":5.0,
             "Waiting (min)":float("inf")},
            {"Algorithm":"FCFS", "Completion (%)":100.0, "Completed":True, "Feasible":True,
             "Started":True, "Requested (kWh)":10.0, "Delivered (kWh)":10.0, "Missed (kWh)":0.0,
             "Waiting (min)":float("nan")},
        ])
        clean_metrics = compute_summary(invalid_analytics)
        metric_row = clean_metrics.iloc[0]
        assert metric_row["Completion Rate (%)"] == 50.0
        assert metric_row["Energy Fulfilment by Deadline (%)"] == 75.0
        assert metric_row["Mean Wait (min)"] is None
        assert all(not isinstance(value, float) or math.isfinite(value)
                   for value in metric_row.tolist() if value is not None)
        print("PASS analytics ignores NaN/infinite inputs and separates vehicle completion from energy fulfilment")

        dirty_chart_data = pd.DataFrame([
            {"Algorithm":"FCFS", "Requested (kWh)":10.0, "Delivered (kWh)":float("inf"),
             "Missed (kWh)":float("nan"), "Waiting (min)":float("inf"), "Completed":False,
             "Completion (%)":float("nan")},
            {"Algorithm":"EDF", "Requested (kWh)":5.0, "Delivered (kWh)":2.0,
             "Missed (kWh)":3.0, "Waiting (min)":float("nan"), "Completed":True,
             "Completion (%)":100.0},
        ])
        charts = [fig_energy_comparison(dirty_chart_data), fig_completion_rate(dirty_chart_data),
                  fig_waiting_time(dirty_chart_data), fig_missed_energy(dirty_chart_data)]
        assert all(np.isfinite(np.asarray(trace.y, dtype=float)).all()
                   for chart in charts for trace in chart.data if trace.y is not None)
        invalid_radar = pd.DataFrame([{"Algorithm":"FCFS", "Completion Rate (%)":float("inf"),
            "Avg Completion (%)":50.0, "Mean Wait (min)":10.0, "Total Missed (kWh)":1.0}])
        radar = fig_radar_summary(invalid_radar)
        assert not radar.data and any("undefined" in str(a.text).lower() for a in radar.layout.annotations)
        print("PASS analytics charts sanitize invalid values and omit undefined radar comparisons")

        latest_results = run_single_algorithm("HYBRID", pd.DataFrame([{
            "session_id":"PERSIST", "arrival_epoch_min":0, "deadline_epoch_min":30,
            "energy_required_kwh":1, "sim_power_kw":7.2, "connector_type":"Type-2 AC",
            "station_id":"S1", "charger_id":"", "environment":"Test"}]), infrastructure,
            station_caps, facility_power_cap_kw=40)
        schedule_id = save_schedule("west-yard", "HYBRID", 40, pd.Timestamp("2026-01-01"), latest_results, db)
        restored = load_latest_schedule("west-yard", db)
        assert restored and restored["id"] == schedule_id and restored["results"][0].charger_id_assigned
        assert not audit_history("west-yard", db).empty
        save_configuration(f1, s1, c1, db)
        assert load_latest_schedule("west-yard", db) is None, "stale schedule remained current after config edit"
        print("PASS facility configuration edits invalidate the current simulation snapshot")
        archive_facility("west-yard", db)
        assert get_facility("west-yard", db)["status"] == "archived"
        assert len(list_facilities(db, include_archived=True)) == 2
        print("PASS facility isolation, persistence, schedule restoration, audit and archive")

        # Reset demo/operational records without deleting facility configuration.
        reset_operational_data("east-depot", db)
        assert get_facility("east-depot", db)["grid_capacity_kw"] == 35
        assert get_station_frame("east-depot", db).shape[0] == 1
        assert load_requests("east-depot", db) == []
        print("PASS operational reset clears local records and preserves facility/charger configuration")


if __name__ == "__main__":
    run()
