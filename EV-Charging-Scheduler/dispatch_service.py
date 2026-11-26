"""Pure live-dispatch orchestration and idempotency keys for VoltFlow."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from data_loader import is_connector_compatible, live_queue_to_sessions_df, vehicle_power_limit
from scheduler import ALGORITHMS, SimResult, run_single_algorithm, validate_hybrid_weights

TERMINAL_STATUSES = {"completed", "cancelled", "failed", "duplicate review"}
_KEY_FIELDS = (
    "request_id", "session_id", "vehicle_reg_no", "station_id", "vehicle_model_id",
    "connector_type", "arrival_time_IST", "departure_time_IST", "energy_required_kwh",
    "max_charge_power_kw", "battery_capacity_kwh", "soc_arrival_pct", "soc_target_pct",
    "veh_max_ac_kw", "veh_max_dc_kw",
)


def dispatchable_requests(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active requests in their current queue order."""
    return [q for q in queue if str(q.get("status", "Queued")).strip().lower() not in TERMINAL_STATUSES]


def dispatch_key(queue: list[dict[str, Any]], stations: pd.DataFrame,
                 facility_power_cap_kw: float, algorithm: str, step_min: float,
                 hybrid_weights: dict[str, float] | None = None) -> str:
    """Stable key for the scheduling inputs; progress/status fields are excluded."""
    # Completed/failed requests remain part of the schedule snapshot during replay.
    # Keep them in its fingerprint so a restart at a later simulation minute does
    # not needlessly rebuild the remaining sessions and reset their progress.
    schedule_records = [q for q in queue if str(q.get("status", "Queued")).strip().lower()
                        not in {"cancelled", "duplicate review"}]
    rows = [{key: q.get(key) for key in _KEY_FIELDS} for q in schedule_records]
    infrastructure_fields = [
        "station_id", "charger_id", "connector_type", "rated_power_kw", "charger_status",
        "station_status", "station_aggregate_power_cap_kw_scenario",
    ]
    infra = []
    if stations is not None and not stations.empty:
        for _, row in stations.iterrows():
            infra.append({key: row.get(key) for key in infrastructure_fields})
    infra.sort(key=lambda r: (str(r.get("station_id")), str(r.get("charger_id"))))
    weights = validate_hybrid_weights(hybrid_weights) if algorithm == "HYBRID" else None
    payload = {
        "requests": rows,
        "infrastructure": infra,
        "facility_power_cap_kw": float(facility_power_cap_kw),
        "algorithm": str(algorithm).upper(),
        "step_min": float(step_min),
        "hybrid_weights": weights,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calculate_live_dispatch(queue: list[dict[str, Any]], stations: pd.DataFrame,
                            facility_power_cap_kw: float, algorithm: str = "HYBRID",
                            step_min: float = 1.0,
                            hybrid_weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Run one validated schedule for active requests without mutating queue/state."""
    algorithm = str(algorithm).upper()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported scheduling policy: {algorithm}")
    if not math.isfinite(float(facility_power_cap_kw)) or facility_power_cap_kw <= 0:
        raise ValueError("Facility power capacity must be finite and positive.")
    active = dispatchable_requests(queue)
    key = dispatch_key(queue, stations, facility_power_cap_kw, algorithm, step_min, hybrid_weights)
    if not active:
        return {"dispatch_key": key, "results": [], "sessions_df": pd.DataFrame(),
                "epoch_anchor": None, "algorithm": algorithm}

    sessions_df = live_queue_to_sessions_df(active, facility_power_cap_kw)
    station_caps = {
        str(station_id): float(group["station_aggregate_power_cap_kw_scenario"].iloc[0])
        for station_id, group in stations.groupby("station_id")
    }
    results = run_single_algorithm(
        algorithm=algorithm,
        sessions_df=sessions_df,
        stations_df=stations,
        station_power_cap_kw=station_caps,
        step_min=step_min,
        facility_power_cap_kw=facility_power_cap_kw,
        hybrid_weights=hybrid_weights if algorithm == "HYBRID" else None,
    )
    _validate_dispatch(results, active, stations, facility_power_cap_kw)
    return {"dispatch_key": key, "results": results, "sessions_df": sessions_df,
            "epoch_anchor": sessions_df["epoch_anchor_IST"].iloc[0], "algorithm": algorithm}


def _validate_dispatch(results: list[SimResult], queue: list[dict[str, Any]],
                       stations: pd.DataFrame, facility_power_cap_kw: float) -> None:
    request_by_session = {str(q["session_id"]): q for q in queue}
    expected = set(request_by_session)
    actual = [str(r.session_id) for r in results]
    if len(actual) != len(set(actual)):
        raise ValueError("Dispatch returned duplicate session IDs; results were not persisted.")
    if set(actual) != expected:
        raise ValueError("Dispatch result IDs do not match the active request sessions.")

    chargers = {str(row.charger_id): row for row in stations.itertuples(index=False)}
    by_charger: dict[str, list[tuple[float, float, float, str, str]]] = {}
    for result in results:
        if not result.charger_id_assigned:
            if result.timeline:
                raise ValueError(f"Session {result.session_id} has charging power without a charger assignment.")
            continue
        charger = chargers.get(str(result.charger_id_assigned))
        if charger is None or str(charger.station_id) != result.station_id:
            raise ValueError(f"Session {result.session_id} is assigned outside its configured station.")
        if str(getattr(charger, "charger_status", "available")).lower() != "available" \
                or str(getattr(charger, "station_status", "available")).lower() != "available":
            raise ValueError(f"Session {result.session_id} is assigned to an unavailable charger.")
        if not is_connector_compatible(result.connector_type, str(charger.connector_type)):
            raise ValueError(f"Session {result.session_id} is assigned an incompatible connector.")
        request = request_by_session[result.session_id]
        vehicle_limit = vehicle_power_limit(
            float(request.get("max_charge_power_kw") or result.sim_power_kw),
            float(request.get("veh_max_ac_kw") or 0.0),
            float(request.get("veh_max_dc_kw") or 0.0),
            str(charger.connector_type),
        )
        limit = min(float(charger.rated_power_kw), float(vehicle_limit))
        for start, end, power in result.timeline:
            if start < result.arrival_min - 1e-8 or end > result.deadline_min + 1e-8:
                raise ValueError(f"Session {result.session_id} charges outside its arrival/deadline window.")
            if power > limit + 1e-7:
                raise ValueError(f"Session {result.session_id} exceeds its vehicle or charger rating.")
            by_charger.setdefault(str(result.charger_id_assigned), []).append(
                (float(start), float(end), float(power), result.station_id, result.session_id))

    all_events = sorted({time for intervals in by_charger.values() for start, end, _, _, _ in intervals
                         for time in (start, end)})
    station_caps = {str(sid): float(group["station_aggregate_power_cap_kw_scenario"].iloc[0])
                    for sid, group in stations.groupby("station_id")}
    for left, right in zip(all_events, all_events[1:]):
        if right <= left:
            continue
        moment = (left + right) / 2.0
        active_intervals = [item for intervals in by_charger.values() for item in intervals
                            if item[0] <= moment < item[1]]
        if len([item[4] for item in active_intervals]) != len(set(item[4] for item in active_intervals)):
            raise ValueError("A session occupies more than one charger simultaneously.")
        seen_ports: set[str] = set()
        for port, intervals in by_charger.items():
            for start, end, _, _, sid in intervals:
                if start <= moment < end:
                    if port in seen_ports:
                        raise ValueError(f"Charger {port} serves multiple vehicles simultaneously.")
                    seen_ports.add(port)
        facility_load = sum(item[2] for item in active_intervals)
        if facility_load > facility_power_cap_kw + 1e-6:
            raise ValueError("Dispatch exceeds the facility electrical power limit.")
        station_loads: dict[str, float] = {}
        for item in active_intervals:
            station_loads[item[3]] = station_loads.get(item[3], 0.0) + item[2]
        for station_id, load in station_loads.items():
            if load > station_caps.get(station_id, 0.0) + 1e-6:
                raise ValueError(f"Dispatch exceeds station {station_id}'s aggregate power limit.")
