"""
scheduler.py
============
EV Charging Scheduling Engine — FCFS, EDF, LAFS
Enforces vehicle-side charging limits, charger rated power, connector compatibility,
charger mutual exclusion, and aggregate station power caps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import pandas as pd
from data_loader import vehicle_power_limit, calculate_power_limit

CHARGE_EFFICIENCY = 0.90   # 90% theoretical charging efficiency
TEST_COUNT = 31            # regression groups plus hybrid, input-validation and preview checks
HYBRID_DEFAULT_WEIGHTS = {"FCFS": 1 / 3, "EDF": 1 / 3, "FAIRNESS": 1 / 3}
ALGORITHMS = ("FCFS", "EDF", "LAFS", "HYBRID")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One EV charging session for simulation."""
    session_id: str
    arrival_min: float          # epoch minutes (from connection_time_IST)
    deadline_min: float         # epoch minutes (from departure_time_IST)
    requested_energy_kwh: float # from energy_required_kwh
    sim_power_kw: float         # session power limit (kW)
    connector_type: str         # connector required by vehicle/session
    environment: str
    station_id: str
    preferred_charger_id: str   # advisory scenario charger assignment
    # Reference / vehicle specification fields
    vehicle_reg_no: str = ""
    manufacturer: str = ""
    model_variant: str = ""
    battery_kwh_verified: float = 0.0
    soc_arrival_pct: float = 0.0
    soc_target_pct: float = 0.0
    vehicle_max_kw: float = 0.0
    max_ac_charge_power_kw: float = 0.0
    max_dc_charge_power_kw: float = 0.0
    compatibility_caveat: bool = False
    model_id: str = ""
    theoretically_feasible: bool = True
    request_id: str = ""

    @property
    def window_min(self) -> float:
        return max(0.0, self.deadline_min - self.arrival_min)

    @property
    def duration_needed_min(self) -> float:
        if self.sim_power_kw <= 0:
            return float("inf")
        return (self.requested_energy_kwh / self.sim_power_kw / CHARGE_EFFICIENCY) * 60

    def __lt__(self, other: "Session") -> bool:
        return self.arrival_min < other.arrival_min


@dataclass
class ChargePoint:
    """A single charging endpoint at a station."""
    cp_id: str
    station_id: str
    connector_type: str
    rated_power_kw: float
    # Runtime state (reset between algorithm runs)
    is_busy: bool = False
    serving_session_id: Optional[str] = None


@dataclass
class SimResult:
    """Result for one session under one algorithm."""
    session_id: str
    algorithm: str
    environment: str
    station_id: str
    charger_id_dataset: str
    charger_id_assigned: Optional[str]
    vehicle_reg_no: str
    manufacturer: str
    model_variant: str
    battery_kwh_verified: float
    soc_arrival_pct: float
    soc_target_pct: float
    connector_type: str
    requested_energy_kwh: float
    energy_delivered_kwh: float
    missed_energy_kwh: float
    energy_completion_pct: float
    completed_by_deadline: bool
    waiting_time_min: Optional[float]   # None = session never started charging
    charge_start_min: Optional[float]
    charge_end_min: Optional[float]
    deadline_min: float
    arrival_min: float
    sim_power_kw: float
    compatibility_caveat: bool
    model_id: str
    theoretically_feasible: bool
    timeline: list[tuple[float, float, float]] = field(default_factory=list)
    started: bool = False   # True once any energy was delivered
    request_id: str = ""

    @property
    def allocated_power_peak_kw(self) -> float:
        """Highest actual kW allocation recorded in the dispatch timeline."""
        return max((float(power) for _, _, power in self.timeline), default=0.0)

    @property
    def charging_duration_min(self) -> float:
        """Total energized minutes (not elapsed time including queue wait)."""
        return sum(max(0.0, float(end) - float(start)) for start, end, _ in self.timeline)

    @property
    def soc_end_pct(self) -> Optional[float]:
        if self.battery_kwh_verified <= 0:
            return None
        end = self.soc_arrival_pct + 100.0 * self.energy_delivered_kwh / self.battery_kwh_verified
        return min(100.0, end)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_sessions(df: pd.DataFrame) -> list[Session]:
    """Convert cleaned sessions DataFrame into Session objects."""
    if df is None or df.empty:
        return []
    if df["session_id"].duplicated().any():
        dups = sorted(df.loc[df["session_id"].duplicated(), "session_id"].astype(str).unique())
        raise ValueError(f"Duplicate session_id values would corrupt the simulation: {dups}")
    if "request_id" in df.columns:
        request_ids = df["request_id"].fillna("").astype(str).str.strip()
        request_ids = request_ids[request_ids.ne("")]
        if request_ids.duplicated().any():
            dups = sorted(request_ids[request_ids.duplicated()].unique().tolist())
            raise ValueError(f"Duplicate request_id values would corrupt the simulation: {dups}")
        session_ids = df["session_id"].fillna("").astype(str).str.strip()
        conflated = sorted(set(request_ids) & set(session_ids))
        if conflated:
            raise ValueError(f"Request and session IDs must be distinct: {conflated}")
    def number(row, keys, default=None, *, label="value"):
        for key in keys:
            value = row.get(key)
            if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {label}: expected a finite number.") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"Invalid {label}: expected a finite number.")
            return parsed
        if default is not None:
            return float(default)
        raise ValueError(f"Missing {label}.")

    sessions = []
    for _, row in df.iterrows():
        model_id_val = str(row.get("vehicle_model_id") or row.get("model_id") or "")
        sid = str(row.get("session_id") or "").strip()
        energy = number(row, ("energy_required_kwh", "requested_energy_kwh"), label=f"energy for {sid}")
        power = number(row, ("sim_power_kw", "charging_power_limit_kw", "effective_power_kw"), label=f"power for {sid}")
        arrival = number(row, ("arrival_epoch_min",), default=0.0, label=f"arrival for {sid}")
        deadline = number(row, ("deadline_epoch_min",), default=0.0, label=f"deadline for {sid}")
        max_ac = number(row, ("max_ac_charge_power_kw",), default=0.0, label=f"AC limit for {sid}")
        max_dc = number(row, ("max_dc_charge_power_kw",), default=0.0, label=f"DC limit for {sid}")
        raw_request_id = row.get("request_id", "")
        request_id = "" if raw_request_id is None or pd.isna(raw_request_id) else str(raw_request_id).strip()
        if not sid:
            raise ValueError("Every session must have a non-empty session_id.")
        if energy <= 0 or power <= 0:
            raise ValueError(f"Session {sid} must have positive requested energy and charging power.")
        if deadline <= arrival:
            raise ValueError(f"Session {sid} must have a deadline after its arrival.")
        if max_ac < 0 or max_dc < 0:
            raise ValueError(f"Session {sid} cannot have a negative AC/DC power limit.")
        station_id = str(row.get("station_id") or "").strip()
        connector = str(row.get("connector_type") or "").strip()
        if not station_id or not connector:
            raise ValueError(f"Session {sid} must specify a station and connector type.")

        s = Session(
            session_id=sid,
            vehicle_reg_no=str(row.get("vehicle_reg_no") or row.get("session_id") or ""),
            arrival_min=arrival,
            deadline_min=deadline,
            requested_energy_kwh=energy,
            sim_power_kw=power,
            connector_type=connector,
            environment=str(row.get("environment") or ""),
            station_id=station_id,
            preferred_charger_id=str(row.get("charger_id") or ""),
            manufacturer=str(row.get("manufacturer") or ""),
            model_variant=str(row.get("model_variant") or ""),
            battery_kwh_verified=float(row.get("battery_capacity_kwh_verified") or 0.0),
            soc_arrival_pct=float(row.get("soc_arrival_pct") or 0.0),
            soc_target_pct=float(row.get("soc_target_pct") or 0.0),
            vehicle_max_kw=float(row.get("vehicle_max_kw") or 0.0),
            max_ac_charge_power_kw=max_ac,
            max_dc_charge_power_kw=max_dc,
            compatibility_caveat=bool(row.get("compatibility_caveat") or False),
            model_id=model_id_val,
            theoretically_feasible=bool(row.get("theoretically_feasible", True)),
            request_id=request_id,
        )
        sessions.append(s)
    return sessions


def build_charge_points(stations_df: pd.DataFrame) -> list[ChargePoint]:
    """Build ChargePoint objects from Station_Configs."""
    if stations_df is None or stations_df.empty:
        return []
    cps = []
    seen_ids: set[str] = set()
    for _, row in stations_df.iterrows():
        if str(row.get("charger_status", "available")).lower() != "available":
            continue
        if str(row.get("station_status", "available")).lower() != "available":
            continue
        cp_id = str(row.get("charger_id") or "").strip()
        station_id = str(row.get("station_id") or "").strip()
        connector = str(row.get("connector_type") or "").strip()
        try:
            rated_power = float(row.get("rated_power_kw"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Charger {cp_id or '<unnamed>'} must have a finite positive power rating.") from exc
        if not cp_id or not station_id or not connector or not math.isfinite(rated_power) or rated_power <= 0:
            raise ValueError(f"Charger {cp_id or '<unnamed>'} must have an ID, station, connector, and finite positive rating.")
        if cp_id in seen_ids:
            raise ValueError(f"Duplicate charger ID in facility configuration: {cp_id}")
        seen_ids.add(cp_id)
        cps.append(ChargePoint(
            cp_id=cp_id,
            station_id=station_id,
            connector_type=connector,
            rated_power_kw=rated_power,
        ))
    return cps


# ---------------------------------------------------------------------------
# Charger allocation logic
# ---------------------------------------------------------------------------

def _find_charger_for_session(
    session: Session,
    charge_points: list[ChargePoint],
    t: float,
) -> Optional[ChargePoint]:
    """
    Find an available and compatible charge point at the same station.
    Enforces mutual exclusion. Prefers pre-assigned charger if free.
    """
    from data_loader import is_connector_compatible

    candidates = [
        cp for cp in charge_points
        if cp.station_id == session.station_id
        and not cp.is_busy
        and is_connector_compatible(session.connector_type, cp.connector_type)
    ]
    if not candidates:
        return None

    # Try preferred charger first
    for cp in candidates:
        if cp.cp_id == session.preferred_charger_id:
            return cp

    return candidates[0]


def validate_hybrid_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    """Validate FCFS/EDF/fairness weights; require finite nonnegative values summing to 1."""
    import math
    values = dict(HYBRID_DEFAULT_WEIGHTS if weights is None else weights)
    keys = ("FCFS", "EDF", "FAIRNESS")
    if set(values) != set(keys):
        raise ValueError("Hybrid weights must define exactly FCFS, EDF, and FAIRNESS.")
    normalized = {key: float(values[key]) for key in keys}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError("Hybrid weights must be finite, nonnegative numbers.")
    if abs(sum(normalized.values()) - 1.0) > 1e-9:
        raise ValueError("Hybrid weights must sum to 1. Adjust the three weights to total 100%.")
    return normalized


def _hybrid_priority_rows(
    sessions: list[Session], now_min: float, energy_delivered: dict[str, float],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score the current waiting queue using comparable, documented 0–1 components."""
    w = validate_hybrid_weights(weights)
    waiting = [s for s in sessions if s.arrival_min <= now_min < s.deadline_min
               and energy_delivered.get(s.session_id, 0.0) < s.requested_energy_kwh - 1e-9]
    ages = {s.session_id: max(0.0, now_min - s.arrival_min) for s in waiting}
    max_age = max(ages.values(), default=0.0)
    slacks = {s.session_id: max(0.0, s.deadline_min - now_min) for s in waiting}
    min_slack, max_slack = (min(slacks.values()), max(slacks.values())) if slacks else (0.0, 0.0)
    rows = []
    for s in waiting:
        progress = min(1.0, max(0.0, energy_delivered.get(s.session_id, 0.0) /
                                  max(s.requested_energy_kwh, 1e-9)))
        fcfs = ages[s.session_id] / max_age if max_age > 0 else 0.0
        # Earlier remaining deadline gets larger urgency; tied deadlines remain tied.
        edf = 1.0 - (slacks[s.session_id] - min_slack) / (max_slack - min_slack) if max_slack > min_slack else 1.0
        # Use accumulated service debt so waiting requests do not all tie at 1.0
        # merely because none has started receiving energy yet.
        elapsed_fraction = min(1.0, max(0.0, ages[s.session_id] / max(s.window_min, 1e-9)))
        fairness = min(1.0, max(0.0, elapsed_fraction - progress))
        score = w["FCFS"] * fcfs + w["EDF"] * edf + w["FAIRNESS"] * fairness
        reasons = []
        if fcfs > 0.5: reasons.append("longer accumulated wait")
        if edf > 0.5: reasons.append("earlier/urgent deadline")
        if fairness > 0.5: reasons.append("larger energy-progress deficit")
        if not reasons: reasons.append("balanced component scores; deterministic deadline/arrival/ID tie-break")
        rows.append({"session": s, "waiting_age": ages[s.session_id], "fcfs": fcfs,
                     "edf": edf, "fairness": fairness, "score": score,
                     "reason": "; ".join(reasons)})
    return sorted(rows, key=lambda r: (-r["score"], r["session"].deadline_min,
                                       r["session"].arrival_min, r["session"].session_id))


def build_hybrid_priority_list(
    sessions_df: pd.DataFrame, stations_df: pd.DataFrame, now_min: float | None = None,
    energy_delivered: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return an explainable ranked view of requests at the next dispatch evaluation."""
    from data_loader import is_connector_compatible
    sessions = build_sessions(sessions_df)
    if not sessions:
        return pd.DataFrame()
    # A queue preview includes future arrivals rather than stopping at the earliest one.
    now = max(s.arrival_min for s in sessions) if now_min is None else float(now_min)
    delivered = energy_delivered or {}
    rows = _hybrid_priority_rows(sessions, now, delivered, weights)
    stations = build_charge_points(stations_df)
    output = []
    for rank, item in enumerate(rows, 1):
        s = item["session"]
        compatible = [cp for cp in stations if cp.station_id == s.station_id and
                      is_connector_compatible(s.connector_type, cp.connector_type)]
        eligible = bool(compatible and any(cp.rated_power_kw > 0 for cp in compatible))
        output.append({"Priority rank": rank, "Vehicle/session ID": s.vehicle_reg_no or s.session_id,
            "Session ID": s.session_id, "Arrival (min)": s.arrival_min,
            "Waiting age (min)": item["waiting_age"], "Departure deadline (min)": s.deadline_min,
            "Required energy (kWh)": s.requested_energy_kwh,
            "Energy delivered (kWh)": float(delivered.get(s.session_id, 0.0)),
            "FCFS contribution": round(item["fcfs"], 4), "EDF contribution": round(item["edf"], 4),
            "Fairness contribution": round(item["fairness"], 4),
            "Combined priority score": round(item["score"], 4),
            "Compatible charger IDs": ", ".join(cp.cp_id for cp in compatible),
            "Eligible now": eligible,
            "Reason for ranking": item["reason"] if eligible else "No available compatible charger at this station; remains queued."})
    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

def _allocate_power(items: list, cap_kw: float, algorithm: str) -> dict[str, float]:
    """
    Split one station's power cap across the sessions currently connected to its chargers.
    `items` are [Session, ChargePoint, desired_kw]; recomputed every step so power is never
    "locked in" at whatever headroom happened to exist when a session connected.

    LAFS : equal-weight proportional sharing.
    FCFS : weighted max-min sharing, with earlier arrivals receiving higher weight.
    EDF  : weighted max-min sharing, with earlier deadlines receiving higher weight.

    Every connected vehicle receives power when the station cap is nonzero. FCFS/EDF
    priorities influence the share without leaving compatible free chargers idle just
    because the highest-priority vehicle alone could consume the entire cap.
    """
    if not items:
        return {}
    if algorithm == "FCFS":
        key = lambda it: (it[0].arrival_min, it[0].session_id)
    elif algorithm == "EDF":
        key = lambda it: (it[0].deadline_min, it[0].arrival_min, it[0].session_id)
    else:
        key = lambda it: (it[0].session_id,)

    ordered = sorted(items, key=key)
    # Earlier FCFS/EDF entries receive larger weights, while equal-weight LAFS
    # divides the available cap evenly relative to each vehicle's requested power.
    weights = {
        it[0].session_id: (1.0 if algorithm in ("LAFS", "HYBRID") else float(len(ordered) - idx))
        for idx, it in enumerate(ordered)
    }
    allocation = {it[0].session_id: 0.0 for it in ordered}
    remaining = max(0.0, float(cap_kw))
    unfilled = list(ordered)

    # Weighted max-min filling: honor each vehicle's power limit, redistribute any
    # unused share, and never allocate more than the station's remaining capacity.
    while unfilled and remaining > 1e-9:
        total_weight = sum(weights[it[0].session_id] for it in unfilled)
        if total_weight <= 0:
            break
        share_unit = remaining / total_weight
        saturated = [
            it for it in unfilled
            if it[2] - allocation[it[0].session_id] <= share_unit * weights[it[0].session_id] + 1e-9
        ]
        if not saturated:
            for it in unfilled:
                allocation[it[0].session_id] += share_unit * weights[it[0].session_id]
            remaining = 0.0
            break
        for it in saturated:
            sid = it[0].session_id
            used = max(0.0, it[2] - allocation[sid])
            allocation[sid] += used
            remaining -= used
            unfilled.remove(it)

    return allocation


def _simulate(
    algorithm: str,
    sessions: list[Session],
    charge_points: list[ChargePoint],
    station_power_cap_kw: float | dict[str, float],
    step_min: float = 1.0,
    facility_power_cap_kw: float | None = None,
    hybrid_weights: dict[str, float] | None = None,
) -> list[SimResult]:
    """
    Discrete-time simulation of scheduling algorithms:
    - FCFS: First Come, First Served
    - EDF: Earliest Deadline First
    - LAFS: Load-Aware Fair Share (proportional power throttling)
    - HYBRID: weighted FCFS-age/EDF-urgency/fairness queue rank plus LAFS equal-weight power sharing

    Guarantees:
    - No two EVs occupy the same charger simultaneously.
    - Power delivered respects min(vehicle limit, charger rating, station cap share).
    - Each station's cap is enforced separately and at every step.
    - Energy delivery stops strictly at session deadline.
    - A session that never receives energy has waiting_time_min = None (it is NOT counted as 0).
    """
    import math
    if not math.isfinite(float(step_min)) or step_min <= 0:
        raise ValueError("Simulation step must be a finite positive number of minutes.")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported scheduling algorithm: {algorithm}")
    if facility_power_cap_kw is not None and (
        not math.isfinite(float(facility_power_cap_kw)) or float(facility_power_cap_kw) < 0
    ):
        raise ValueError("Facility power cap must be finite and nonnegative.")
    if isinstance(station_power_cap_kw, dict):
        for station_id, cap in station_power_cap_kw.items():
            if not math.isfinite(float(cap)) or float(cap) < 0:
                raise ValueError(f"Station {station_id} power cap must be finite and nonnegative.")
    elif not math.isfinite(float(station_power_cap_kw)) or float(station_power_cap_kw) < 0:
        raise ValueError("Station power cap must be finite and nonnegative.")
    cp_ids = [cp.cp_id for cp in charge_points]
    if len(set(cp_ids)) != len(cp_ids):
        raise ValueError("Charger IDs must be unique within a facility simulation.")
    for cp in charge_points:
        cp.is_busy = False
        cp.serving_session_id = None

    sessions = sorted(sessions, key=lambda s: s.arrival_min)
    if not sessions:
        return []
    if len({s.session_id for s in sessions}) != len(sessions):
        raise ValueError("Duplicate session_id values in simulation input")

    def _get_cap(st_id: str) -> float:
        if isinstance(station_power_cap_kw, dict):
            # Missing configuration must fail closed rather than inventing 100 kW.
            value = float(station_power_cap_kw.get(st_id, 0.0))
        else:
            value = float(station_power_cap_kw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Station {st_id} power cap must be finite and nonnegative.")
        return value

    t_min = min(s.arrival_min for s in sessions)
    t_max = max(s.deadline_min for s in sessions)

    energy_delivered: dict[str, float] = {s.session_id: 0.0 for s in sessions}
    waiting_time:     dict[str, Optional[float]] = {s.session_id: None for s in sessions}
    charge_start:     dict[str, Optional[float]] = {s.session_id: None for s in sessions}
    charge_end:       dict[str, Optional[float]] = {s.session_id: None for s in sessions}
    assigned_cp:      dict[str, Optional[str]]   = {s.session_id: None for s in sessions}
    timelines:        dict[str, list] = {s.session_id: [] for s in sessions}

    from data_loader import vehicle_power_limit

    waiting_queue: list[Session] = []
    # active_charging: list of [Session, ChargePoint, desired_power_kw]
    active_charging: list[list] = []
    arrived_set: set[str] = set()

    t = t_min
    while t < t_max - 1e-9:
        # Split a nominal simulation step at arrivals and deadlines so no vehicle
        # receives power before arrival or after departure when times are off-grid.
        next_events = [x for s in sessions for x in (s.arrival_min, s.deadline_min) if x > t + 1e-9]
        interval_end = min([t + step_min, t_max, *next_events])
        interval_min = interval_end - t
        if interval_min <= 1e-9:
            break
        # 1. Admit arrivals
        for s in sessions:
            if s.session_id not in arrived_set and s.arrival_min <= t + 1e-9 and t < s.deadline_min - 1e-9:
                arrived_set.add(s.session_id)
                waiting_queue.append(s)

        # 2. Drop expired uncharged sessions from waiting queue
        waiting_queue = [s for s in waiting_queue if s.deadline_min > t]

        # 3. Release finished or deadline-expired sessions
        still_active = []
        for item in active_charging:
            s, cp = item[0], item[1]
            energy_rem = s.requested_energy_kwh - energy_delivered[s.session_id]
            if energy_rem <= 1e-6:
                cp.is_busy = False
                cp.serving_session_id = None
                if charge_end[s.session_id] is None:
                    charge_end[s.session_id] = t
            elif t >= s.deadline_min:
                cp.is_busy = False
                cp.serving_session_id = None
                charge_end[s.session_id] = min(t, s.deadline_min)
            else:
                still_active.append(item)
        active_charging = still_active

        # 4. Sort waiting queue by algorithm policy
        if algorithm == "FCFS":
            waiting_queue.sort(key=lambda s: (s.arrival_min, s.session_id))
        elif algorithm == "EDF":
            waiting_queue.sort(key=lambda s: (s.deadline_min, s.arrival_min, s.session_id))
        elif algorithm == "LAFS":
            # NOTE: sessions in this queue have not started, so their delivered fraction is always 0
            # and this ordering reduces to deadline order. LAFS differs from EDF only in how power
            # is shared once EVs are connected (step 6), not in who gets a free charger first.
            def _lafs_key(s: Session):
                delivered_frac = energy_delivered[s.session_id] / max(s.requested_energy_kwh, 1e-9)
                return (delivered_frac, s.deadline_min, s.arrival_min)
            waiting_queue.sort(key=_lafs_key)
        elif algorithm == "HYBRID":
            ranked = _hybrid_priority_rows(waiting_queue, t, energy_delivered, hybrid_weights)
            waiting_queue = [row["session"] for row in ranked]

        # 5. Connect waiting sessions to free, compatible chargers
        newly_started: set[str] = set()
        for s in waiting_queue:
            if facility_power_cap_kw is not None and facility_power_cap_kw <= 1e-9:
                continue
            if _get_cap(s.station_id) <= 1e-9:
                continue
            cp = _find_charger_for_session(s, charge_points, t)
            if cp is None:
                continue
            des_pwr = min(
                vehicle_power_limit(s.sim_power_kw, s.max_ac_charge_power_kw,
                                    s.max_dc_charge_power_kw, cp.connector_type),
                cp.rated_power_kw,
            )
            if des_pwr <= 0:
                continue
            cp.is_busy = True
            cp.serving_session_id = s.session_id
            if assigned_cp[s.session_id] is None:
                assigned_cp[s.session_id] = cp.cp_id
            active_charging.append([s, cp, des_pwr])
            newly_started.add(s.session_id)

        waiting_queue = [s for s in waiting_queue if s.session_id not in newly_started]

        # 6. Power allocation: apply station and site caps together, recomputed each step.
        active_by_station: dict[str, list] = {}
        for item in active_charging:
            active_by_station.setdefault(item[0].station_id, []).append(item)
        power_step_alloc: dict[str, float] = {}
        station_remaining = {sid: max(0.0, _get_cap(sid)) for sid in active_by_station}
        facility_remaining = float("inf") if facility_power_cap_kw is None else max(0.0, facility_power_cap_kw)
        pending_kw = {item[0].session_id: max(0.0, float(item[2])) for item in active_charging}
        # The second pass redistributes site headroom when a station-local cap binds.
        # A single global split followed by min(station allocation, global allocation)
        # could strand power at a capped station while another station still had room.
        while facility_remaining > 1e-9 and any(v > 1e-9 for v in pending_kw.values()):
            candidates = [[item[0], item[1], pending_kw[item[0].session_id]]
                          for item in active_charging if pending_kw[item[0].session_id] > 1e-9]
            if facility_power_cap_kw is None:
                global_alloc = {item[0].session_id: pending_kw[item[0].session_id] for item in candidates}
            else:
                share_policy = "LAFS" if algorithm == "HYBRID" else algorithm
                global_alloc = _allocate_power(candidates, facility_remaining, share_policy)
            step_alloc: dict[str, float] = {}
            for st_id in active_by_station:
                station_items = [[item[0], item[1], global_alloc.get(item[0].session_id, 0.0)]
                                 for item in active_by_station[st_id]
                                 if global_alloc.get(item[0].session_id, 0.0) > 1e-9]
                if station_items and station_remaining[st_id] > 1e-9:
                    share_policy = "LAFS" if algorithm == "HYBRID" else algorithm
                    step_alloc.update(_allocate_power(station_items, station_remaining[st_id], share_policy))
            used = sum(step_alloc.values())
            if used <= 1e-9:
                break
            for sid, kw in step_alloc.items():
                power_step_alloc[sid] = power_step_alloc.get(sid, 0.0) + kw
                pending_kw[sid] = max(0.0, pending_kw.get(sid, 0.0) - kw)
            for st_id, items in active_by_station.items():
                station_remaining[st_id] = max(0.0, station_remaining[st_id] - sum(step_alloc.get(item[0].session_id, 0.0) for item in items))
            if facility_power_cap_kw is not None:
                facility_remaining = max(0.0, facility_remaining - used)

        # 7. Deliver energy for this step
        for item in active_charging:
            s = item[0]
            pwr = power_step_alloc.get(s.session_id, 0.0)
            energy_rem = s.requested_energy_kwh - energy_delivered[s.session_id]
            if energy_rem <= 0 or t >= s.deadline_min or pwr <= 0:
                continue
            if charge_start[s.session_id] is None:
                charge_start[s.session_id] = t
                waiting_time[s.session_id] = t - s.arrival_min
            energy_step = min(pwr * (interval_min / 60.0) * CHARGE_EFFICIENCY, energy_rem)
            energy_delivered[s.session_id] += energy_step
            actual_min = min(interval_min, energy_step / max(pwr * CHARGE_EFFICIENCY, 1e-12) * 60.0)
            timelines[s.session_id].append((t, t + actual_min, pwr))
            if energy_delivered[s.session_id] >= s.requested_energy_kwh - 1e-9:
                charge_end[s.session_id] = t + actual_min

        t = interval_end

    # Finalize any sessions still active when simulation terminates
    for item in active_charging:
        s, cp = item[0], item[1]
        cp.is_busy = False
        cp.serving_session_id = None
        if charge_end[s.session_id] is None:
            charge_end[s.session_id] = min(t_max, s.deadline_min)

    # Compile results
    results = []
    for s in sessions:
        delivered = round(energy_delivered[s.session_id], 4)
        missed = round(max(0.0, s.requested_energy_kwh - delivered), 4)
        pct = round(100.0 * delivered / max(s.requested_energy_kwh, 1e-9), 2)
        completed = pct >= 99.5

        c_start = charge_start[s.session_id]
        c_end = charge_end[s.session_id]
        if timelines[s.session_id]:
            if c_start is None:
                c_start = timelines[s.session_id][0][0]
            if c_end is None:
                c_end = timelines[s.session_id][-1][1]
        wait = waiting_time[s.session_id]

        results.append(SimResult(
            session_id=s.session_id,
            algorithm=algorithm,
            environment=s.environment,
            station_id=s.station_id,
            charger_id_dataset=s.preferred_charger_id,
            charger_id_assigned=assigned_cp[s.session_id],
            vehicle_reg_no=s.vehicle_reg_no,
            manufacturer=s.manufacturer,
            model_variant=s.model_variant,
            battery_kwh_verified=s.battery_kwh_verified,
            soc_arrival_pct=s.soc_arrival_pct,
            soc_target_pct=s.soc_target_pct,
            connector_type=s.connector_type,
            requested_energy_kwh=s.requested_energy_kwh,
            energy_delivered_kwh=delivered,
            missed_energy_kwh=missed,
            energy_completion_pct=pct,
            completed_by_deadline=completed,
            waiting_time_min=round(wait, 2) if wait is not None else None,
            charge_start_min=c_start,
            charge_end_min=c_end,
            deadline_min=s.deadline_min,
            arrival_min=s.arrival_min,
            sim_power_kw=s.sim_power_kw,
            compatibility_caveat=s.compatibility_caveat,
            model_id=s.model_id,
            theoretically_feasible=s.theoretically_feasible,
            timeline=timelines[s.session_id],
            started=bool(timelines[s.session_id]),
            request_id=s.request_id,
        ))

    return results


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

def run_all_algorithms(
    sessions_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    station_power_cap_kw: float | dict[str, float] | None = None,
    step_min: float = 1.0,
    facility_power_cap_kw: float | None = None,
    hybrid_weights: dict[str, float] | None = None,
) -> dict[str, list[SimResult]]:
    """Run FCFS, EDF, LAFS and Hybrid on identical inputs."""
    sessions = build_sessions(sessions_df)
    base_cps = build_charge_points(stations_df)

    if station_power_cap_kw is None:
        if stations_df is None or stations_df.empty:
            station_power_cap_kw = {}
        elif "station_id" in stations_df.columns and stations_df["station_id"].nunique() > 1:
            station_power_cap_kw = {
                str(station_id): float(cap)
                for station_id, cap in stations_df.groupby("station_id")["station_aggregate_power_cap_kw_scenario"].first().items()
            }
        else:
            station_power_cap_kw = float(stations_df["station_aggregate_power_cap_kw_scenario"].iloc[0])

    results: dict[str, list[SimResult]] = {}
    for algo in ALGORITHMS:
        cps = [
            ChargePoint(
                cp_id=cp.cp_id,
                station_id=cp.station_id,
                connector_type=cp.connector_type,
                rated_power_kw=cp.rated_power_kw,
            )
            for cp in base_cps
        ]
        results[algo] = _simulate(
            algorithm=algo,
            sessions=sessions,
            charge_points=cps,
            station_power_cap_kw=station_power_cap_kw,
            step_min=step_min,
            facility_power_cap_kw=facility_power_cap_kw,
            hybrid_weights=hybrid_weights,
        )
    return results


def run_single_algorithm(
    algorithm: str,
    sessions_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    station_power_cap_kw: float | dict[str, float] | None = None,
    step_min: float = 1.0,
    facility_power_cap_kw: float | None = None,
    hybrid_weights: dict[str, float] | None = None,
) -> list[SimResult]:
    """Run a single selected algorithm for live charging operations."""
    sessions = build_sessions(sessions_df)
    base_cps = build_charge_points(stations_df)
    if station_power_cap_kw is None:
        station_power_cap_kw = (0.0 if stations_df is None or stations_df.empty
                                else float(stations_df["station_aggregate_power_cap_kw_scenario"].iloc[0]))

    cps = [
        ChargePoint(
            cp_id=cp.cp_id,
            station_id=cp.station_id,
            connector_type=cp.connector_type,
            rated_power_kw=cp.rated_power_kw,
        )
        for cp in base_cps
    ]
    return _simulate(
        algorithm=algorithm,
        sessions=sessions,
        charge_points=cps,
        station_power_cap_kw=station_power_cap_kw,
        step_min=step_min,
        facility_power_cap_kw=facility_power_cap_kw,
        hybrid_weights=hybrid_weights,
    )


def results_to_dataframe(results: dict[str, list[SimResult]]) -> pd.DataFrame:
    """Flatten results into a structured DataFrame."""
    rows = []
    for algo, sim_list in results.items():
        for r in sim_list:
            rows.append({
                "Algorithm":           r.algorithm,
                "Session ID":          r.session_id,
                "Vehicle Reg No":      r.vehicle_reg_no,
                "Environment":         r.environment,
                "Station ID":          r.station_id,
                "Charger (Dataset)":   r.charger_id_dataset,
                "Charger (Assigned)":  r.charger_id_assigned,
                "Manufacturer":        r.manufacturer,
                "Model Variant":       r.model_variant,
                "Battery (kWh)":       r.battery_kwh_verified,
                "SOC Arrival (%)":     r.soc_arrival_pct,
                "SOC Target (%)":      r.soc_target_pct,
                "Connector":           r.connector_type,
                "Requested (kWh)":     r.requested_energy_kwh,
                "Delivered (kWh)":     r.energy_delivered_kwh,
                "Missed (kWh)":        r.missed_energy_kwh,
                "Completion (%)":      r.energy_completion_pct,
                "Completed":           r.completed_by_deadline,
                "Feasible":            r.theoretically_feasible,
                "Waiting (min)":       r.waiting_time_min,
                "Started":             r.started,
                "Charge Start (min)":  round(r.charge_start_min, 1) if r.charge_start_min is not None else None,
                "Charge End (min)":    round(r.charge_end_min, 1) if r.charge_end_min is not None else None,
                "Charging Duration (min)": round(r.charging_duration_min, 2),
                "Peak Allocated Power (kW)": round(r.allocated_power_peak_kw, 3),
                "Vehicle Max Power (kW)": r.sim_power_kw,
                "SOC End (%)": round(r.soc_end_pct, 2) if r.soc_end_pct is not None else None,
                "Deadline (min)":      round(r.deadline_min, 1),
                "Arrival (min)":       round(r.arrival_min, 1),
                "Sim Power (kW)":      r.sim_power_kw,
                "Compat. Caveat":      r.compatibility_caveat,
                "Model ID":            r.model_id,
            })
    return pd.DataFrame(rows, dtype=object)


def compute_summary(results_df: pd.DataFrame, sim_results: dict | None = None, step_min: float = 1.0,
                    stations_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute aggregate KPIs for all algorithms.
    Wait statistics cover sessions that actually started charging; sessions that never
    received energy are counted separately in 'Never Started' (not as 0-minute waits).
    """
    if results_df is None or results_df.empty or "Algorithm" not in results_df.columns:
        return pd.DataFrame(columns=["Algorithm", "Sessions", "Completed", "Completion Rate (%)",
            "Mean Wait (min)", "Total Delivered (kWh)", "Total Missed (kWh)",
            "Charger Utilization (%)", "Peak Station Load (kW)", "Peak Facility Demand (kW)"])
    rows = []
    for algo in results_df["Algorithm"].unique():
        sub = results_df[results_df["Algorithm"] == algo]

        # Peak load: per station per step, then the highest single-station peak
        peak_load = None
        peak_facility_load = None
        if sim_results and algo in sim_results:
            station_events: dict[str, dict[float, float]] = {}
            facility_events: dict[float, float] = {}
            for r in sim_results[algo]:
                for t0, t1, pwr in r.timeline:
                    if t1 <= t0 or pwr <= 0:
                        continue
                    events = station_events.setdefault(r.station_id, {})
                    events[t0] = events.get(t0, 0.0) + pwr
                    events[t1] = events.get(t1, 0.0) - pwr
                    facility_events[t0] = facility_events.get(t0, 0.0) + pwr
                    facility_events[t1] = facility_events.get(t1, 0.0) - pwr
            peak_load = round(max((max(_sweep_power(events), default=0.0) for events in station_events.values()), default=0.0), 3)
            peak_facility_load = round(max(_sweep_power(facility_events), default=0.0), 3)

        completion = pd.to_numeric(sub["Completion (%)"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0).clip(lower=0, upper=100)
        progress = (completion / 100).tolist()
        requested = pd.to_numeric(sub["Requested (kWh)"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0).clip(lower=0)
        delivered = pd.to_numeric(sub["Delivered (kWh)"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0).clip(lower=0)
        missed = pd.to_numeric(sub["Missed (kWh)"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0).clip(lower=0)
        waits = pd.to_numeric(sub["Waiting (min)"], errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
        completed_flags = sub["Completed"].map(lambda value: str(value).strip().lower() in ("true", "1"))
        sum_progress = sum(progress)
        fairness = (sum_progress * sum_progress / (len(progress) * sum(x * x for x in progress))) if progress and sum(x * x for x in progress) else (1.0 if progress else 0.0)
        utilization = None
        if sim_results and stations_df is not None and algo in sim_results:
            util = compute_charger_utilization({algo: sim_results[algo]}, stations_df, step_min)
            utilization = round(float(util["Utilization (%)"].mean()), 2) if not util.empty else 0.0

        rows.append({
            "Algorithm":             algo,
            "Sessions":              len(sub),
            "Completed":             int(completed_flags.sum()),
            "Infeasible (inherent)": int((~sub["Feasible"]).sum()),
            "Never Started":         int((~sub["Started"]).sum()),
            "Completion Rate (%)":   round(100.0 * completed_flags.mean(), 2) if len(completed_flags) else None,
            "Energy Fulfilment by Deadline (%)": round(100.0 * delivered.sum() / requested.sum(), 2) if requested.sum() > 0 else None,
            "Mean Wait (min)":       _finite_or_none(waits.mean()),
            "Median Wait (min)":     _finite_or_none(waits.median()),
            "Max Wait (min)":        _finite_or_none(waits.max()),
            "Total Requested (kWh)": round(float(requested.sum()), 2),
            "Total Delivered (kWh)": round(float(delivered.sum()), 2),
            "Total Missed (kWh)":    round(float(missed.sum()), 2),
            "Avg Completion (%)":    _finite_or_none(completion.mean()),
            "Energy Progress Fairness (Jain)": round(fairness, 4),
            "Charger Utilization (%)": utilization,
            "Peak Station Load (kW)": peak_load,
            "Peak Facility Demand (kW)": peak_facility_load,
        })
    return pd.DataFrame(rows, dtype=object)


def _finite_or_none(value):
    """Represent undefined analytics as missing, never as a fabricated zero or NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 2) if math.isfinite(number) else None


def _sweep_power(events: dict[float, float]) -> list[float]:
    """Power immediately after each timeline boundary, for accurate concurrent peak checks."""
    total = 0.0
    values = []
    for t in sorted(events):
        total += events[t]
        values.append(max(0.0, total))
    return values


def compute_charger_utilization(
    results: dict[str, list[SimResult]],
    stations_df: pd.DataFrame,
    step_min: float = 1.0,
) -> pd.DataFrame:
    """
    Per-charger utilization for each algorithm:
        busy minutes / length of that charger's STATION scheduling window x 100
    The window runs from the station's first arrival to its last deadline. (Using one window for
    all stations understated utilization badly in the multi-station view: 30% vs 93%.)
    """
    rows = []
    for algo, sim_list in results.items():
        if not sim_list:
            continue
        windows: dict[str, tuple[float, float]] = {}
        for r in sim_list:
            lo, hi = windows.get(r.station_id, (r.arrival_min, r.deadline_min))
            windows[r.station_id] = (min(lo, r.arrival_min), max(hi, r.deadline_min))

        charger_busy: dict[str, float] = {}
        for r in sim_list:
            cp = r.charger_id_assigned
            if cp is None:
                continue
            busy_min = sum(t1 - t0 for t0, t1, _ in r.timeline)
            charger_busy[cp] = charger_busy.get(cp, 0.0) + busy_min

        for _, row in stations_df.iterrows():
            charger_status = str(row.get("charger_status", "available")).lower()
            station_status = str(row.get("station_status", "available")).lower()
            if charger_status != "available" or station_status != "available":
                continue
            cp_id = str(row["charger_id"])
            sid = str(row.get("station_id", ""))
            lo, hi = windows.get(sid, (0.0, 0.0))
            sim_duration = max(hi - lo, 1.0)
            busy = charger_busy.get(cp_id, 0.0)
            rows.append({
                "Algorithm":        algo,
                "Environment":      str(row.get("environment", "")),
                "Station ID":       sid,
                "Charger ID":       cp_id,
                "Connector":        str(row["connector_type"]),
                "Rated Power (kW)": float(row.get("rated_power_kw", 0) or 0),
                "Busy (min)":       round(busy, 1),
                "Sim Window (min)": round(sim_duration, 1),
                "Utilization (%)":  round(100.0 * busy / sim_duration, 2),
            })
    return pd.DataFrame(rows)


def snapshot_at(results: list[SimResult], minute: float) -> dict:
    """
    State of the station at one simulation minute, computed from the schedule:
      charging  : {charger_id: (SimResult, kW)}   ports delivering power at this minute
      occupied  : {charger_id: SimResult}         ports reserved by an active session
      waiting   : sessions that have arrived but are not yet charging
      demand_kw : total power drawn at that minute
    """
    charging: dict[str, tuple[SimResult, float]] = {}
    occupied: dict[str, SimResult] = {}
    waiting: list[SimResult] = []
    for r in results:
        if (r.charger_id_assigned and r.charge_start_min is not None
                and r.charge_start_min <= minute
                and (r.charge_end_min is None or minute < r.charge_end_min)):
            occupied[r.charger_id_assigned] = r
        kw = next((p for t0, t1, p in r.timeline if t0 <= minute < t1), None)
        if kw is not None and r.charger_id_assigned:
            charging[r.charger_id_assigned] = (r, kw)
        elif r.arrival_min <= minute < r.deadline_min and (r.charge_start_min is None or minute < r.charge_start_min):
            waiting.append(r)
    return {"charging": charging, "occupied": occupied, "waiting": waiting,
            "demand_kw": round(sum(kw for _, kw in charging.values()), 3)}


def energy_delivered_at(result: SimResult, minute: float) -> float:
    """Battery-side kWh delivered up to a specific simulation minute."""
    energy = sum(max(0.0, min(end, minute) - start) * power / 60.0 * CHARGE_EFFICIENCY
                 for start, end, power in result.timeline if start < minute)
    return min(float(result.requested_energy_kwh), energy)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _run_tests():
    """Unit tests for the scheduling engine and constraints."""
    def _make_session(sid, arrival, deadline, energy, power,
                      connector="Type-2 AC", station="ST1", charger="CP1",
                      max_ac=0.0, max_dc=0.0):
        return Session(
            session_id=sid, arrival_min=arrival, deadline_min=deadline,
            requested_energy_kwh=energy, sim_power_kw=power,
            connector_type=connector, environment="Test",
            station_id=station, preferred_charger_id=charger,
            max_ac_charge_power_kw=max_ac, max_dc_charge_power_kw=max_dc,
        )

    def _make_cp(cpid, station="ST1", connector="Type-2 AC", power=7.2):
        return ChargePoint(cp_id=cpid, station_id=station,
                           connector_type=connector, rated_power_kw=power)

    cap = 20.0

    # Test 1: Single session full delivery
    for algo in ALGORITHMS:
        res = _simulate(algo, [_make_session("A", 0, 120, 5.0, 7.0)], [_make_cp("CP1")], cap)
        assert res[0].completed_by_deadline, f"{algo} Test 1 failed"
    print("PASS Test 1: single session full delivery")

    # Test 2: FCFS ordering (earlier arrival served first)
    s1 = _make_session("S1", 0, 200, 3.0, 7.0, charger="CP1")
    s2 = _make_session("S2", 5, 200, 3.0, 7.0, charger="CP1")
    res = _simulate("FCFS", [s1, s2], [_make_cp("CP1")], cap)
    r = {r.session_id: r for r in res}
    assert r["S1"].waiting_time_min <= r["S2"].waiting_time_min, "FCFS Test 2 failed"
    print("PASS Test 2: FCFS ordering")

    # Test 3: EDF ordering (earliest deadline served first)
    s_late  = _make_session("LATE",  0, 300, 2.0, 7.0, charger="CP1")
    s_early = _make_session("EARLY", 0,  60, 2.0, 7.0, charger="CP1")
    res = _simulate("EDF", [s_late, s_early], [_make_cp("CP1")], cap)
    r = {r.session_id: r for r in res}
    assert r["EARLY"].waiting_time_min <= r["LATE"].waiting_time_min, "EDF Test 3 failed"
    print("PASS Test 3: EDF prefers early deadline")

    # Test 4: Power cap enforcement
    sessions4 = [_make_session(f"C{i}", 0, 60, 5.0, 7.0, charger=f"CP{i}", station="ST1") for i in range(5)]
    cps4 = [_make_cp(f"CP{i}", power=7.0) for i in range(5)]
    res4 = _simulate("FCFS", sessions4, cps4, 10.0)
    total_delivered = sum(r.energy_delivered_kwh for r in res4)
    max_possible = 10.0 * (60.0 / 60.0) * CHARGE_EFFICIENCY
    assert total_delivered <= max_possible + 0.1, f"Test 4 failed: {total_delivered:.3f} > {max_possible:.3f}"
    print(f"PASS Test 4: power cap (delivered {total_delivered:.3f} <= {max_possible:.3f} kWh)")

    # Test 5: Incompatible connector blocks charging
    s_inc = _make_session("INC", 0, 120, 3.0, 3.3, connector="Bharat AC-001")
    cp_ccs2 = ChargePoint(cp_id="CCS2_CP", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=50)
    res5 = _simulate("FCFS", [s_inc], [cp_ccs2], cap)
    assert res5[0].energy_delivered_kwh == 0.0, "Test 5 failed"
    print("PASS Test 5: incompatible connector blocks charging")

    # Test 6: Charger mutual exclusion
    sA = _make_session("A", 0, 120, 3.0, 7.0, charger="CP1")
    sB = _make_session("B", 0, 120, 3.0, 7.0, charger="CP1")
    res6 = _simulate("FCFS", [sA, sB], [_make_cp("CP1")], cap)
    r = {r.session_id: r for r in res6}
    times_A = {t for t, _, _ in r["A"].timeline}
    times_B = {t for t, _, _ in r["B"].timeline}
    assert len(times_A & times_B) == 0, "Test 6 failed: charger overlap"
    print("PASS Test 6: charger mutual exclusion enforced")

    # Test 7: Infeasible session partial delivery
    s_inf = _make_session("INF", 0, 10, 10.0, 1.0, charger="CP1")
    res7 = _simulate("FCFS", [s_inf], [_make_cp("CP1", power=1.0)], cap)
    assert not res7[0].completed_by_deadline, "Test 7 failed"
    print(f"PASS Test 7: infeasible session partial delivery ({res7[0].energy_delivered_kwh:.3f} kWh / 10.0 kWh)")

    # Test 8: Vehicle-side AC input constraint (Vehicle onboard limit 3.3 kW on 22 kW charger)
    s_ac = _make_session("AC_LIM", 0, 60, 10.0, 22.0, connector="Type-2 AC", max_ac=3.3)
    cp_22 = _make_cp("CP_22", power=22.0)
    res8 = _simulate("FCFS", [s_ac], [cp_22], 50.0)
    max_delivered_33 = 3.3 * (60.0 / 60.0) * CHARGE_EFFICIENCY
    assert res8[0].energy_delivered_kwh <= max_delivered_33 + 0.05, f"Test 8 failed: {res8[0].energy_delivered_kwh}"
    print(f"PASS Test 8: vehicle-side AC limit capped at 3.3 kW (delivered {res8[0].energy_delivered_kwh:.3f} kWh)")

    # Test 9: Vehicle-side DC input constraint (Vehicle onboard DC cap 30 kW on 150 kW charger)
    s_dc = _make_session("DC_LIM", 0, 30, 20.0, 150.0, connector="CCS2 DC", max_dc=30.0)
    cp_150 = ChargePoint(cp_id="CP_150", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=150.0)
    res9 = _simulate("FCFS", [s_dc], [cp_150], 250.0)
    max_delivered_30 = 30.0 * (30.0 / 60.0) * CHARGE_EFFICIENCY
    assert res9[0].energy_delivered_kwh <= max_delivered_30 + 0.05, f"Test 9 failed: {res9[0].energy_delivered_kwh}"
    print(f"PASS Test 9: vehicle-side DC limit capped at 30 kW (delivered {res9[0].energy_delivered_kwh:.3f} kWh)")

    # Test 10: Multi-session LAFS proportional throttling under severe overload
    s_lafs1 = _make_session("L1", 0, 60, 20.0, 50.0, connector="CCS2 DC", charger="CP1")
    s_lafs2 = _make_session("L2", 0, 60, 20.0, 50.0, connector="CCS2 DC", charger="CP2")
    cps10 = [
        ChargePoint(cp_id="CP1", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=50.0),
        ChargePoint(cp_id="CP2", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=50.0),
    ]
    res10 = _simulate("LAFS", [s_lafs1, s_lafs2], cps10, 40.0)
    tot10 = sum(r.energy_delivered_kwh for r in res10)
    max_tot_40 = 40.0 * (60.0 / 60.0) * CHARGE_EFFICIENCY
    assert tot10 <= max_tot_40 + 0.1, f"Test 10 failed: {tot10} > {max_tot_40}"
    print(f"PASS Test 10: LAFS proportional sharing under 40 kW cap (total {tot10:.2f} <= {max_tot_40:.2f} kWh)")

    # Test 11: Charger conflict / concurrent exclusive charger contention
    # Both arrive at t=0, both request CP1. Only CP1 exists.
    s_conf_1 = _make_session("CONF_1", 0, 100, 5.0, 10.0, charger="CP1")
    s_conf_2 = _make_session("CONF_2", 0, 100, 5.0, 10.0, charger="CP1")
    cp_single = [_make_cp("CP1", power=10.0)]
    res11 = _simulate("FCFS", [s_conf_1, s_conf_2], cp_single, 50.0)
    r11 = {r.session_id: r for r in res11}
    slots_1 = {t for t, _, _ in r11["CONF_1"].timeline}
    slots_2 = {t for t, _, _ in r11["CONF_2"].timeline}
    assert len(slots_1 & slots_2) == 0, "Test 11 failed: Charger conflict not serialized!"
    assert r11["CONF_2"].waiting_time_min > 0, "Test 11 failed: Second session should have waited"
    print("PASS Test 11: charger contention serialized with zero overlap")

    # Test 12: Incompatible connector strict rejection
    s_bad_conn = _make_session("BAD_CONN", 0, 60, 10.0, 50.0, connector="15A Domestic Socket")
    cp_ccs = [ChargePoint(cp_id="CCS_ONLY", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=120.0)]
    res12 = _simulate("FCFS", [s_bad_conn], cp_ccs, 120.0)
    assert res12[0].energy_delivered_kwh == 0.0, "Test 12 failed: Incompatible connector charged!"
    assert res12[0].charger_id_assigned is None, "Test 12 failed: Incompatible charger assigned!"
    print("PASS Test 12: incompatible connector rejected (zero energy, unassigned)")

    # Test 13: Strict / Missed departure deadline detection
    # Session needs 15 kWh at 7.2 kW (~140 min needed), but deadline is at 30 min
    s_urgent = _make_session("TIGHT_DEADLINE", 0, 30, 15.0, 7.2, charger="CP1")
    res13 = _simulate("FCFS", [s_urgent], [_make_cp("CP1", power=7.2)], 20.0)
    assert not res13[0].completed_by_deadline, "Test 13 failed: Should have missed deadline"
    assert res13[0].missed_energy_kwh > 0.0, "Test 13 failed: Missed energy should be positive"
    assert res13[0].energy_delivered_kwh > 0.0, "Test 13 failed: Should have delivered partial energy"
    print(f"PASS Test 13: tight deadline detected (missed {res13[0].missed_energy_kwh:.2f} kWh, partial delivery)")

    # Test 14: LAFS concurrent multi-tenant sharing vs EDF prioritization
    # Two vehicles arrive together under 50 kW station cap with two 50 kW chargers.
    # In LAFS, both vehicles connect and charge simultaneously at ~25 kW each.
    # In EDF, the earlier-deadline vehicle takes full 50 kW first.
    s_edf_early = _make_session("EARLY_EV", 0, 40, 15.0, 50.0, charger="CP1")
    s_edf_late  = _make_session("LATE_EV",  0, 90, 15.0, 50.0, charger="CP2")
    cps14 = [_make_cp("CP1", power=50.0), _make_cp("CP2", power=50.0)]

    res_edf = _simulate("EDF", [s_edf_early, s_edf_late], cps14, 50.0)
    r_edf = {r.session_id: r for r in res_edf}

    res_lafs = _simulate("LAFS", [s_edf_early, s_edf_late], cps14, 50.0)
    r_lafs = {r.session_id: r for r in res_lafs}

    # Both policies connect both vehicles immediately; priority changes the split, not
    # whether a compatible free charger is used.
    assert r_edf["EARLY_EV"].charge_start_min == 0.0
    assert r_edf["LATE_EV"].charge_start_min == 0.0, "EDF should use both free compatible chargers"
    edf_kw = {sid: next(p for t0, _, p in result.timeline if t0 == 0)
              for sid, result in r_edf.items()}
    assert edf_kw["EARLY_EV"] > edf_kw["LATE_EV"] > 0, "EDF priority should weight the earlier deadline"

    # Under LAFS, BOTH vehicles start charging at t=0, sharing 50 kW capacity (25 kW each)
    assert r_lafs["EARLY_EV"].charge_start_min == 0.0
    assert r_lafs["LATE_EV"].charge_start_min == 0.0, "LAFS should connect both vehicles concurrently"
    assert r_lafs["LATE_EV"].waiting_time_min == 0.0, "LAFS should have 0 waiting time when chargers are free"
    print("PASS Test 14: EDF priority and LAFS share two compatible chargers concurrently")

    # Test 15: connector direction. An AC-only (Type-2) vehicle must NOT be placed on a CCS2 DC charger,
    # while a CCS2-inlet vehicle may use an AC charger, limited by its own AC input limit.
    s_ac_only = _make_session("AC_ONLY", 0, 120, 5.0, 7.0, connector="Type-2 AC")
    cp_dc = ChargePoint(cp_id="DC1", station_id="ST1", connector_type="CCS2 DC", rated_power_kw=50.0)
    r15 = _simulate("FCFS", [s_ac_only], [cp_dc], 100.0)
    assert r15[0].energy_delivered_kwh == 0.0 and r15[0].charger_id_assigned is None, "Test 15 failed: AC-only EV on DC charger"
    s_ccs = _make_session("CCS_ON_AC", 0, 120, 3.0, 50.0, connector="CCS2 DC", max_ac=7.2, max_dc=50.0)
    r15b = _simulate("FCFS", [s_ccs], [_make_cp("AC1", power=22.0)], 100.0)
    assert r15b[0].energy_delivered_kwh > 0 and max(p for _, _, p in r15b[0].timeline) <= 7.2 + 1e-9, "Test 15b failed"
    print("PASS Test 15: connector direction (AC-only EV blocked from DC; CCS2 EV limited to its AC input on AC charger)")

    # Test 16: power is recomputed every step - a session that connected with little headroom speeds up
    # once the other session finishes (FCFS and EDF).
    for algo in ("FCFS", "EDF"):
        a = _make_session("QUICK", 0, 600, 2.0, 7.0, charger="CP1")
        b = _make_session("LONG", 1, 600, 20.0, 7.0, charger="CP2")   # arrives later -> lower priority
        r16 = {x.session_id: x for x in _simulate(algo, [a, b], [_make_cp("CP1", power=7.0), _make_cp("CP2", power=7.0)], 10.0)}
        powers = [p for _, _, p in r16["LONG"].timeline]
        assert min(powers) < 7.0 - 1e-9 and max(powers) >= 7.0 - 1e-9, f"Test 16 failed ({algo}): power never recovered {sorted(set(powers))}"
    print("PASS Test 16: power is re-allocated every step (headroom no longer locked at connect time)")

    # Test 17: a session that never gets a charger has waiting_time None (not 0) and started=False
    a = _make_session("A", 0, 300, 30.0, 7.0, charger="CP1")
    b = _make_session("B", 0, 30, 5.0, 7.0, charger="CP1")
    r17 = {x.session_id: x for x in _simulate("FCFS", [a, b], [_make_cp("CP1")], 50.0)}
    assert r17["A"].started and r17["A"].waiting_time_min == 0.0, "Test 17 failed: A"
    assert (not r17["B"].started) and r17["B"].waiting_time_min is None, "Test 17 failed: B reported a wait"
    print("PASS Test 17: never-served session reports no waiting time (not a fabricated 0)")

    # Test 18: each station's own cap is enforced in a multi-station run
    caps18 = {"ST1": 10.0, "ST2": 100.0}
    sess18 = [_make_session(f"X{i}", 0, 120, 8.0, 7.0, station="ST1", charger=f"A{i}") for i in range(3)] + \
             [_make_session(f"Y{i}", 0, 120, 8.0, 7.0, station="ST2", charger=f"B{i}") for i in range(3)]
    cps18 = [_make_cp(f"A{i}", station="ST1") for i in range(3)] + [_make_cp(f"B{i}", station="ST2") for i in range(3)]
    for algo in ALGORITHMS:
        res18 = _simulate(algo, sess18, cps18, caps18)
        load: dict[tuple[str, float], float] = {}
        for r in res18:
            for t0, _, p in r.timeline:
                load[(r.station_id, t0)] = load.get((r.station_id, t0), 0.0) + p
        for (sid, _), kw in load.items():
            assert kw <= caps18[sid] + 1e-6, f"Test 18 failed ({algo}): {sid} drew {kw:.1f} kW > cap {caps18[sid]}"
    print("PASS Test 18: per-station power caps respected in a multi-station simulation")

    # Test 19: utilization is measured over each station's own scheduling window
    stations19 = pd.DataFrame([
        {"charger_id": "CPa", "environment": "E", "station_id": "ST1", "connector_type": "Type-2 AC", "rated_power_kw": 7.2},
        {"charger_id": "CPb", "environment": "E", "station_id": "ST2", "connector_type": "Type-2 AC", "rated_power_kw": 7.2},
    ])
    s_a = _make_session("U1", 0, 60, 5.0, 7.0, station="ST1", charger="CPa")
    s_b = _make_session("U2", 1000, 1060, 5.0, 7.0, station="ST2", charger="CPb")
    res19 = _simulate("FCFS", [s_a, s_b], [_make_cp("CPa", station="ST1"), _make_cp("CPb", station="ST2")], 50.0)
    u19 = compute_charger_utilization({"FCFS": res19}, stations19).set_index("Charger ID")
    assert u19.loc["CPa", "Sim Window (min)"] == 60.0 and u19.loc["CPa", "Utilization (%)"] > 50.0, "Test 19 failed"
    print(f"PASS Test 19: per-station utilization window (CPa {u19.loc['CPa', 'Utilization (%)']:.0f}% over its own 60 min)")

    # Test 20: snapshot never reports more occupied chargers than exist, and demand matches
    res20 = _simulate("FCFS", sessions4, cps4, 10.0)
    for minute in (0, 5, 20, 40):
        snap = snapshot_at(res20, minute)
        assert len(snap["charging"]) <= len(cps4), "Test 20 failed: more occupied than chargers"
        assert snap["demand_kw"] <= 10.0 + 1e-6, "Test 20 failed: demand above cap"
    print("PASS Test 20: charger snapshot consistent (occupied <= chargers, demand <= cap)")

    # Test 21: duplicate session IDs are rejected instead of silently merging two vehicles
    try:
        build_sessions(pd.DataFrame({"session_id": ["X", "X"]}))
        raise AssertionError("Test 21 failed: duplicate IDs accepted")
    except ValueError:
        pass
    print("PASS Test 21: duplicate session IDs rejected")

    # Test 22: the 3/5/10 vehicle fleet regression with a fixed three-port station.
    # Vehicles beyond physical port capacity stay queued, then start as ports free;
    # every available port is used concurrently and the grid cap is never exceeded.
    for fleet_size in (3, 5, 10):
        fleet_sessions = [
            _make_session(f"F{fleet_size}-{i}", 0, 600, 10.0, 50.0,
                          connector="CCS2 DC", charger=f"CP-{i}")
            for i in range(fleet_size)
        ]
        fleet_cps = [_make_cp(f"CP-{i}", connector="CCS2 DC", power=50.0)
                     for i in range(3)]
        for algo in ALGORITHMS:
            fleet_res = _simulate(algo, fleet_sessions, fleet_cps, 30.0)
            first_step = [r for r in fleet_res if any(t0 == 0 for t0, _, _ in r.timeline)]
            assert len(first_step) == min(fleet_size, len(fleet_cps)), \
                f"Test 22 failed ({fleet_size}/{algo}): free ports not used at once"
            first_ids = [r.charger_id_assigned for r in first_step]
            assert len(set(first_ids)) == len(first_ids) and None not in first_ids, \
                f"Test 22 failed ({fleet_size}/{algo}): simultaneous vehicles share a port"
            assert sum(next(p for t0, _, p in r.timeline if t0 == 0) for r in first_step) <= 30.0 + 1e-6, \
                f"Test 22 failed ({fleet_size}/{algo}): grid cap exceeded"
            assert all(r.started and r.charge_start_min is not None for r in fleet_res), \
                f"Test 22 failed ({fleet_size}/{algo}): queued vehicles never got a released port"
            per_minute: dict[float, float] = {}
            for result in fleet_res:
                for t0, _, power in result.timeline:
                    per_minute[t0] = per_minute.get(t0, 0.0) + power
            assert max(per_minute.values(), default=0.0) <= 30.0 + 1e-6, \
                f"Test 22 failed ({fleet_size}/{algo}): grid cap exceeded later"
            snap = snapshot_at(fleet_res, 0.0)
            assert len(snap["charging"]) == min(fleet_size, len(fleet_cps)) \
                   and len(snap["occupied"]) == len(snap["charging"]), \
                f"Test 22 failed ({fleet_size}/{algo}): charger snapshot lost assignments"
            assert len(snap["waiting"]) == max(0, fleet_size - len(fleet_cps)), \
                f"Test 22 failed ({fleet_size}/{algo}): vehicles without a free port not queued"
        print(f"PASS Test 22: {fleet_size} vehicles, fixed 3 ports, queue recovery, and 30 kW cap across all policies")

    # Hybrid-specific checks (Tests 23–28).
    score_sessions = [_make_session("OLD", 0, 100, 10, 7), _make_session("NEW", 10, 100, 10, 7)]
    age_scores = {r["session"].session_id: r["fcfs"] for r in _hybrid_priority_rows(score_sessions, 20, {})}
    assert age_scores["OLD"] > age_scores["NEW"]
    print("PASS Test 23: accumulated waiting age increases FCFS contribution")

    deadline_sessions = [_make_session("EARLY", 0, 60, 10, 7), _make_session("LATE", 0, 120, 10, 7)]
    deadline_scores = {r["session"].session_id: r["edf"] for r in _hybrid_priority_rows(deadline_sessions, 0, {})}
    assert deadline_scores["EARLY"] > deadline_scores["LATE"]
    print("PASS Test 24: earlier deadline increases normalized EDF urgency")

    fairness_sessions = [_make_session("PROGRESS", 0, 100, 10, 7),
                         _make_session("UNSERVED", 0, 100, 10, 7)]
    progress_scores = {r["session"].session_id: r["fairness"] for r in
                       _hybrid_priority_rows(fairness_sessions, 20, {"PROGRESS": 8.0})}
    assert progress_scores["PROGRESS"] < progress_scores["UNSERVED"]
    assert progress_scores["UNSERVED"] > 0
    print("PASS Test 25: service-debt fairness distinguishes waiting requests and progress")

    assert validate_hybrid_weights() == HYBRID_DEFAULT_WEIGHTS
    for invalid in ({"FCFS": .4, "EDF": .4, "FAIRNESS": .1},
                    {"FCFS": -.1, "EDF": .6, "FAIRNESS": .5}):
        try:
            validate_hybrid_weights(invalid)
            raise AssertionError("Hybrid accepted invalid weights")
        except ValueError:
            pass
    print("PASS Test 26: weights require finite nonnegative values summing to one")

    tied = [_make_session("B", 0, 100, 10, 7), _make_session("A", 0, 100, 10, 7)]
    tied_rank = [r["session"].session_id for r in _hybrid_priority_rows(tied, 0, {})]
    assert tied_rank == ["A", "B"] and tied_rank == [r["session"].session_id for r in _hybrid_priority_rows(tied, 0, {})]
    print("PASS Test 27: equal combined scores use deterministic deadline/arrival/session-ID order")

    hybrid_demo = [_make_session("H1", 0, 60, 20, 50, connector="CCS2 DC", charger="HCP1"),
                   _make_session("H2", 0, 60, 20, 50, connector="CCS2 DC", charger="HCP2")]
    hybrid_out = _simulate("HYBRID", hybrid_demo,
        [_make_cp("HCP1", connector="CCS2 DC", power=50), _make_cp("HCP2", connector="CCS2 DC", power=50)], 30)
    assert all(r.charge_start_min == 0 for r in hybrid_out)
    assert sum(next(p for t0, _, p in r.timeline if t0 == 0) for r in hybrid_out) <= 30 + 1e-6
    assert len({r.charger_id_assigned for r in hybrid_out}) == 2
    print("PASS Test 28: hybrid uses compatible ports concurrently and LAFS shares under station cap")

    bad_sessions = pd.DataFrame([{"session_id":"NAN", "arrival_epoch_min":0, "deadline_epoch_min":30,
        "energy_required_kwh":float("nan"), "sim_power_kw":7.2, "connector_type":"Type-2 AC",
        "station_id":"ST1"}])
    try:
        build_sessions(bad_sessions)
        raise AssertionError("Non-finite session energy was accepted")
    except ValueError:
        pass
    print("PASS Test 29: non-finite session inputs are rejected before dispatch")

    bad_caps = pd.DataFrame([{"station_id":"ST1", "charger_id":"BAD", "connector_type":"Type-2 AC",
        "rated_power_kw":float("nan"), "charger_status":"available", "station_status":"available"}])
    try:
        build_charge_points(bad_caps)
        raise AssertionError("Non-finite charger rating was accepted")
    except ValueError:
        pass
    print("PASS Test 30: non-finite charger ratings are rejected")

    preview_df = pd.DataFrame([
        {"session_id":"NOW", "arrival_epoch_min":0, "deadline_epoch_min":100, "energy_required_kwh":2,
         "sim_power_kw":7, "connector_type":"Type-2 AC", "station_id":"ST1"},
        {"session_id":"LATER", "arrival_epoch_min":10, "deadline_epoch_min":110, "energy_required_kwh":2,
         "sim_power_kw":7, "connector_type":"Type-2 AC", "station_id":"ST1"},
    ])
    preview = build_hybrid_priority_list(preview_df, pd.DataFrame([
        {"station_id":"ST1", "charger_id":"CP1", "connector_type":"Type-2 AC", "rated_power_kw":7.2,
         "charger_status":"available", "station_status":"available"}]))
    assert set(preview["Session ID"]) == {"NOW", "LATER"}
    print("PASS Test 31: default Hybrid preview includes later-arriving requests")

    print(f"\nAll {TEST_COUNT} scheduler checks passed successfully.")

if __name__ == "__main__":
    _run_tests()
