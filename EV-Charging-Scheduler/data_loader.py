"""
data_loader.py
==============
Loads and validates the VoltFlow EV Charging & Scheduling dataset.
Supports both the baseline 48-session experimental dataset and the
extended multi-environment scenario dataset (96 sessions total).
Provides human-readable display mappings and strict schema validation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import math
import uuid
from pathlib import Path
import streamlit as st

DATASET_PATH = Path(__file__).parent / "data" / "India_EV_Charging_Scheduler_Complete_Dataset.xlsx"

# ---------------------------------------------------------------------------
# Required columns per sheet (checked before any processing)
# ---------------------------------------------------------------------------
REQUIRED_SESSIONS_COLS = {
    # Core identifiers
    "session_id", "environment", "station_id",
    # Timing & demand
    "connection_time_IST", "departure_time_IST",
    "energy_required_kwh", "charging_power_limit_kw",
    "charger_id", "station_power_limit_kw",
    # Vehicle specifications
    "vehicle_model_id", "connector_type_scenario",
    "battery_capacity_kwh_verified", "manufacturer", "model_variant",
}

REQUIRED_STATIONS_COLS = {
    "environment", "station_id", "charger_id", "connector_type",
    "rated_power_kw", "station_aggregate_power_cap_kw_scenario",
}

REQUIRED_CATALOG_COLS = {
    "model_id", "vehicle_category", "manufacturer", "model_variant",
    "battery_capacity_kwh_verified",
}

# ---------------------------------------------------------------------------
# Human-Readable Display Labels (Strictly No Snake_Case in UI)
# ---------------------------------------------------------------------------
DISPLAY_COLUMNS = {
    "session_id": "Session ID",
    "environment": "Charging Environment",
    "station_id": "Station ID",
    "charger_id": "Assigned Charger ID",
    "charger_id_dataset": "Preferred Charger (Dataset)",
    "charger_id_assigned": "Allocated Charger",
    "vehicle_model_id": "Model ID",
    "model_id": "Model ID",
    "manufacturer": "Manufacturer",
    "model_name": "Model Name",
    "model_variant": "Model Variant",
    "variant_model_year": "Variant & Model Year",
    "vehicle_category": "Vehicle Category",
    "market_status": "Market Status",
    "battery_capacity_kwh_verified": "Battery Capacity (kWh)",
    "battery_kwh_verified": "Battery Capacity (kWh)",
    "battery_kwh_scenario": "Scenario Battery (kWh)",
    "battery_capacity_kwh": "Battery Capacity (kWh)",
    "usable_capacity_kwh": "Usable Capacity (kWh)",
    "connection_time_IST": "Connection Time (IST)",
    "departure_time_IST": "Departure Deadline (IST)",
    "arrival_time_IST": "Connection Time (IST)",
    "deadline_time_IST": "Departure Deadline (IST)",
    "available_window_min": "Available Window (min)",
    "window_min": "Available Window (min)",
    "soc_at_arrival_pct_scenario": "Arrival SOC (%)",
    "soc_arrival_pct": "Arrival SOC (%)",
    "target_soc_pct_scenario": "Target SOC (%)",
    "soc_target_pct": "Target SOC (%)",
    "energy_required_kwh": "Required Energy (kWh)",
    "requested_energy_kwh": "Required Energy (kWh)",
    "requested_energy_kwh_scenario": "Required Energy (kWh)",
    "connector_type": "Connector Type",
    "connector_type_scenario": "Connector Type",
    "charging_connector": "Charging Connector",
    "charging_standard": "Charging Standard",
    "charging_power_limit_kw": "Session Power Limit (kW)",
    "effective_session_power_limit_kw": "Session Power Limit (kW)",
    "effective_power_kw": "Session Power Limit (kW)",
    "sim_power_kw": "Effective Charging Power (kW)",
    "max_ac_charge_power_kw": "Max AC Charging Input (kW)",
    "max_dc_charge_power_kw": "Max DC Charging Input (kW)",
    "dc_fast_charging_supported": "DC Fast Charging",
    "station_power_limit_kw": "Station Power Limit (kW)",
    "station_aggregate_power_cap_kw_scenario": "Station Power Limit (kW)",
    "theoretically_feasible": "Feasible Within Window",
    "compatibility_caveat": "Compatibility Caveat",
    "data_origin": "Data Provenance",
    "waiting_time_min": "Waiting Time (min)",
    "energy_delivered_kwh": "Delivered Energy (kWh)",
    "missed_energy_kwh": "Unserved Energy (kWh)",
    "energy_completion_pct": "Completion Rate (%)",
    "completed_by_deadline": "Deadline Met",
    "completed": "Deadline Met",
    "charge_start_min": "Charging Start (min)",
    "charge_end_min": "Charging End (min)",
    "deadline_min": "Deadline (min)",
    "arrival_min": "Arrival (min)",
    "verification_status": "Verification Status",
    "official_ac_charge_duration": "Official AC Charging Duration",
    "official_dc_charge_duration": "Official DC Charging Duration",
    "source_url": "OEM Specification Source",
    "rated_power_kw": "Rated Power (kW)",
    "algorithm": "Scheduling Algorithm",
    "Algorithm": "Scheduling Algorithm",
}

def humanize_dataframe(df: pd.DataFrame, extra_mapping: dict | None = None) -> pd.DataFrame:
    """Return a copy of df with all column names mapped to human-readable labels, deduplicating target names."""
    if df is None or df.empty:
        return df
    mapping = {**DISPLAY_COLUMNS, **(extra_mapping or {})}
    df_out = df.copy()

    seen_labels: set[str] = set()
    cols_to_keep = []
    new_names = []
    for col in df_out.columns:
        lbl = mapping.get(col, col)
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            cols_to_keep.append(col)
            new_names.append(lbl)

    df_out = df_out[cols_to_keep]
    df_out.columns = new_names
    return df_out

# ---------------------------------------------------------------------------
# Connector compatibility matrix (BEE / MoP India standards)
# ---------------------------------------------------------------------------
# Key   = connector (inlet) the VEHICLE has.
# Value = charger connector types that vehicle can physically use.
# NOTE (fixed): the earlier table let a "Type-2 AC" vehicle use a "CCS2 DC" charger.
# That is backwards: a CCS2 *inlet* accepts a Type-2 AC plug, but a Type-2-only inlet
# cannot accept a CCS2 plug. Bharat DC-001 is a different (low-voltage DC) standard and
# is not interchangeable with CCS2.  Adapter-based entries are explicit assumptions.
CONNECTOR_COMPAT: dict[str, list[str]] = {
    "Type-2 AC":                 ["Type-2 AC"],
    "CCS2 DC":                   ["CCS2 DC", "Type-2 AC"],
    "Bharat AC-001":             ["Bharat AC-001"],
    "Bharat DC-001":             ["Bharat DC-001"],
    "Proprietary 3-pin OEM AC":  ["Bharat AC-001", "Type-2 AC"],   # assumption: 15A adapter
    "Ather Grid (IS 17017)":     ["Bharat AC-001", "Type-2 AC"],   # assumption: adapter
    "15A Domestic Socket":       ["Bharat AC-001"],                # assumption: 15A adapter
    "IEC 60309 / 16A Industrial Plug": ["Bharat AC-001"],
}

def is_connector_compatible(vehicle_connector: str, station_connector: str) -> bool:
    """Return True if a vehicle with `vehicle_connector` can use a charger with `station_connector`."""
    vc = str(vehicle_connector).strip()
    sc = str(station_connector).strip()
    if vc == sc:
        return True
    return sc in CONNECTOR_COMPAT.get(vc, [])


# Catalog `charging_connector` text -> vehicle inlet types (explicit, no invented defaults)
_CATALOG_CONNECTOR_MAP: dict[str, list[str]] = {
    "CCS2 DC / Type-2 AC": ["CCS2 DC", "Type-2 AC"],
    "Type-2 AC Only": ["Type-2 AC"],
    "Proprietary 3-pin OEM AC": ["Proprietary 3-pin OEM AC"],
    "Ather Grid (Proprietary / Light Electric Vehicle)": ["Ather Grid (IS 17017)"],
    "Ather Grid (IS 17017)": ["Ather Grid (IS 17017)"],
    "Ola Hypercharger / Proprietary Home Socket": ["Proprietary 3-pin OEM AC"],
    "Standard 5A/15A Indian AC socket": ["15A Domestic Socket"],
    "Standard 15A Indian Plug (No CCS)": ["15A Domestic Socket"],
    "IEC 60309 / 16A Industrial Plug": ["IEC 60309 / 16A Industrial Plug"],
    "Bharat AC-001 / 16A plug": ["Bharat AC-001"],
    "Bharat DC-001 / AC 3.3 kW": ["Bharat DC-001", "Bharat AC-001"],
    "Bharat DC-001 (GB/T) / 15A AC": ["Bharat DC-001", "Bharat AC-001"],
}

def parse_vehicle_connectors(raw) -> list[str]:
    """Map the catalog's connector text to vehicle inlet types. Unknown text -> AC-only (conservative)."""
    raw_s = str(raw or "").strip()
    if raw_s in _CATALOG_CONNECTOR_MAP:
        return list(_CATALOG_CONNECTOR_MAP[raw_s])
    out: list[str] = []
    if "CCS" in raw_s:
        out += ["CCS2 DC", "Type-2 AC"]
    elif "Type-2" in raw_s or "Type 2" in raw_s:
        out.append("Type-2 AC")
    if "Bharat DC" in raw_s or "GB/T" in raw_s:
        out.append("Bharat DC-001")
    if "Bharat AC" in raw_s:
        out.append("Bharat AC-001")
    return out or ["Type-2 AC"]


def _num(x, default: float = 0.0) -> float:
    """float() that maps missing, non-finite and malformed values to a default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(v) else v


def vehicle_power_limit(sim_power_kw, max_ac_kw, max_dc_kw, charger_connector) -> float:
    """Vehicle-side power limit on a given charger: DC chargers use the DC limit, others the AC limit.
    A missing/zero OEM figure means 'unknown', so it does not further restrict the session limit."""
    limit = float(sim_power_kw)
    cap = _num(max_dc_kw if "DC" in str(charger_connector) else max_ac_kw, 0.0)
    return min(limit, cap) if cap > 0 else limit


def calculate_power_limit(
    vehicle_power_kw=None,
    charger_rated_kw=None,
    station_headroom_kw=None,
    session_limit_kw=None,
    *args,
    **kwargs,
) -> float:
    """
    Calculate effective charging power limit subject to multi-tier physical constraints:
        P_eff = min(P_vehicle, P_charger, P_station_headroom, P_session_limit)

    Compatible with all calling conventions:
    1. calculate_power_limit(sim_power_kw, max_ac_kw, max_dc_kw, charger_connector)
       (delegates to vehicle_power_limit if 4 positional args with connector string)
    2. calculate_power_limit(vehicle_power_kw, charger_rated_kw, station_headroom_kw, session_limit_kw)
    3. Keyword arguments: calculate_power_limit(vehicle_max_kw=..., charger_rated_kw=..., etc.)
    """
    # Check if called as vehicle_power_limit(sim_power_kw, max_ac_kw, max_dc_kw, charger_connector)
    if (len(args) >= 1 and isinstance(args[-1], str)) or isinstance(session_limit_kw, str):
        sim_power = _num(vehicle_power_kw, 0.0)
        max_ac = _num(charger_rated_kw, 0.0)
        max_dc = _num(station_headroom_kw, 0.0)
        conn = str(session_limit_kw if isinstance(session_limit_kw, str) else args[-1])
        return vehicle_power_limit(sim_power, max_ac, max_dc, conn)

    if "charger_connector" in kwargs:
        sim_power = _num(kwargs.get("sim_power_kw", vehicle_power_kw), 0.0)
        max_ac = _num(kwargs.get("max_ac_kw", charger_rated_kw), 0.0)
        max_dc = _num(kwargs.get("max_dc_kw", station_headroom_kw), 0.0)
        conn = str(kwargs["charger_connector"])
        return vehicle_power_limit(sim_power, max_ac, max_dc, conn)

    # Multi-tier power limit calculation: min of all positive non-zero limits
    candidates = []
    for val in [vehicle_power_kw, charger_rated_kw, station_headroom_kw, session_limit_kw] + list(args) + list(kwargs.values()):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            num = _num(val, default=-1.0)
            if num > 0:
                candidates.append(num)

    if candidates:
        return float(min(candidates))

    return _num(vehicle_power_kw, default=0.0)


def calculate_charging_duration(energy_required_kwh: float, charging_power_kw: float, efficiency: float = 0.90) -> float:
    """Calculate uninterrupted charging duration in minutes given energy demand (kWh), effective power (kW), and efficiency."""
    e = _num(energy_required_kwh, 0.0)
    p = _num(charging_power_kw, 0.0)
    eff = _num(efficiency, 0.90)
    if p <= 0 or eff <= 0:
        return float("inf")
    return round((e / (p * eff)) * 60.0, 2)


def calculate_required_energy(battery_capacity_kwh: float, target_soc: float, current_soc: float) -> float:
    """Calculate net required energy (kWh) to charge from current_soc to target_soc."""
    cap = _num(battery_capacity_kwh, 0.0)
    t_soc = _num(target_soc, 80.0)
    c_soc = _num(current_soc, 20.0)
    delta_soc = max(0.0, t_soc - c_soc)
    return round(cap * (delta_soc / 100.0), 3)


calculate_energy_required = calculate_required_energy


def calculate_available_window(arrival_time, departure_time) -> float:
    """Calculate available charging window in minutes between arrival and departure."""
    t_arr = pd.to_datetime(arrival_time)
    t_dep = pd.to_datetime(departure_time)
    return max(0.0, (t_dep - t_arr).total_seconds() / 60.0)


def check_feasibility(energy_required_kwh: float, charging_power_kw: float, available_window_min: float, efficiency: float = 0.90) -> bool:
    """Return True if session can theoretically complete within the available window at maximum power."""
    duration = calculate_charging_duration(energy_required_kwh, charging_power_kw, efficiency)
    window = _num(available_window_min, 0.0)
    return duration <= window


is_charging_feasible = check_feasibility


# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------

def _assert_columns(df: pd.DataFrame, required: set, sheet_name: str) -> None:
    """Raise a clear ValueError if any required column is missing."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {sorted(missing)}.\n"
            f"Present columns: {sorted(df.columns)}"
        )

@st.cache_data(show_spinner=False)
def load_all_sheets() -> dict[str, pd.DataFrame]:
    """Load every sheet from the Excel dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    xf = pd.ExcelFile(DATASET_PATH)
    return {name: pd.read_excel(DATASET_PATH, sheet_name=name) for name in xf.sheet_names}

@st.cache_data(show_spinner=False)
def load_sessions(scenario_mode: str = "Combined") -> pd.DataFrame:
    """
    Load, validate, and clean charging sessions.
    scenario_mode options:
      - 'Combined': All 96 sessions (Baseline 48 + Extended 48)
      - 'Baseline': Original 48 benchmark sessions (Residential, Workplace, Public)
      - 'Extended': 48 extended sessions (Highway Fast Hub, Urban Fleet Depot)
    """
    sheets = load_all_sheets()
    sheet_map = {
        "Combined": "EV_Sessions_Combined",
        "Baseline": "EV_Sessions_48",
        "Extended": "EV_Sessions_Extended",
    }
    target_sheet = sheet_map.get(scenario_mode, "EV_Sessions_Combined")
    if target_sheet not in sheets:
        target_sheet = "EV_Sessions_48"

    df = sheets[target_sheet].copy()

    # 1. Schema check
    _assert_columns(df, REQUIRED_SESSIONS_COLS, target_sheet)

    # 2. Date parsing
    for col in ["connection_time_IST", "departure_time_IST", "arrival_time_IST", "deadline_time_IST"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Reject unparseable timestamps (errors="coerce" turns them into NaT, which would otherwise slip through)
    for _col in ("connection_time_IST", "departure_time_IST"):
        if df[_col].isna().any():
            raise ValueError(f"Unparseable {_col} in sessions: {df.loc[df[_col].isna(), 'session_id'].tolist()}")

    # Validate departure after arrival
    bad_window = df["departure_time_IST"] <= df["connection_time_IST"]
    if bad_window.any():
        bad_ids = df.loc[bad_window, "session_id"].tolist()
        raise ValueError(f"Sessions with departure_time_IST <= connection_time_IST: {bad_ids}")

    # 3. Numeric conversions
    numeric_cols = [
        "energy_required_kwh", "charging_power_limit_kw", "station_power_limit_kw",
        "battery_capacity_kwh_scenario", "soc_at_arrival_pct_scenario",
        "target_soc_pct_scenario", "requested_energy_kwh_scenario",
        "charger_rated_power_kw_reference", "vehicle_max_charge_power_kw_scenario",
        "effective_session_power_limit_kw", "station_aggregate_power_cap_kw_scenario",
        "available_window_min", "battery_capacity_kwh_verified",
        "max_ac_charge_power_kw", "max_dc_charge_power_kw",
        "model_adjusted_effective_power_kw",
        "estimated_uninterrupted_duration_min_requested",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Positive validation (NaN must fail too: NaN <= 0 is False)
    _nan = df[["energy_required_kwh", "charging_power_limit_kw"]].isna().any(axis=1)
    if _nan.any():
        raise ValueError(f"Missing energy/power values in sessions: {df.loc[_nan, 'session_id'].tolist()}")
    if (df["energy_required_kwh"] <= 0).any():
        bad = df.loc[df["energy_required_kwh"] <= 0, "session_id"].tolist()
        raise ValueError(f"Non-positive energy_required_kwh in sessions: {bad}")
    if (df["charging_power_limit_kw"] <= 0).any():
        bad = df.loc[df["charging_power_limit_kw"] <= 0, "session_id"].tolist()
        raise ValueError(f"Non-positive charging_power_limit_kw in sessions: {bad}")

    # 4. Standard aliases
    df = df.rename(columns={
        "battery_capacity_kwh_scenario": "battery_kwh_scenario",
        "soc_at_arrival_pct_scenario": "soc_arrival_pct",
        "target_soc_pct_scenario": "soc_target_pct",
        "connector_type_scenario": "connector_type",
        "vehicle_max_charge_power_kw_scenario": "vehicle_max_kw",
        "charger_rated_power_kw_reference": "charger_rated_kw",
        "model_adjusted_effective_power_kw": "effective_power_verified_kw",
    })

    df["arrival_time_IST"] = df["connection_time_IST"]
    df["deadline_time_IST"] = df["departure_time_IST"]
    df["requested_energy_kwh"] = df["energy_required_kwh"]
    df["effective_power_kw"] = df["charging_power_limit_kw"]
    df["station_power_cap_kw"] = df["station_power_limit_kw"]

    # 5. Epoch minutes
    epoch_anchor = df["connection_time_IST"].min()
    df["arrival_epoch_min"] = ((df["connection_time_IST"] - epoch_anchor).dt.total_seconds() / 60).round(1)
    df["deadline_epoch_min"] = ((df["departure_time_IST"] - epoch_anchor).dt.total_seconds() / 60).round(1)
    df["window_min"] = (df["deadline_epoch_min"] - df["arrival_epoch_min"]).clip(lower=0)
    # Every epoch-minute value is measured from this single anchor. Charts must add minutes to
    # THIS timestamp (not to an environment's own first arrival) to get correct wall-clock times.
    df["epoch_anchor_IST"] = epoch_anchor

    # 6. Feasibility is computed below, after sim_power_kw is known.

    # 7. Compatibility caveats
    df["compatibility_caveat"] = False
    df.loc[df["vehicle_model_id"].astype(str).str.contains("TVS-IQUBE"), "compatibility_caveat"] = True

    # 8. Effective simulation power
    df["sim_power_kw"] = df["effective_power_verified_kw"].combine_first(df["charging_power_limit_kw"]).clip(lower=0.1)

    # 9. Feasibility at BEST-CASE power: the fastest compatible charger at the session's station,
    #    limited by the vehicle's own AC/DC limit, with no queueing and 90% efficiency.
    #    (The old check used the raw session limit, which the dataset sets above what some vehicles
    #    accept, so it under-reported infeasible sessions.)
    _stations = load_stations()

    def _best_power(row) -> float:
        ch = _stations[_stations["station_id"] == row["station_id"]]
        best = 0.0
        for conn, rated in zip(ch["connector_type"], ch["rated_power_kw"]):
            if is_connector_compatible(row["connector_type"], conn):
                p = min(vehicle_power_limit(row["sim_power_kw"], row.get("max_ac_charge_power_kw"),
                                            row.get("max_dc_charge_power_kw"), conn), _num(rated))
                best = max(best, p)
        return best

    df["feasibility_power_kw"] = df.apply(_best_power, axis=1)
    _p = df["feasibility_power_kw"].where(df["feasibility_power_kw"] > 0)
    df["min_duration_needed_min"] = (df["energy_required_kwh"] / (_p * 0.9) * 60).fillna(float("inf"))
    df["theoretically_feasible"] = df["min_duration_needed_min"] <= df["window_min"]

    return df

@st.cache_data(show_spinner=False)
def load_stations() -> pd.DataFrame:
    """Load and validate Station_Configs across all environments."""
    sheets = load_all_sheets()
    df = sheets["Station_Configs"].copy()
    _assert_columns(df, REQUIRED_STATIONS_COLS, "Station_Configs")
    df["rated_power_kw"] = pd.to_numeric(df["rated_power_kw"], errors="coerce")
    df["station_aggregate_power_cap_kw_scenario"] = pd.to_numeric(
        df["station_aggregate_power_cap_kw_scenario"], errors="coerce"
    )
    return df

@st.cache_data(show_spinner=False)
def load_model_catalog() -> pd.DataFrame:
    """Load comprehensive 56-vehicle Indian EV catalogue."""
    sheets = load_all_sheets()
    df = sheets["EV_Model_Catalog"].copy()
    _assert_columns(df, REQUIRED_CATALOG_COLS, "EV_Model_Catalog")
    for col in ["battery_capacity_kwh_verified", "usable_capacity_kwh", "max_ac_charge_power_kw", "max_dc_charge_power_kw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_comparison_summary() -> pd.DataFrame:
    """Load Comparison_Summary sheet."""
    sheets = load_all_sheets()
    df = sheets["Comparison_Summary"].copy()
    numeric_cols = [
        "session_count", "mean_waiting_time_min", "median_waiting_time_min",
        "total_requested_energy_kwh", "total_delivered_energy_kwh",
        "total_missed_energy_kwh", "completed_sessions",
        "completion_rate_pct", "peak_station_load_kw",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def get_sessions_for_env(env: str, sessions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter sessions by environment, or return all sessions if 'All Environments' is selected."""
    if sessions is None:
        sessions = load_sessions()
    if env in ("All Environments", "All Environments (Full Dataset)", "All"):
        return sessions.copy().reset_index(drop=True)
    return sessions[sessions["environment"] == env].copy().reset_index(drop=True)

def get_stations_for_env(env: str, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter stations by environment, or return all stations if 'All Environments' is selected."""
    if stations is None:
        stations = load_stations()
    if env in ("All Environments", "All Environments (Full Dataset)", "All"):
        return stations.copy().reset_index(drop=True)
    return stations[stations["environment"] == env].copy().reset_index(drop=True)

# ---------------------------------------------------------------------------
# Live Charging Management & Catalogue Query Helpers
# ---------------------------------------------------------------------------

LIVE_QUEUE_PATH = Path(__file__).parent / "data" / "live_queue.json"

def get_oem_list() -> list[str]:
    """Get sorted list of unique EV manufacturers from EV_Model_Catalog."""
    cat = load_model_catalog()
    return sorted([str(x) for x in cat["manufacturer"].dropna().unique() if str(x).strip()])

def get_models_for_oem(oem: str) -> list[str]:
    """Get list of model variants for a given manufacturer."""
    cat = load_model_catalog()
    matched = cat[cat["manufacturer"] == oem]
    if matched.empty:
        return []
    variants = matched["model_variant"].dropna().unique().tolist()
    return sorted([str(v) for v in variants if str(v).strip()])

def lookup_vehicle_specs(model_variant: str) -> dict:
    """Look up verified technical specifications for a model variant (no invented connectors)."""
    cat = load_model_catalog()
    row = cat[cat["model_variant"] == model_variant]
    if row.empty:
        row = cat[cat["model_name"].astype(str).str.contains(model_variant, regex=False, case=False)]
    if row.empty:
        return {
            "model_id": "UNKNOWN",
            "model_name": model_variant,
            "battery_capacity_kwh": 30.0,
            "max_ac_charge_power_kw": 7.2,
            "max_dc_charge_power_kw": 0.0,
            "supported_connectors": ["Type-2 AC"],
            "category": "Passenger Car (4W)",
        }
    r = row.iloc[0]
    return {
        "model_id": str(r.get("model_id") or "EV-CUSTOM"),
        "model_name": str(r.get("model_name") or model_variant),
        "battery_capacity_kwh": _num(r.get("battery_capacity_kwh_verified"), 30.0),
        "max_ac_charge_power_kw": _num(r.get("max_ac_charge_power_kw"), 0.0),
        "max_dc_charge_power_kw": _num(r.get("max_dc_charge_power_kw"), 0.0),
        "supported_connectors": parse_vehicle_connectors(r.get("charging_connector")),
        "category": str(r.get("vehicle_category") or "Passenger Car (4W)"),
        "dc_supported": str(r.get("dc_fast_charging_supported") or "No"),
    }


def get_station_catalog(stations: pd.DataFrame | None = None) -> list[dict]:
    """Stations derived from Station_Configs (never hard-coded IDs)."""
    if stations is None:
        stations = load_stations()
    out = []
    for sid, g in stations.groupby("station_id", sort=False):
        env = str(g["environment"].iloc[0])
        cap = float(g["station_aggregate_power_cap_kw_scenario"].iloc[0])
        out.append({
            "station_id": str(sid), "environment": env, "cap_kw": cap, "n_chargers": int(len(g)),
            "label": f"{sid} — {env} ({cap:.0f} kW cap, {len(g)} chargers)",
        })
    return out


def next_live_session_id(queue: list[dict]) -> str:
    """Legacy human-readable ticket label; never use this value as a session ID."""
    import re
    nums = []
    for q in queue:
        m = re.search(r"(\d+)$", str(q.get("session_id", "")))
        if m:
            nums.append(int(m.group(1)))
    return f"LIVE-EV-{(max(nums) + 1 if nums else 1):03d}"


def new_live_request_ids() -> tuple[str, str]:
    """Return separate, globally unique identifiers for a request and its session."""
    return f"req_{uuid.uuid4().hex}", f"ses_{uuid.uuid4().hex}"


# (model_id from EV_Model_Catalog, registration, SOC arrival %, SOC target %, arrives +min, stays min, note)
_DEMO_FLEET_SPEC = [
    # Simultaneous arrivals intentionally demonstrate parallel assignment on the first tick.
    ("CAR-TATA-NEXON-45",      "DL-01-EV-2026", 20, 80,  0,  90, "Simulated fleet demonstration"),
    ("CAR-HYUNDAI-IONIQ5",     "MH-12-AB-9876", 35, 85,  0,  90, "Simulated fleet demonstration"),
    ("CAR-TATA-TIAGO-LR",      "KA-03-EV-1122", 15, 90,  0,  90, "Simulated fleet demonstration"),
    ("CAR-MG-WINDSOR-38",      "HR-26-CD-4455", 30, 80,  0,  90, "Simulated fleet demonstration"),
    ("CAR-MAHINDRA-XUV400-39", "UP-16-EV-7890", 25, 85,  0,  90, "Simulated fleet demonstration"),
    ("3W-MAHINDRA-TREO",       "DL-04-AB-3344", 20, 95,  0,  90, "Simulated fleet demonstration"),
]


def get_default_live_fleet(station_id: str | None = None,
                           stations_df: pd.DataFrame | None = None) -> list[dict]:
    """Build a demo fleet from catalog rows and the supplied facility infrastructure.

    The bundled public station is used only when callers omit infrastructure for legacy scripts.
    """
    cat = load_model_catalog()
    chargers = stations_df.copy() if stations_df is not None else load_stations()
    if station_id is not None:
        chargers = chargers[chargers["station_id"].astype(str) == str(station_id)]
    if "charger_status" in chargers.columns:
        chargers = chargers[chargers["charger_status"].eq("available")]
    if "status" in chargers.columns:
        chargers = chargers[chargers["status"].eq("available")]
    if "station_status" in chargers.columns:
        chargers = chargers[chargers["station_status"].eq("available")]
    present = set(chargers["connector_type"])
    now_base = pd.Timestamp.now().floor("min")
    fleet: list[dict] = []
    for mid, reg, s0, s1, arr_after, stay, note in _DEMO_FLEET_SPEC:
        row = cat[cat["model_id"] == mid]
        if row.empty:
            continue
        r = row.iloc[0]
        conns = parse_vehicle_connectors(r["charging_connector"])
        conn = (next((c for c in conns if c in present), None)
                or next((c for c in conns if any(is_connector_compatible(c, k) for k in present)), None))
        if conn is None:
            continue
        ac, dc = _num(r["max_ac_charge_power_kw"]), _num(r["max_dc_charge_power_kw"])
        p_lim = dc if ("DC" in conn and dc > 0) else (ac if ac > 0 else 7.2)
        bat = _num(r["battery_capacity_kwh_verified"], 30.0)
        request_id, session_id = new_live_request_ids()
        fleet.append({
            "request_id": request_id,
            "session_id": session_id,
            "vehicle_reg_no": reg,
            "manufacturer": str(r["manufacturer"]),
            "model_name": str(r["model_name"]),
            "model_variant": str(r["model_variant"]),
            "vehicle_model_id": mid,
            "battery_capacity_kwh": bat,
            "soc_arrival_pct": float(s0),
            "soc_target_pct": float(s1),
            "energy_required_kwh": round(bat * (s1 - s0) / 100.0, 3),
            "max_charge_power_kw": p_lim,
            "veh_max_ac_kw": ac,
            "veh_max_dc_kw": dc,
            "connector_type": conn,
            "station_id": str(station_id or (chargers["station_id"].iloc[0] if not chargers.empty else "")),
            "arrival_time_IST": str(now_base + pd.Timedelta(minutes=arr_after)),
            "departure_time_IST": str(now_base + pd.Timedelta(minutes=arr_after + stay)),
            "status": "Queued",
            "record_type": "demo",
            "notes": "SIMULATED DEMONSTRATION DATA — not live vehicle or charger telemetry.",
        })
    return fleet


def merge_live_demo_requests(existing: list[dict], incoming: list[dict]) -> tuple[list[dict], int]:
    """Add only demo vehicles not already active at the same configured station."""
    merged = [dict(request) for request in existing]
    active_statuses = {"completed", "cancelled", "failed", "duplicate review"}
    active_keys = {
        (str(q.get("station_id", "")), str(q.get("vehicle_reg_no", "")).strip().upper())
        for q in merged
        if str(q.get("status", "Queued")).strip().lower() not in active_statuses
    }
    legacy_demo_notes = {
        "Fast transit charging", "Highway transit stop", "Office parking daytime",
        "Mall shopper top-up", "Ride-hailing fleet top-up",
    }
    demo_by_key = {
        (str(q.get("station_id", "")), str(q.get("vehicle_reg_no", "")).strip().upper()): q
        for q in incoming
    }
    # Upgrade known untagged sample rows from older VoltFlow versions without
    # replacing their persistent IDs or dropping their request records.
    for request in merged:
        key = (str(request.get("station_id", "")), str(request.get("vehicle_reg_no", "")).strip().upper())
        demo = demo_by_key.get(key)
        if (demo and not request.get("record_type")
                and str(request.get("vehicle_model_id", "")) == str(demo.get("vehicle_model_id", ""))
                and str(request.get("notes", "")) in legacy_demo_notes):
            request["record_type"] = "demo"
            request["data_origin"] = "Simulated demonstration data"

    added = 0
    for request in incoming:
        key = (str(request.get("station_id", "")), str(request.get("vehicle_reg_no", "")).strip().upper())
        if key in active_keys:
            continue
        merged.append(dict(request))
        active_keys.add(key)
        added += 1
    return merged, added

def load_live_queue() -> list[dict]:
    """Load the operator's saved queue. Start empty on first launch."""
    import json
    if LIVE_QUEUE_PATH.exists():
        try:
            with open(LIVE_QUEUE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (OSError, ValueError, TypeError):
            return []
    save_live_queue([])
    return []

def save_live_queue(queue: list[dict]) -> None:
    """Save live queue to disk persistently."""
    import json
    LIVE_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LIVE_QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, default=str)

def live_queue_to_sessions_df(queue: list[dict], station_power_limit_kw: float = 100.0) -> pd.DataFrame:
    """Convert live queue dicts to a validated DataFrame compatible with the scheduling engine."""
    if not queue:
        return pd.DataFrame()
    rows = []
    for q in queue:
        try:
            arr_t = pd.Timestamp(q["arrival_time_IST"])
            dep_t = pd.Timestamp(q["departure_time_IST"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each live request needs valid arrival and departure times.") from exc
        if pd.isna(arr_t) or pd.isna(dep_t) or dep_t <= arr_t:
            raise ValueError(f"Request {q.get('session_id', '<unknown>')} needs a departure after arrival.")

        def required_number(key: str, label: str) -> float:
            try:
                value = float(q[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Request {q.get('session_id', '<unknown>')} has invalid {label}.") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Request {q.get('session_id', '<unknown>')} requires positive finite {label}.")
            return value

        e_req = required_number("energy_required_kwh", "requested energy")
        p_lim = required_number("max_charge_power_kw", "vehicle power limit")
        v_bat = required_number("battery_capacity_kwh", "battery capacity")
        c_type = str(q.get("connector_type") or "Type-2 AC")
        soc_arrival = float(q.get("soc_arrival_pct", 20.0))
        soc_target = float(q.get("soc_target_pct", 80.0))
        if not math.isfinite(soc_arrival) or not math.isfinite(soc_target) or not (0 <= soc_arrival < soc_target <= 100):
            raise ValueError(f"Request {q.get('session_id', '<unknown>')} has an invalid SOC range.")

        # Vehicle AC/DC limits: use the catalog values stored at registration.
        # Legacy queue items (saved before this change) fall back to the old assumption.
        max_ac = _num(q.get("veh_max_ac_kw"), -1.0)
        max_dc = _num(q.get("veh_max_dc_kw"), -1.0)
        if max_ac < 0 or max_dc < 0:
            if "DC" in c_type:
                max_dc, max_ac = p_lim, 7.2
            else:
                max_ac, max_dc = p_lim, 0.0

        rows.append({
            "request_id": str(q.get("request_id") or ""),
            "session_id": str(q["session_id"]),
            "vehicle_reg_no": str(q.get("vehicle_reg_no", "DL-01-EV-0000")),
            "environment": "Configured Facility Simulation",
            # Do not silently route requests with missing facility configuration to a
            # bundled demonstration station. Such requests remain unassigned until
            # an operator selects a station in the facility queue.
            "station_id": str(q.get("station_id") or ""),
            "charger_id": "",  # To be dynamically assigned by scheduler
            "connection_time_IST": arr_t,
            "departure_time_IST": dep_t,
            "arrival_time_IST": arr_t,
            "deadline_time_IST": dep_t,
            "energy_required_kwh": round(e_req, 3),
            "requested_energy_kwh": round(e_req, 3),
            "charging_power_limit_kw": round(p_lim, 2),
            "effective_power_kw": round(p_lim, 2),
            "sim_power_kw": round(p_lim, 2),
            "station_power_limit_kw": station_power_limit_kw,
            "station_power_cap_kw": station_power_limit_kw,
            "vehicle_model_id": str(q.get("vehicle_model_id") or "LIVE-EV"),
            "model_id": str(q.get("vehicle_model_id") or "LIVE-EV"),
            "manufacturer": str(q.get("manufacturer") or "Generic"),
            "model_name": str(q.get("model_name") or "EV"),
            "model_variant": str(q.get("model_variant") or "EV Standard"),
            "battery_capacity_kwh_verified": v_bat,
            "battery_kwh_verified": v_bat,
            "soc_arrival_pct": soc_arrival,
            "soc_target_pct": soc_target,
            "connector_type": c_type,
            "connector_type_scenario": c_type,
            "vehicle_max_kw": p_lim,
            "max_ac_charge_power_kw": max_ac,
            "max_dc_charge_power_kw": max_dc,
            "compatibility_caveat": False,
            "status": str(q.get("status", "Queued")),
            "notes": str(q.get("notes", "")),
        })

    df = pd.DataFrame(rows)
    epoch_anchor = df["connection_time_IST"].min()
    df["arrival_epoch_min"] = ((df["connection_time_IST"] - epoch_anchor).dt.total_seconds() / 60.0).round(1)
    df["deadline_epoch_min"] = ((df["departure_time_IST"] - epoch_anchor).dt.total_seconds() / 60.0).round(1)
    df["window_min"] = (df["deadline_epoch_min"] - df["arrival_epoch_min"]).clip(lower=0.1)
    df["epoch_anchor_IST"] = epoch_anchor
    df["min_duration_needed_min"] = (df["energy_required_kwh"] / (df["charging_power_limit_kw"] * 0.9) * 60.0).round(1)
    df["theoretically_feasible"] = df["min_duration_needed_min"] <= df["window_min"]
    return df


# ---------------------------------------------------------------------------
# Data-quality report (computed from the data, never hard-coded)
# ---------------------------------------------------------------------------

def session_quality_report(sessions: pd.DataFrame, stations: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per detected problem: Session ID, Check, Detail."""
    if stations is None:
        stations = load_stations()
    cp = stations.set_index("charger_id")
    rows = []
    for _, r in sessions.iterrows():
        sid = r["session_id"]
        lim = _num(r["charging_power_limit_kw"])
        oem = _num(r["max_dc_charge_power_kw"] if "DC" in str(r["connector_type"]) else r["max_ac_charge_power_kw"])
        if oem > 0 and lim > oem + 1e-9:
            rows.append((sid, "Session limit above vehicle OEM limit",
                         f"{lim:g} kW given; vehicle accepts {oem:g} kW on {r['connector_type']} (simulation uses {r['sim_power_kw']:g} kW)"))
        pc = str(r.get("charger_id", ""))
        if pc in cp.index and lim > _num(cp.loc[pc, "rated_power_kw"]) + 1e-9:
            rows.append((sid, "Session limit above its scenario charger rating",
                         f"{lim:g} kW given; {pc} is rated {_num(cp.loc[pc, 'rated_power_kw']):g} kW"))
        if not bool(r["theoretically_feasible"]):
            need = r["min_duration_needed_min"]
            need_s = "no compatible charger" if need == float("inf") else f"{need:.0f} min needed"
            rows.append((sid, "Infeasible even at best-case power",
                         f"{need_s} vs {r['window_min']:.0f} min window"))
        if "tariff_inr_per_kwh" in sessions.columns and pd.isna(r.get("tariff_inr_per_kwh")):
            rows.append((sid, "No tariff supplied", "Energy cost cannot be computed for this session"))
    return pd.DataFrame(rows, columns=["Session ID", "Check", "Detail"])

