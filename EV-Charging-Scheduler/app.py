"""
app.py — VoltFlow: Intelligent EV Charging & Scheduling Platform
================================================================
Enterprise Dark SaaS EV Charging Management & Multi-Algorithm Dispatch Platform.
Engineered in accordance with Linear, Stripe, and Vercel design principles.

Complete Design Tokens:
  - Main Canvas Background:     #0B1014
  - Navigation Sidebar:          #0E151B
  - Primary Surface:             #121B22
  - Secondary Surface:           #18232B
  - Hover Surface:               #1D2A33
  - Primary Border:              #2A3943
  - Subtle Border:               #202D36
  - Primary Typography:          #E8F0F2
  - Secondary Typography:        #A9B8BE
  - Muted Typography:            #82939B
  - Primary Brand Accent:        #22C7A5
  - Accent Hover:                #36DDB3
  - Secondary Blue:              #76A9FF
  - Warning State:               #F4BE5E
  - Critical / Error State:      #F07878
  - Success State:               #22C7A5

Navigation Architecture:
  1. Overview           (Facility health, charging capacity, waiting vehicles, recent activity)
  2. Register EV        (Vehicle check-in)
  3. Live Queue         (Active requests and simulation progress)
  4. Charging Operations (Hybrid simulation and charger assignments)
  5. Analytics          (Facility results and separate research comparisons)
  6. Settings           (Facility identity, capacity and operating defaults)
"""

from __future__ import annotations
import io
import datetime
import copy
import math
import os
import time
import pandas as pd
import numpy as np
import streamlit as st

# ── 1. Page Configuration (Must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="VoltFlow | Intelligent EV Charging & Scheduling Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Module Imports ─────────────────────────────────────────────────────────
from data_loader import (
    load_sessions, load_stations, load_model_catalog, load_comparison_summary,
    get_sessions_for_env, get_stations_for_env, humanize_dataframe,
    CONNECTOR_COMPAT, DISPLAY_COLUMNS,
    get_oem_list, get_models_for_oem, lookup_vehicle_specs,
    load_live_queue as load_legacy_live_queue, get_default_live_fleet, merge_live_demo_requests,
    live_queue_to_sessions_df,
    is_connector_compatible, vehicle_power_limit, calculate_power_limit, get_station_catalog,
    new_live_request_ids, session_quality_report,
)
from facility_store import (
    initialize as initialize_facility_store, list_facilities, get_facility,
    get_station_frame, save_configuration, archive_facility,
    load_requests, save_requests, save_simulation_snapshot, save_schedule, save_dispatch_state, load_latest_schedule,
    audit_history, facility_data_summary, slugify, connect as connect_facility_store,
    reset_operational_data, normalize_request_records,
)
from scheduler import (
    run_all_algorithms, run_single_algorithm, results_to_dataframe,
    compute_summary, compute_charger_utilization, snapshot_at, TEST_COUNT, _run_tests,
    build_hybrid_priority_list, validate_hybrid_weights, HYBRID_DEFAULT_WEIGHTS,
    energy_delivered_at,
)
from dispatch_service import dispatchable_requests, dispatch_key, calculate_live_dispatch
from visualizations import (
    fig_gantt, fig_station_load, fig_energy_comparison,
    fig_waiting_time, fig_completion_rate, fig_missed_energy,
    fig_session_scatter, fig_radar_summary, fig_session_timeline_bar,
    fig_live_gantt, fig_live_power_profile, fig_charger_utilization,
    ALGO_COLORS, ALGO_ORDER,
)


def sync_queue_to_simulation(queue, results, minute, station_frame=None, epoch_anchor=None):
    """Persist request progress/status for the currently selected simulation snapshot."""
    if not results or minute is None:
        return False
    result_by_id = {result.session_id: result for result in results}
    snapshot = snapshot_at(results, float(minute))
    changed = False
    for request in queue:
        result = result_by_id.get(str(request.get("session_id", "")))
        if result is None or str(request.get("status", "")).lower() in {
            "cancelled", "duplicate review"
        }:
            continue
        delivered = energy_delivered_at(result, float(minute))
        battery = float(request.get("battery_capacity_kwh") or result.battery_kwh_verified or 0.0)
        current_soc = min(100.0, result.soc_arrival_pct + (100.0 * delivered / battery if battery > 0 else 0.0))
        reason = ""
        if result.completed_by_deadline and result.charge_end_min is not None and result.charge_end_min <= minute:
            status = "Completed"
        elif result.deadline_min <= minute:
            status = "Failed"
        elif result.arrival_min > minute:
            status = "Waiting"
            reason = "Future arrival — dispatch waits until the registered arrival time."
        elif any(active_result.session_id == result.session_id
                 for active_result, _ in snapshot["charging"].values()):
            status = "Charging"
            reason = "Charging on the assigned endpoint."
        elif result.charger_id_assigned and result.charge_start_min is not None and result.charge_start_min > minute:
            status = "Assigned"
            start_label = (epoch_anchor + datetime.timedelta(minutes=result.charge_start_min)).strftime("%H:%M") if epoch_anchor is not None else f"in {result.charge_start_min - minute:.0f} min"
            reason = f"Assigned to {result.charger_id_assigned}; charging starts at {start_label}."
        else:
            status = "Waiting"
            reason = _waiting_reason(request, result, minute, station_frame, snapshot)
        assigned = result.charger_id_assigned if result.arrival_min <= minute else None
        active_session = next((value for value in snapshot["charging"].values()
                               if value[0].session_id == result.session_id), None)
        power = active_session[1] if active_session else 0.0
        updates = {
            "status": status,
            "waiting_reason": "" if status in ("Charging", "Completed", "Failed") else reason,
            "energy_delivered_kwh": round(delivered, 4),
            "current_soc_pct": round(current_soc, 2),
            "assigned_charger_id": assigned,
            "allocated_power_kw": round(float(power), 3),
            "simulation_minute": float(minute),
        }
        for key, value in updates.items():
            if request.get(key) != value:
                request[key] = value
                changed = True
    return changed


def _waiting_reason(request, result, minute, station_frame, snapshot):
    if station_frame is None or station_frame.empty:
        return "No charger infrastructure is configured at this station."
    station = station_frame[station_frame["station_id"].astype(str) == str(request.get("station_id", ""))]
    compatible = station[station["connector_type"].apply(
        lambda connector: is_connector_compatible(str(request.get("connector_type", "")), str(connector)))]
    if compatible.empty:
        return "No compatible charger is configured at this station."
    available = compatible[
        compatible["charger_status"].astype(str).str.lower().eq("available")
        & compatible["station_status"].astype(str).str.lower().eq("available")
    ]
    if available.empty:
        return "Compatible chargers are offline or under maintenance."
    port_ids = set(available["charger_id"].astype(str))
    if port_ids and port_ids.issubset(set(snapshot["occupied"])):
        return "All compatible chargers are occupied; dispatch retries as an endpoint frees."
    if result.charger_id_assigned is None:
        return "Insufficient station or facility power capacity for this request right now."
    return "Waiting for an eligible endpoint and available power capacity."


def _simulation_timeline_options(results, interval_min=5, include_minute=None):
    """Return replay ticks plus exact arrival, start, completion and deadline events."""
    if not results:
        return []
    start = min(float(result.arrival_min) for result in results)
    end = max(float(result.deadline_min) for result in results)
    count = int(np.floor(max(0.0, end - start) / interval_min))
    options = {round(start + i * interval_min, 2) for i in range(count + 1)}
    options.add(round(end, 2))
    if include_minute is not None and start - 1e-6 <= float(include_minute) <= end + 1e-6:
        options.add(round(float(include_minute), 4))
    for result in results:
        for event in (result.arrival_min, result.charge_start_min,
                      result.charge_end_min, result.deadline_min):
            if event is not None and start - 1e-6 <= float(event) <= end + 1e-6:
                options.add(round(float(event), 2))
    return sorted(options)


def _simulation_event_minutes(results):
    events = set()
    for result in results or []:
        for event in (result.arrival_min, result.charge_start_min,
                      result.charge_end_min, result.deadline_min):
            if event is not None:
                events.add(round(float(event), 2))
    return sorted(events)


def _advance_simulation_clock(current_minute, elapsed_seconds, speed_multiplier, end_minute):
    """Advance the replay cursor by elapsed wall time, keeping it monotonic and bounded."""
    try:
        current = float(current_minute)
        elapsed = float(elapsed_seconds)
        speed = float(speed_multiplier)
        end = float(end_minute)
    except (TypeError, ValueError):
        return current_minute
    if not all(math.isfinite(value) for value in (current, elapsed, speed, end)):
        return current_minute
    if end <= current or elapsed <= 0 or speed <= 0:
        return current
    return min(end, current + elapsed * speed)


def _next_simulation_event(results, minute):
    """Return the next planned operational event as (simulation minute, readable label)."""
    if minute is None:
        return None
    events = []
    now = float(minute)
    for result in results or []:
        vehicle = str(getattr(result, "vehicle_reg_no", "Vehicle"))
        candidates = (
            (getattr(result, "arrival_min", None), "arrives"),
            (getattr(result, "charge_start_min", None), "charging starts"),
            (getattr(result, "charge_end_min", None), "charging completes"),
            (getattr(result, "deadline_min", None), "departure deadline"),
        )
        for event_minute, label in candidates:
            if event_minute is None:
                continue
            event_minute = float(event_minute)
            if event_minute > now + 1e-6:
                events.append((event_minute, f"{label} · {vehicle}"))
    return min(events, key=lambda event: event[0]) if events else None


def _format_simulation_time(epoch_anchor, minute, with_date=False):
    if epoch_anchor is None or minute is None:
        return "—"
    stamp = pd.Timestamp(epoch_anchor) + pd.to_timedelta(float(minute), unit="m")
    return stamp.strftime("%d %b · %H:%M" if with_date else "%H:%M")


def _request_arrival_label(request, result, epoch_anchor):
    if result is not None and epoch_anchor is not None:
        return _format_simulation_time(epoch_anchor, result.arrival_min)
    value = request.get("arrival_time_IST")
    try:
        return pd.Timestamp(value).strftime("%H:%M")
    except (TypeError, ValueError):
        return "—"


def _compact_charger_label(charger_id, station_id=""):
    if not charger_id:
        return "—"
    value = str(charger_id)
    endpoint = value.rsplit("-", 1)[-1]
    if endpoint.startswith("CP"):
        return f"{endpoint} · {station_id}" if station_id else endpoint
    return value


def _restored_snapshot_minute(queue):
    """Restore the active fleet's replay point without letting stale history jump it forward."""
    terminal = {"completed", "cancelled", "failed", "duplicate review"}
    active = [request for request in queue
              if str(request.get("status", "Queued")).strip().lower() not in terminal]
    source = active if active else queue
    saved = [float(request["simulation_minute"]) for request in source
             if request.get("simulation_minute") is not None]
    return max(saved) if saved else None

# ── 3. Enterprise Dark SaaS Theme CSS ─────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #E8F0F2;
}

/* Overall page background */
.stApp {
    background-color: #0B1014 !important;
}

/* Header bar */
[data-testid="stHeader"] {
    background: rgba(11, 15, 20, 0.85) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #202D36;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0E151B !important;
    border-right: 1px solid #202D36 !important;
}
[data-testid="stSidebar"] section {
    width: 240px !important;
}
[data-testid="stSidebar"] * {
    color: #E8F0F2 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #82939B !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Sidebar Navigation Items */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: #A9B8BE !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease-in-out !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #18232B !important;
    color: #E8F0F2 !important;
    border-color: #202D36 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[aria-checked="true"] {
    background: rgba(37, 201, 160, 0.1) !important;
    color: #22C7A5 !important;
    border-color: rgba(37, 201, 160, 0.28) !important;
    font-weight: 600 !important;
}

/* Main Container Dimensions */
.main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1500px !important;
}

/* Page Header */
.page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
    border-bottom: 1px solid #202D36;
    padding: 0.2rem 0 1.15rem;
    margin-bottom: 1.25rem;
}
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E8F0F2;
    letter-spacing: -0.025em;
    margin: 0;
}
.page-subtitle {
    font-size: 0.86rem;
    color: #82939B;
    margin: 4px 0 0 0;
}
.header-badge {
    color: #E8F0F2;
    font-size: 0.78rem;
    padding: 0.45rem 0.75rem;
    background: #121B22;
    border: 1px solid #2A3943;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}

/* Card Containers */
.saas-card {
    background: #121B22;
    border: 1px solid #202D36;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.15rem;
    transition: border-color 0.15s ease;
}
.saas-card:hover {
    border-color: #2A3943;
}
.saas-card-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #E8F0F2;
    letter-spacing: -0.01em;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Facility Context Banner */
.status-banner {
    background: #121B22;
    border: 1px solid #202D36;
    border-left: 3px solid #22C7A5;
    border-radius: 8px;
    padding: 0.75rem 1.15rem;
    margin-bottom: 1.25rem;
    color: #A9B8BE;
    font-size: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Metric KPI Containers */
div[data-testid="metric-container"] {
    background: #121B22 !important;
    border: 1px solid #202D36 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.15rem !important;
    transition: border-color 0.15s ease, transform 0.15s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: #2A3943 !important;
}
div[data-testid="metric-container"] label {
    color: #82939B !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8F0F2 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* Primary Hero Action Button (GO / RUN SCHEDULER / DISPATCH) */
button[kind="primary"] {
    background-color: #22C7A5 !important;
    color: #0B1014 !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    min-height: 48px !important;
    box-shadow: 0 2px 6px rgba(37, 201, 160, 0.25) !important;
    transition: all 0.18s ease-in-out !important;
}
button[kind="primary"]:hover {
    background-color: #36DDB3 !important;
    color: #0B1014 !important;
    box-shadow: 0 4px 14px rgba(54, 221, 179, 0.35) !important;
    transform: translateY(-1px) !important;
}
button[kind="primary"]:active {
    background-color: #1EB28C !important;
    transform: translateY(1px) !important;
}

/* Secondary Button */
button[kind="secondary"] {
    background-color: #18232B !important;
    color: #E8F0F2 !important;
    border: 1px solid #2A3943 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    min-height: 40px !important;
    transition: all 0.18s ease-in-out !important;
}
button[kind="secondary"]:hover {
    background-color: #1D2A33 !important;
    border-color: #76A9FF !important;
    color: #E8F0F2 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.15rem !important;
    color: #82939B !important;
    background: transparent !important;
    border: none !important;
    transition: color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #E8F0F2 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #22C7A5 !important;
    border-bottom: 2px solid #22C7A5 !important;
}

/* Status Badges */
.badge-pill {
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    display: inline-block;
    letter-spacing: 0.02em;
}
.badge-success { background: rgba(53, 201, 149, 0.12); color: #22C7A5; border: 1px solid rgba(53, 201, 149, 0.25); }
.badge-info { background: rgba(87, 153, 247, 0.12); color: #76A9FF; border: 1px solid rgba(87, 153, 247, 0.25); }
.badge-warning { background: rgba(245, 184, 75, 0.12); color: #F4BE5E; border: 1px solid rgba(245, 184, 75, 0.25); }
.badge-danger { background: rgba(241, 109, 117, 0.12); color: #F07878; border: 1px solid rgba(241, 109, 117, 0.25); }
.badge-neutral { background: #18232B; color: #A9B8BE; border: 1px solid #2A3943; }

/* Dispenser Hardware Cards */
.charger-card {
    background: #121B22;
    border: 1px solid #202D36;
    border-radius: 8px;
    padding: 0.95rem;
    margin-bottom: 0.65rem;
    transition: all 0.15s ease;
}
.charger-card:hover {
    border-color: #2A3943;
    background: #18232B;
}
.charger-card-avail {
    border-top: 2px solid #22C7A5;
}
.charger-card-busy {
    border-top: 2px solid #76A9FF;
}

/* Inputs and Selectors */
div[data-baseweb="select"] > div,
input[type="text"], input[type="number"] {
    background-color: #18232B !important;
    border: 1px solid #2A3943 !important;
    color: #E8F0F2 !important;
    border-radius: 7px !important;
}
div[data-baseweb="select"] > div:hover,
input[type="text"]:focus, input[type="number"]:focus {
    border-color: #22C7A5 !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #202D36;
    border-radius: 8px;
    overflow: hidden;
}

/* Lighter hierarchy: surfaces mark important work areas without boxing every section. */
.page-header { margin-bottom: 1.6rem; padding-bottom: 1rem; }
.page-title { font-size: clamp(1.45rem, 2.4vw, 1.9rem); font-weight: 650; }
.page-subtitle { max-width: 58rem; line-height: 1.55; }
.saas-card { background: transparent; border: 0; border-radius: 0; padding: 0; margin-bottom: 1rem; }
.saas-card:hover { border-color: transparent; }
.saas-card-header { border-bottom: 1px solid #202D36; padding-bottom: .55rem; margin-bottom: .85rem; }
div[data-testid="metric-container"] { background: transparent !important; border: 0 !important; border-bottom: 1px solid #202D36 !important; border-radius: 0 !important; padding: .55rem .45rem !important; }
div[data-testid="metric-container"]:hover { border-color: #22C7A5 !important; transform: none !important; }
.status-banner { flex-wrap: wrap; gap: .55rem 1.2rem; }

@media (max-width: 900px) {
    .main .block-container { padding-left: 1.1rem !important; padding-right: 1.1rem !important; }
    .page-header { align-items: flex-start; flex-direction: column; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
}
@media (max-width: 560px) {
    .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
    .page-title { font-size: 1.4rem; }
    [data-testid="stDataFrame"] { max-width: calc(100vw - 1.5rem); }
}
</style>
""", unsafe_allow_html=True)

# ── 4. Persistent multi-facility operational store ────────────────────────────
reference_stations = load_stations()
initialize_facility_store(reference_stations)
facility_choices = list_facilities()
if not facility_choices:
    st.error("No active facilities are configured. Create a facility in the local database to continue.")
    st.stop()

# Migrate the legacy queue once into facility-scoped storage without discarding requests.
facility_by_station = {}
for _f in facility_choices:
    for _sid in get_station_frame(_f["id"])["station_id"].astype(str).unique():
        facility_by_station[_sid] = _f["id"]
with connect_facility_store() as _con:
    _stored_request_count = _con.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
if _stored_request_count == 0:
    _legacy = load_legacy_live_queue()
    _by_facility = {}
    for _request in _legacy:
        _fid = facility_by_station.get(str(_request.get("station_id", "")))
        if _fid:
            _by_facility.setdefault(_fid, []).append(_request)
    for _fid, _requests in _by_facility.items():
        save_requests(_fid, _requests)

if "active_facility_id" not in st.session_state:
    st.session_state["active_facility_id"] = facility_choices[0]["id"]
if "switch_to_facility" in st.session_state:
    st.session_state["active_facility_id"] = st.session_state.pop("switch_to_facility")
if st.session_state["active_facility_id"] not in {f["id"] for f in facility_choices}:
    st.session_state["active_facility_id"] = facility_choices[0]["id"]

model_catalog = load_model_catalog()

# ── 5. Sidebar Navigation ─────────────────────────────────────────────────────
with st.sidebar:
    # Branding Header — VoltFlow: Intelligent EV Charging & Scheduling Platform
    st.markdown("""
    <div style="padding: 4px 0 14px 0; border-bottom: 1px solid #202D36; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(34, 199, 165, 0.12); border: 1px solid rgba(34, 199, 165, 0.32); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22C7A5" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path>
                </svg>
            </div>
            <div>
                <div style="font-size: 1.05rem; font-weight: 750; color: #E8F0F2; letter-spacing: -0.025em; line-height: 1.1;">Volt<span style="color: #22C7A5;">Flow</span></div>
                <div style="font-size: 0.65rem; color: #82939B; font-weight: 500; letter-spacing: 0.01em; margin-top: 3px; line-height: 1.2;">Intelligent EV Charging &amp; Scheduling</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Primary Navigation (Strictly preserves 'Primary navigation' label for smoke test compatibility)
    nav_page = st.radio(
        "Primary navigation",
        ["Overview", "Register EV", "Live Queue", "Charging Operations", "Analytics", "Settings"],
        index=0,
        label_visibility="collapsed",
    )
    analytics_view = "Performance"
    if nav_page == "Analytics":
        analytics_view = st.selectbox("Analytics view", ["Facility activity", "Algorithm comparison", "Reference data"])
        if analytics_view == "Reference data":
            nav_page = "Reference data"

    if nav_page == "Settings":
        settings_view = st.radio("Settings section", ["Facility", "Infrastructure"], horizontal=True,
                                 label_visibility="collapsed", key="settings_section")
        if settings_view == "Infrastructure":
            nav_page = "Charging Stations"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div style='font-size:0.72rem;font-weight:600;text-transform:uppercase;color:#82939B;letter-spacing:.05em;margin-bottom:6px;'>Active facility</div>", unsafe_allow_html=True)
    active_facility_id = st.selectbox(
        "Active Facility", [f["id"] for f in facility_choices],
        format_func=lambda fid: next(f["name"] for f in facility_choices if f["id"] == fid),
        key="active_facility_id", label_visibility="collapsed",
    )
    _facility_just_loaded = st.session_state.get("_loaded_facility_id") != active_facility_id
    if _facility_just_loaded:
        st.session_state["live_queue"] = load_requests(active_facility_id)
        st.session_state["operational_snapshot_minute"] = _restored_snapshot_minute(
            st.session_state["live_queue"])
        # Playback is a browser-session control; a facility switch/restart always
        # restores the persisted simulation paused at its last saved minute.
        st.session_state["simulation_playing"] = False
        st.session_state["_simulation_last_tick_monotonic"] = None
        st.session_state["live_policy_comparison"] = None
        _saved_schedule = load_latest_schedule(active_facility_id)
        st.session_state["live_schedule_results"] = _saved_schedule["results"] if _saved_schedule else None
        st.session_state["live_epoch_anchor"] = pd.Timestamp(_saved_schedule["epoch_anchor"]) if _saved_schedule else None
        st.session_state["live_algo"] = _saved_schedule["algorithm"] if _saved_schedule else "HYBRID"
        st.session_state["live_station_cap"] = _saved_schedule["grid_capacity_kw"] if _saved_schedule else None
        st.session_state["live_dispatch_key"] = _saved_schedule.get("dispatch_key") if _saved_schedule else None
        st.session_state["_dispatch_attempt_key"] = None
        st.session_state["_dispatch_error"] = None
        st.session_state["_loaded_facility_id"] = active_facility_id

    active_facility = get_facility(active_facility_id)
    facility_grid_cap = float(active_facility["grid_capacity_kw"])
    all_stations = get_station_frame(active_facility_id)
    station_ids = all_stations["station_id"].astype(str).drop_duplicates().tolist()
    if station_ids:
        if nav_page in ("Register EV", "Live Queue"):
            selected_st_id = st.selectbox("Check-in station", station_ids, key=f"active_station_{active_facility_id}")
        else:
            selected_st_id = st.session_state.get(f"active_station_{active_facility_id}", station_ids[0])
            if selected_st_id not in station_ids:
                selected_st_id = station_ids[0]
    else:
        selected_st_id = ""
    st_info = all_stations[all_stations["station_id"].astype(str) == selected_st_id].copy() if selected_st_id else all_stations.copy()
    st_env_name = active_facility["name"]
    st_default_cap = float(st_info["station_aggregate_power_cap_kw_scenario"].iloc[0]) if not st_info.empty else facility_grid_cap
    if "live_queue" not in st.session_state:
        st.session_state["live_queue"] = []
    if "live_schedule_results" not in st.session_state:
        st.session_state["live_schedule_results"] = None
    if "live_epoch_anchor" not in st.session_state:
        st.session_state["live_epoch_anchor"] = None

    # A long-lived Streamlit session may still hold a pre-migration queue object.
    # Repair duplicate legacy IDs once, retaining each vehicle request and flagging
    # duplicate active registrations for review before any preview/scheduler call.
    _identity_issues = normalize_request_records(st.session_state["live_queue"])[1]
    if _identity_issues:
        try:
            save_requests(active_facility_id, st.session_state["live_queue"])
            st.session_state["_dispatch_on_next_rerun"] = True
        except Exception as _identity_error:
            st.session_state["_dispatch_error"] = f"Queue identity repair could not be persisted: {_identity_error}"

    def save_live_queue(queue):
        save_requests(active_facility_id, queue)
        st.session_state["live_policy_comparison"] = None

    facility_stations = all_stations.copy()

    def attempt_automatic_dispatch(*, force=False, quiet=False,
                                   power_cap_override=None, step_override=None):
        """Dispatch only when scheduling inputs changed or the operator requests a re-run."""
        queue = st.session_state["live_queue"]
        active_requests = dispatchable_requests(queue)
        if not active_requests:
            return False
        algorithm = str(active_facility.get("scheduling_algorithm", "HYBRID")).upper()
        weights = {
            "FCFS": float(st.session_state.get("hybrid_w_fcfs", HYBRID_DEFAULT_WEIGHTS["FCFS"])),
            "EDF": float(st.session_state.get("hybrid_w_edf", HYBRID_DEFAULT_WEIGHTS["EDF"])),
            "FAIRNESS": float(st.session_state.get("hybrid_w_fairness", HYBRID_DEFAULT_WEIGHTS["FAIRNESS"])),
        }
        effective_cap = float(power_cap_override if power_cap_override is not None else facility_grid_cap)
        step = float(step_override if step_override is not None else active_facility.get("simulation_step_min", 1.0))
        try:
            key = dispatch_key(queue, facility_stations, effective_cap, algorithm, step, weights)
        except Exception as exc:
            st.session_state["_dispatch_error"] = str(exc)
            if not quiet:
                st.error(f"Automatic dispatch could not validate the queue: {exc}")
            return False
        if not force and (key == st.session_state.get("live_dispatch_key")
                          or key == st.session_state.get("_dispatch_attempt_key")):
            return False
        st.session_state["_dispatch_attempt_key"] = key
        try:
            plan = calculate_live_dispatch(queue, facility_stations, effective_cap,
                                           algorithm=algorithm, step_min=step,
                                           hybrid_weights=weights)
            working_queue = [copy.deepcopy(request) for request in queue]
            anchor = pd.Timestamp(plan["epoch_anchor"])
            first_minute = min(result.arrival_min for result in plan["results"])
            last_minute = max(result.deadline_min for result in plan["results"])
            old_anchor = st.session_state.get("live_epoch_anchor")
            old_minute = st.session_state.get("operational_snapshot_minute")
            if old_anchor is not None and old_minute is not None:
                absolute_time = pd.Timestamp(old_anchor) + pd.to_timedelta(float(old_minute), unit="m")
                current_minute = (absolute_time - anchor).total_seconds() / 60.0
                minute = min(last_minute, max(first_minute, current_minute))
            else:
                minute = first_minute
            sync_queue_to_simulation(working_queue, plan["results"], minute,
                                     facility_stations, anchor.to_pydatetime())
            save_dispatch_state(active_facility_id, working_queue, algorithm,
                                effective_cap, anchor, plan["results"], plan["dispatch_key"])
        except Exception as exc:
            # Keep both the old displayed schedule and the queued request data intact.
            st.session_state["_dispatch_error"] = str(exc)
            if not quiet:
                st.error(f"Dispatch failed safely; the queue and previous charging schedule were preserved. {exc}")
            return False

        st.session_state["live_queue"] = working_queue
        st.session_state["live_schedule_results"] = plan["results"]
        st.session_state["live_epoch_anchor"] = anchor
        st.session_state["live_algo"] = algorithm
        st.session_state["live_station_cap"] = effective_cap
        st.session_state["live_dispatch_key"] = plan["dispatch_key"]
        st.session_state["operational_snapshot_minute"] = minute
        st.session_state["live_policy_comparison"] = None
        st.session_state["_dispatch_error"] = None
        if not quiet:
            st.success(f"Automatic dispatch complete · {algorithm} · {sum(bool(r.started) for r in plan['results'])} vehicle(s) started.")
        return True

    # Cold starts recover the persisted schedule. A changed queue or configuration
    # receives one automatic attempt; the persisted dispatch key makes reruns no-ops.
    _dispatch_event_pending = bool(st.session_state.pop("_dispatch_on_next_rerun", False))
    if (_facility_just_loaded or _dispatch_event_pending) and dispatchable_requests(st.session_state["live_queue"]):
        attempt_automatic_dispatch(quiet=True)

    st.markdown("---")
    # System Status Indicator
    st.markdown(f"""
    <div style="font-size: 0.76rem; color: #A9B8BE; margin-bottom: 8px;">
        <span class="badge-pill badge-info" style="margin-right: 6px;">SIMULATION MODE</span> 
        <span style="color: #82939B;">Facility:</span> <b style="color: #E8F0F2;">{active_facility['name']}</b>
    </div>
        <div style="font-size: 0.74rem; color: #82939B;">
        Dispensers: <b style="color: #A9B8BE;">{len(facility_stations)} endpoints</b> &nbsp;|&nbsp; 
        Grid: <b style="color: #22C7A5;">{facility_grid_cap:.0f} kW</b>
    </div>
    """, unsafe_allow_html=True)

    _clock_test_mode = os.getenv("VFLOW_CLOCK_TEST_MODE") == "1"
    @st.fragment(run_every=None if _clock_test_mode else 1.0, key="voltflow_simulation_clock")
    def render_simulation_clock():
        """Run the persisted simulation clock without blocking Streamlit's script thread."""
        clock_results = st.session_state.get("live_schedule_results") or []
        clock_minute = st.session_state.get("operational_snapshot_minute")
        clock_end_minute = (max(float(result.deadline_min) for result in clock_results)
                            if clock_results else None)
        clock_finished = (clock_end_minute is not None and clock_minute is not None
                          and float(clock_minute) >= clock_end_minute - 1e-6)
        speed_labels = ("1x", "2x", "5x", "10x")

        with st.container(border=True):
            st.markdown("**Simulation clock**")
            speed_label = st.selectbox(
                "Simulation speed", speed_labels, key="simulation_speed",
                help="At 1x, one real second advances one simulated minute.",
            )
            play_col, pause_col = st.columns(2)
            play_clicked = play_col.button(
                "Play", key="simulation_play_button",
                disabled=not clock_results or clock_finished or bool(st.session_state.get("simulation_playing")),
                use_container_width=True,
            )
            pause_clicked = pause_col.button(
                "Pause", key="simulation_pause_button",
                disabled=not bool(st.session_state.get("simulation_playing")),
                use_container_width=True,
            )
            reset_clicked = st.button(
                "Reset", key="simulation_reset_button", disabled=not clock_results,
                use_container_width=True,
            )

            if reset_clicked and clock_results:
                start_minute = min(float(result.arrival_min) for result in clock_results)
                reset_queue = copy.deepcopy(st.session_state["live_queue"])
                try:
                    sync_queue_to_simulation(
                        reset_queue, clock_results, start_minute, facility_stations,
                        st.session_state.get("live_epoch_anchor"),
                    )
                    if not save_simulation_snapshot(active_facility_id, reset_queue, allow_rewind=True):
                        raise RuntimeError("A newer simulation snapshot was saved by another session.")
                except Exception as exc:
                    st.error(f"Simulation reset could not be saved; current state was kept. {exc}")
                else:
                    st.session_state["live_queue"] = reset_queue
                    st.session_state["operational_snapshot_minute"] = start_minute
                    st.session_state["simulation_playing"] = False
                    st.session_state["_simulation_last_tick_monotonic"] = None
                    st.rerun(scope="app")
            elif pause_clicked:
                st.session_state["simulation_playing"] = False
                st.session_state["_simulation_last_tick_monotonic"] = None
            elif play_clicked and clock_results:
                end_minute = max(float(result.deadline_min) for result in clock_results)
                first_minute = min(float(result.arrival_min) for result in clock_results)
                clock_minute = float(clock_minute if clock_minute is not None else first_minute)
                if clock_minute < end_minute - 1e-6:
                    st.session_state["simulation_playing"] = True
                    st.session_state["_simulation_last_tick_monotonic"] = time.monotonic()

            is_playing = bool(st.session_state.get("simulation_playing"))
            if is_playing and clock_results:
                now = time.monotonic()
                last_tick = st.session_state.get("_simulation_last_tick_monotonic")
                if last_tick is None:
                    st.session_state["_simulation_last_tick_monotonic"] = now
                else:
                    elapsed_seconds = max(0.0, now - float(last_tick))
                    # Ignore ordinary widget/app reruns and let the 1 s fragment
                    # cadence produce clock ticks. This prevents a fast full-app
                    # rerun loop while still accumulating elapsed wall time.
                    if elapsed_seconds >= 0.9:
                        speed = float(speed_label[:-1])
                        first_minute = min(float(result.arrival_min) for result in clock_results)
                        end_minute = max(float(result.deadline_min) for result in clock_results)
                        current_minute = float(clock_minute if clock_minute is not None else first_minute)
                        next_minute = _advance_simulation_clock(
                            current_minute, elapsed_seconds, speed, end_minute,
                        )
                    else:
                        next_minute = clock_minute
                        current_minute = clock_minute
                        end_minute = max(float(result.deadline_min) for result in clock_results)
                    if (next_minute is not None
                            and current_minute is not None
                            and next_minute > current_minute + 1e-6):
                        advanced_queue = copy.deepcopy(st.session_state["live_queue"])
                        try:
                            sync_queue_to_simulation(
                                advanced_queue, clock_results, next_minute, facility_stations,
                                st.session_state.get("live_epoch_anchor"),
                            )
                            if not save_simulation_snapshot(active_facility_id, advanced_queue):
                                raise RuntimeError("A newer simulation snapshot was saved by another session.")
                        except Exception as exc:
                            st.session_state["simulation_playing"] = False
                            st.session_state["_simulation_last_tick_monotonic"] = None
                            st.error(f"Clock paused because progress could not be saved. Current state was kept. {exc}")
                        else:
                            st.session_state["live_queue"] = advanced_queue
                            st.session_state["operational_snapshot_minute"] = next_minute
                            st.session_state["_simulation_last_tick_monotonic"] = now
                            if next_minute >= end_minute - 1e-6:
                                st.session_state["simulation_playing"] = False
                                st.session_state["_simulation_last_tick_monotonic"] = None
                            # The sidebar fragment advances the shared state; refresh
                            # the selected page so queue, power and completion metrics
                            # render from the same persisted snapshot.
                            st.rerun(scope="app")

            clock_minute = st.session_state.get("operational_snapshot_minute")
            if clock_results and clock_minute is not None:
                next_event = _next_simulation_event(clock_results, clock_minute)
                current_label = _format_simulation_time(
                    st.session_state.get("live_epoch_anchor"), clock_minute, with_date=True,
                )
                next_label = (
                    f"{_format_simulation_time(st.session_state.get('live_epoch_anchor'), next_event[0])} · {next_event[1]}"
                    if next_event else "No upcoming events"
                )
                state_label = ("Running" if st.session_state.get("simulation_playing") else
                               "Complete" if clock_finished else "Paused")
            else:
                current_label = "Waiting for a vehicle registration"
                next_label = "No schedule yet"
                state_label = "Ready"
            st.caption(f"{state_label} · Current simulation time: {current_label}")
            st.caption(f"Next scheduled event: {next_label} · {speed_label} (simulated minutes per real second)")

    render_simulation_clock()

# ── PAGE 1: OVERVIEW ──────────────────────────────────────────────────────────
if nav_page == "Overview":
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Enterprise Operations</div>
            <h1 class="page-title">Overview</h1>
            <p class="page-subtitle">Monitor charging demand, station availability and scheduling performance.</p>
        </div>
        <div class="header-badge">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22C7A5;"></span>
            <span>Facility: {active_facility['name']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    active_live_queue = dispatchable_requests(st.session_state["live_queue"])
    sched_res_list = st.session_state.get("live_schedule_results")
    live_anchor = st.session_state.get("live_epoch_anchor")

    snap = {"charging": {}, "occupied": {}, "waiting": [], "demand_kw": 0.0}
    snap_caption = "No schedule has run; availability reflects the facility configuration."
    total_deliv_live = 0.0
    snap_minute = None

    if sched_res_list and live_anchor is not None:
        total_deliv_live = sum(r.energy_delivered_kwh for r in sched_res_list)
        _opts = _simulation_timeline_options(
            sched_res_list, include_minute=st.session_state.get("operational_snapshot_minute"))
        _clock = lambda m: _format_simulation_time(live_anchor, m)
        if len(_opts) > 1:
            _saved_minute = st.session_state.get("operational_snapshot_minute")
            if _saved_minute is None:
                _saved_minute = _opts[0]
            if _saved_minute not in _opts:
                _saved_minute = min(_opts, key=lambda value: abs(value - _saved_minute))
            st.session_state["operational_snapshot_minute"] = _saved_minute
            snap_minute = st.select_slider("Simulation timeline", options=_opts, value=_saved_minute,
                                           format_func=_clock, key="operational_snapshot_minute")
        else:
            snap_minute = _opts[0]
            st.session_state["operational_snapshot_minute"] = snap_minute
        snap = snapshot_at(sched_res_list, snap_minute)
        if sync_queue_to_simulation(st.session_state["live_queue"], sched_res_list, snap_minute,
                                    facility_stations, live_anchor.to_pydatetime()):
            save_requests(active_facility_id, st.session_state["live_queue"])
        active_live_queue = [
            q for q in st.session_state["live_queue"]
            if q.get("status", "Queued") not in ("Completed", "Cancelled", "Failed")
        ]
        snap_caption = f"Schedule replay at {_clock(snap_minute)} · simulation clock, independent of computer time."

    simulation_clock = _format_simulation_time(live_anchor, snap_minute, with_date=True)
    computer_clock = datetime.datetime.now().astimezone().strftime("%d %b %Y · %H:%M %Z")
    clock_col, note_col = st.columns([1, 2])
    with clock_col:
        st.metric("Current simulation time", simulation_clock if sched_res_list else "Not started")
    with note_col:
        st.caption(f"Simulation time is a replay control. Computer time (local): {computer_clock}.")

    total_chargers = len(facility_stations)
    charger_ids = set(facility_stations["charger_id"].astype(str)) if total_chargers else set()
    available_ids = set(facility_stations.loc[
        facility_stations["charger_status"].astype(str).str.lower().eq("available")
        & facility_stations["station_status"].astype(str).str.lower().eq("available"), "charger_id"
    ].astype(str)) if total_chargers else set()
    occupied_ids = set(snap["occupied"]) & charger_ids
    active_ids = set(snap["charging"]) & charger_ids
    active_count = len(active_ids)
    occupied_count = len(occupied_ids)
    available_count = len(available_ids - occupied_ids)
    waiting_vehicle_count = len(snap["waiting"]) if sched_res_list else len(active_live_queue)
    arrived_vehicle_count = (sum(r.arrival_min <= snap_minute for r in sched_res_list)
                             if sched_res_list and snap_minute is not None else 0)
    upcoming_vehicle_count = (sum(r.arrival_min > snap_minute for r in sched_res_list)
                              if sched_res_list and snap_minute is not None else 0)
    current_kw_live = snap["demand_kw"]

    st.markdown(f"""
    <div class="status-banner">
        <div>
            <strong style="color:#E8F0F2;">Facility:</strong> {active_facility['name']} &nbsp;|&nbsp; 
            <strong style="color:#E8F0F2;">Facility Limit:</strong> {facility_grid_cap:.0f} kW &nbsp;|&nbsp; 
            <strong style="color:#E8F0F2;">Current Load:</strong> {current_kw_live:.1f} kW &nbsp;|&nbsp; 
            <strong style="color:#E8F0F2;">Dispensers:</strong> {total_chargers} endpoints &nbsp;|&nbsp; 
            <strong style="color:#E8F0F2;">Queued Fleet:</strong> {len(active_live_queue)} vehicles
        </div>
        <div style="font-size:0.75rem; color:#82939B;">{snap_caption}</div>
    </div>
    """, unsafe_allow_html=True)

    # Arrival, queue and charger state are the first-read operational signals.
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Vehicles arrived", arrived_vehicle_count)
    k2.metric("Upcoming arrivals", upcoming_vehicle_count)
    k3.metric("Vehicles charging", active_count)
    k4.metric("Vehicles waiting", waiting_vehicle_count)
    k5.metric("Available chargers", available_count)
    k6.metric("Occupied chargers", occupied_count)

    st.markdown("### Operations")
    status_col, activity_col = st.columns([0.9, 1.4])
    with status_col:
        st.caption(f"{active_count} charging · {occupied_count} occupied · {available_count} available · {waiting_vehicle_count} waiting")
        station_rows = []
        for station_id, group in facility_stations.groupby("station_id", sort=False) if not facility_stations.empty else []:
            station_ids_here = set(group["charger_id"].astype(str))
            station_rows.append({
                "Station": station_id,
                "Status": str(group["station_status"].iloc[0]).title(),
                "Ports": len(group),
                "Charging": len(active_ids & station_ids_here),
                "Occupied": len(occupied_ids & station_ids_here),
                "Available": len(available_ids & station_ids_here) - len(occupied_ids & station_ids_here),
                "Power limit": f"{float(group['station_aggregate_power_cap_kw_scenario'].iloc[0]):g} kW",
            })
        if station_rows:
            st.dataframe(pd.DataFrame(station_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No stations configured. Add infrastructure in Charging Stations.")
    with activity_col:
        st.markdown("#### Recent activity")
        if sched_res_list and live_anchor is not None:
            log_rows = []
            for r in sched_res_list[:6]:
                start = (live_anchor + datetime.timedelta(minutes=r.charge_start_min)).strftime("%H:%M") if r.charge_start_min is not None else "Waiting"
                end = (live_anchor + datetime.timedelta(minutes=r.charge_end_min)).strftime("%H:%M") if r.charge_end_min is not None else "—"
                r_status = ("Complete" if r.completed_by_deadline and r.charge_end_min is not None and r.charge_end_min <= snap_minute
                            else "Charging" if r.charger_id_assigned in snap["charging"]
                            else "Occupied" if r.charger_id_assigned in snap["occupied"]
                            else "Waiting" if r in snap["waiting"]
                            else "Scheduled" if r.arrival_min > snap_minute
                    else "Partial / missed" if r.started and r.charge_end_min is not None and r.charge_end_min <= snap_minute
                    else "Failed" if r.deadline_min <= snap_minute
                    else "Not started")
                log_rows.append({"Vehicle": r.vehicle_reg_no, "Charger": r.charger_id_assigned or "Queued",
                    "Session": f"{start}–{end}", "Delivered at snapshot": f"{energy_delivered_at(r, snap_minute):.1f} kWh",
                    "Status": r_status})
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No dispatch has run for this facility yet. Schedule results will appear here after dispatch.")


# ── PAGE 2: CHARGING STATIONS ────────────────────────────────────────────────
elif nav_page == "Charging Stations":
    st.markdown(f"""<div class="page-header"><div>
    <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Infrastructure Management</div>
    <h1 class="page-title">Charging Stations</h1>
    <p class="page-subtitle">Manage the infrastructure at {active_facility['name']}.</p></div>
    <div class="header-badge">{active_facility['data_origin']}</div></div>""", unsafe_allow_html=True)
    st.caption("Illustrative facility configuration. Configured availability is separate from the dispatch snapshot; no hardware telemetry is connected.")
    _station_results = st.session_state.get("live_schedule_results") or []
    if _station_results:
        _minute = st.session_state.get("operational_snapshot_minute")
        if _minute is None:
            _minute = min(r.arrival_min for r in _station_results)
        if not (min(r.arrival_min for r in _station_results) <= _minute <= max(r.deadline_min for r in _station_results)):
            _minute = min(r.arrival_min for r in _station_results)
        _station_snapshot = snapshot_at(_station_results, _minute)
        _station_runtime = facility_stations.copy()
        def _endpoint_state(row):
            cid = str(row["charger_id"])
            if cid in _station_snapshot["charging"]: return "Charging"
            if cid in _station_snapshot["occupied"]: return "Occupied"
            if str(row["charger_status"]).lower() != "available" or str(row["station_status"]).lower() != "available": return "Unavailable"
            return "Available"
        _station_runtime["Simulated state"] = _station_runtime.apply(_endpoint_state, axis=1)
        st.markdown("#### Endpoint state at the selected simulation time")
        st.dataframe(_station_runtime[["station_id", "charger_id", "connector_type", "rated_power_kw", "charger_status", "Simulated state"]],
                     use_container_width=True, hide_index=True)
    station_rows = facility_stations[["station_id", "station_name", "station_location", "station_aggregate_power_cap_kw_scenario", "station_status"]].drop_duplicates("station_id").rename(columns={
        "station_id":"id", "station_name":"name", "station_location":"location",
        "station_aggregate_power_cap_kw_scenario":"power_cap_kw", "station_status":"status"})
    charger_rows = facility_stations[["station_id", "charger_id", "connector_type", "power_type", "rated_power_kw", "charger_status"]].rename(columns={
        "charger_id":"id", "charger_status":"status"})
    with st.form("infrastructure_editor"):
        st.number_input("Facility power limit (kW)", min_value=0.1, value=float(active_facility["grid_capacity_kw"]), step=5.0, key=f"station_editor_grid_cap_{active_facility_id}")
        st.markdown("#### Stations")
        edited_stations = st.data_editor(station_rows, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"infra_stations_{active_facility_id}", column_config={
            "status":st.column_config.SelectboxColumn(options=["available","maintenance","offline"], required=True),
            "power_cap_kw":st.column_config.NumberColumn(min_value=0.1, step=1.0)})
        st.markdown("#### Charger endpoints")
        edited_chargers = st.data_editor(charger_rows, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"infra_chargers_{active_facility_id}", column_config={
            "power_type":st.column_config.SelectboxColumn(options=["AC","DC"], required=True),
            "status":st.column_config.SelectboxColumn(options=["available","maintenance","offline"], required=True),
            "rated_power_kw":st.column_config.NumberColumn(min_value=0.1, step=1.0)})
        save_infrastructure = st.form_submit_button("Save infrastructure", type="primary")
    if save_infrastructure:
        try:
            s_records = [r for r in edited_stations.fillna("").to_dict("records") if str(r.get("id", "")).strip()]
            c_records = [r for r in edited_chargers.fillna("").to_dict("records") if str(r.get("id", "")).strip()]
            facility_config = dict(active_facility)
            facility_config["grid_capacity_kw"] = st.session_state[f"station_editor_grid_cap_{active_facility_id}"]
            save_configuration(facility_config, s_records, c_records)
            st.session_state["simulation_playing"] = False
            st.session_state["_simulation_last_tick_monotonic"] = None
            st.session_state["live_schedule_results"] = None
            st.session_state["live_epoch_anchor"] = None
            st.session_state["live_dispatch_key"] = None
            st.session_state["_dispatch_on_next_rerun"] = True
            st.success("Infrastructure saved.")
            st.rerun()
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))


# ── PAGE 3: SCHEDULER (HERO PAGE!) ───────────────────────────────────────────
elif nav_page == "Charging Operations":
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Dispatch Engine</div>
            <h1 class="page-title">Charging Operations</h1>
            <p class="page-subtitle">Dispatch compatible vehicles, then replay charging progress and completion.</p>
        </div>
        <div class="header-badge">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22C7A5;"></span>
            <span>Simulation Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    dispatch_queue = dispatchable_requests(st.session_state["live_queue"])

    col_sch_cfg, col_sch_sum = st.columns([6, 6])

    with col_sch_cfg:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>Dispatch Configuration</span>
                <span class="badge-pill badge-info">Parameters</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("**Hybrid Priority Scheduler — Recommended**")
        st.caption("Automatic dispatch runs after registration, queue changes and on recovery. This button is an optional re-run / optimize action. The scheduler admits arrivals at their configured times and releases endpoints for the next eligible vehicle.")

        with st.expander("Advanced Configuration", expanded=False):
            st.caption("Priority weights must sum to 1. Equal thirds are the experimental baseline.")
            weight_cols = st.columns(3)
            w_fcfs = weight_cols[0].number_input("FCFS weight", 0.0, 1.0, 1 / 3, 0.01, key="hybrid_w_fcfs")
            w_edf = weight_cols[1].number_input("EDF weight", 0.0, 1.0, 1 / 3, 0.01, key="hybrid_w_edf")
            w_fairness = weight_cols[2].number_input("Fairness weight", 0.0, 1.0, 1 / 3, 0.01, key="hybrid_w_fairness")
            hybrid_weights = {"FCFS": w_fcfs, "EDF": w_edf, "FAIRNESS": w_fairness}
            st.caption(f"Weight total: {sum(hybrid_weights.values()):.2f}")
            st_power_cap = st.slider(
                "Dispatch power ceiling (kW)", min_value=0.1,
                max_value=float(max(facility_grid_cap, 0.1)), value=float(facility_grid_cap), step=0.5,
                help="Cannot exceed the configured facility grid limit. Station caps remain in force.")
            step_min_val = st.select_slider(
                "Simulation step (minutes)", options=[0.5, 1.0, 2.0, 5.0],
                value=float(active_facility.get("simulation_step_min", 1.0)),
                help="Smaller steps provide finer power reallocation.")
        # The advanced widgets exist even while collapsed, so these values are always available.
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("Run Scheduler", type="primary", use_container_width=True)
        compare_btn = st.button("Compare policies", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if dispatch_queue:
            preview_df = live_queue_to_sessions_df(dispatch_queue, facility_grid_cap)
            if not preview_df.empty:
                with st.expander("Hybrid priority list and score details", expanded=False):
                    priority_df = build_hybrid_priority_list(preview_df, facility_stations,
                                                             weights=hybrid_weights)
                    if not priority_df.empty:
                        st.dataframe(priority_df, use_container_width=True, hide_index=True)
                    st.caption("Preview scores are recalculated from the current queue. During simulation, wait age, remaining deadline and energy progress are recalculated at every step. Requests without a compatible available endpoint remain queued.")

        # Dispatch and comparison both call the production simulation engine.
        if st.session_state.get("_dispatch_error"):
            st.warning(f"Automatic dispatch needs attention: {st.session_state['_dispatch_error']}")
        elif st.session_state.get("live_dispatch_key"):
            st.caption("Automatic dispatch status: schedule is current for the active queue and configured infrastructure.")

        if run_btn:
            if not dispatch_queue:
                st.warning("Queue is empty. Register an EV or load the demo scenario from Live Queue.")
            else:
                with st.spinner(f"Re-running {active_facility.get('scheduling_algorithm', 'HYBRID')} for {len(dispatch_queue)} active vehicles..."):
                    attempt_automatic_dispatch(force=True, power_cap_override=st_power_cap,
                                              step_override=step_min_val)

        if compare_btn:
            if not dispatch_queue:
                st.warning("Queue is empty. Register an EV or load the demo scenario from Live Queue.")
            else:
                live_df = live_queue_to_sessions_df(dispatch_queue, facility_grid_cap)
                station_caps = {str(sid): float(group["station_aggregate_power_cap_kw_scenario"].iloc[0])
                                for sid, group in facility_stations.groupby("station_id")}
                comparisons = run_all_algorithms(live_df, facility_stations, station_caps, step_min_val,
                                                 facility_power_cap_kw=st_power_cap)
                comparison_frame = results_to_dataframe(comparisons)
                st.session_state["live_policy_comparison"] = compute_summary(comparison_frame, comparisons, step_min_val, facility_stations)
                st.success("Policy comparison complete for the current queue.")

    with col_sch_sum:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>Fleet Pre-Flight Status</span>
                <span class="badge-pill badge-neutral">Queue Backlog</span>
            </div>
        """, unsafe_allow_html=True)

        tot_q_energy = sum(q.get("energy_required_kwh", 0.0) for q in dispatch_queue)

        st.write(f"**{len(dispatch_queue)} vehicles** · {tot_q_energy:.1f} kWh requested")
        st.caption(f"Dispatch ceiling {st_power_cap:.1f} kW · station limits remain active")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Post-Dispatch Results Section ─────────────────────────────────────────
    sched_res = st.session_state.get("live_schedule_results")
    live_anchor = st.session_state.get("live_epoch_anchor")
    active_algo = st.session_state.get("live_algo", "HYBRID")
    active_algo_display = {"HYBRID": "Hybrid Priority Scheduler", "FCFS": "FCFS Baseline",
                           "EDF": "EDF Baseline", "LAFS": "LAFS Baseline"}.get(active_algo, active_algo)
    active_cap = st.session_state.get("live_station_cap", st_power_cap)

    if sched_res and live_anchor is not None:
        sim_options = _simulation_timeline_options(
            sched_res, include_minute=st.session_state.get("operational_snapshot_minute"))
        saved_minute = st.session_state.get("operational_snapshot_minute")
        if saved_minute not in sim_options:
            saved_minute = min(sim_options, key=lambda value: abs(value - (
                saved_minute if saved_minute is not None else sim_options[0])))
        st.session_state["operational_snapshot_minute"] = saved_minute
        if len(sim_options) > 1:
            sim_minute = st.select_slider(
                "Simulation timeline",
                options=sim_options, value=saved_minute,
                format_func=lambda minute: _format_simulation_time(live_anchor, minute),
                key="operational_snapshot_minute",
            )
        else:
            sim_minute = sim_options[0]
            st.session_state["operational_snapshot_minute"] = sim_minute
        if sync_queue_to_simulation(st.session_state["live_queue"], sched_res, sim_minute,
                                    facility_stations, live_anchor.to_pydatetime()):
            save_requests(active_facility_id, st.session_state["live_queue"])
        live_snapshot = snapshot_at(sched_res, sim_minute)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="status-banner">
            <div>
                <strong style="color:#22C7A5;">Simulation Output:</strong> {active_algo_display} &nbsp;|&nbsp; 
                <strong style="color:#E8F0F2;">Vehicles Scheduled:</strong> {len(sched_res)} &nbsp;|&nbsp; 
                <strong style="color:#E8F0F2;">Power Cap:</strong> {active_cap:.0f} kW
            </div>
            <span class="badge-pill badge-success">DISPATCH READY</span>
        </div>
        """, unsafe_allow_html=True)

        results_live_df = results_to_dataframe({active_algo: sched_res})
        total_requested = sum(r.requested_energy_kwh for r in sched_res)
        total_delivered = sum(energy_delivered_at(r, sim_minute) for r in sched_res)
        fulfillment_pct = (total_delivered / max(total_requested, 1e-6)) * 100.0
        served_waits = [r.waiting_time_min for r in sched_res if r.waiting_time_min is not None]
        avg_wait = np.mean(served_waits) if served_waits else None

        rk1, rk2, rk3, rk4, rk5 = st.columns(5)
        rk1.metric("Fulfillment Rate", f"{fulfillment_pct:.1f}%")
        rk2.metric("Delivered Energy", f"{total_delivered:.1f} kWh")
        rk3.metric("Unserved Energy", f"{(total_requested - total_delivered):.1f} kWh")
        rk4.metric("Avg Wait Time", f"{avg_wait:.1f} min" if avg_wait is not None else "—")
        rk5.metric("Vehicles scheduled", f"{len(sched_res)} EVs")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabbed Visualizations
        t_gantt, t_power, t_util = st.tabs([
            "📅 Allocation Timeline (Gantt)",
            "⚡ Station Power Profile",
            "📊 Dispenser Utilization",
        ])

        with t_gantt:
            st.plotly_chart(
                fig_live_gantt(results_live_df, active_algo, live_anchor),
                use_container_width=True,
                key=f"sched_live_gantt_{active_algo}",
            )

        with t_power:
            st.plotly_chart(
                fig_live_power_profile({active_algo: sched_res}, active_algo, live_anchor, active_cap),
                use_container_width=True,
                key=f"sched_live_power_{active_algo}",
            )

        with t_util:
            util_df = compute_charger_utilization({active_algo: sched_res}, facility_stations)
            if not util_df.empty:
                st.plotly_chart(fig_charger_utilization(util_df), use_container_width=True, key="sched_live_util")
                st.dataframe(util_df[["Charger ID", "Connector", "Rated Power (kW)", "Busy (min)", "Utilization (%)"]], use_container_width=True, hide_index=True)

        # Detailed Allocation Table
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-header">
                <span>Detailed Dispenser Allocation Register</span>
                <span class="badge-pill badge-neutral">Session Records</span>
            </div>
        """, unsafe_allow_html=True)

        sched_table_rows = []
        for r in sched_res:
            start_ist = (live_anchor + datetime.timedelta(minutes=r.charge_start_min)).strftime("%H:%M") if r.charge_start_min is not None else "Pending"
            end_ist = (live_anchor + datetime.timedelta(minutes=r.charge_end_min)).strftime("%H:%M") if r.charge_end_min is not None else "Pending"
            result_snapshot_status = ("Completed" if r.completed_by_deadline and r.charge_end_min is not None and r.charge_end_min <= sim_minute
                                      else "Charging" if any(active.session_id == r.session_id for active, _ in live_snapshot["charging"].values())
                                      else "Waiting" if r.arrival_min > sim_minute
                                      else "Assigned" if r.charge_start_min is not None and r.charge_start_min > sim_minute
                                      else "Waiting" if r in live_snapshot["waiting"]
                                      else "Failed" if r.charge_end_min is not None and r.charge_end_min <= sim_minute
                                      else "Waiting")
            compliance = "Met target" if result_snapshot_status == "Completed" else result_snapshot_status

            sched_table_rows.append({
                "Registration No": r.vehicle_reg_no,
                "Model Variant": r.model_variant,
                "Assigned Charger": r.charger_id_assigned or "Unassigned",
                "Status": result_snapshot_status,
                "Start Time": start_ist,
                "End Time": end_ist,
                "Peak Allocated Power (kW)": round(r.allocated_power_peak_kw, 2),
                "Current SOC (%)": round(r.soc_arrival_pct + 100 * energy_delivered_at(r, sim_minute) / max(r.battery_kwh_verified, 1e-9), 1) if r.battery_kwh_verified > 0 else None,
                "Delivered now (kWh)": round(energy_delivered_at(r, sim_minute), 2),
                "Vehicle Max (kW)": r.sim_power_kw,
                "Charging Duration (min)": round(r.charging_duration_min, 1),
                "SOC at departure (%)": round(r.soc_end_pct, 1) if r.soc_end_pct is not None else "Not available",
                "Requested (kWh)": r.requested_energy_kwh,
                    "Delivered (kWh)": r.energy_delivered_kwh,
                "Fulfillment (%)": f"{r.energy_completion_pct:.1f}%",
                "Wait Time (min)": f"{r.waiting_time_min:.1f}" if r.waiting_time_min is not None else "Not served",
                "Compliance": compliance,
            })

        st.dataframe(pd.DataFrame(sched_table_rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    _policy_comparison = st.session_state.get("live_policy_comparison")
    if _policy_comparison is not None:
        st.markdown("### Policy comparison · current queue")
        st.dataframe(humanize_dataframe(_policy_comparison), use_container_width=True, hide_index=True)


# ── PAGE 3: VEHICLES ─────────────────────────────────────────────────────────
elif nav_page in ("Register EV", "Live Queue"):
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Fleet Requests</div>
            <h1 class="page-title">{'Register EV' if nav_page == 'Register EV' else 'Live Queue'}</h1>
            <p class="page-subtitle">{'Check in a vehicle at ' + active_facility['name'] + '.' if nav_page == 'Register EV' else 'Review request status and simulated charging progress.'}</p>
        </div>
        <div class="header-badge">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22C7A5;"></span>
            <span>{'Vehicle check-in' if nav_page == 'Register EV' else 'Active requests'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if nav_page == "Register EV":
        # Check-In Card
        with st.expander("Register a vehicle", expanded=True):
            st.caption("Simulation request · catalog specifications are editable before check-in.")
            oem_list = get_oem_list()
            default_oem = "Tata Motors" if "Tata Motors" in oem_list else oem_list[0]
    
            col_f1, col_f2 = st.columns(2)
    
            with col_f1:
                in_reg = st.text_input("Vehicle Registration Plate", placeholder="e.g. DL-01-EV-2026")
                in_oem = st.selectbox("Manufacturer (OEM)", oem_list, index=oem_list.index(default_oem) if default_oem in oem_list else 0)
                models_for_oem = get_models_for_oem(in_oem)
                if not models_for_oem:
                    models_for_oem = ["Standard EV Variant"]
                in_model = st.selectbox("Model & Variant", models_for_oem, index=0)
    
            # Specs from verified catalogue
            specs = lookup_vehicle_specs(in_model)
            cat_battery = specs["battery_capacity_kwh"]
            cat_ac = specs["max_ac_charge_power_kw"]
            cat_dc = specs["max_dc_charge_power_kw"]
            cat_connectors = specs["supported_connectors"]
    
            with col_f2:
                in_soc_curr = st.slider("Current Battery SOC (%)", min_value=0, max_value=95, value=25, step=1)
                in_soc_target = st.slider("Target Battery SOC (%)", min_value=in_soc_curr + 5, max_value=100, value=85, step=1)
                in_battery = st.number_input("Battery capacity (kWh)", min_value=1.0, max_value=200.0, value=float(cat_battery), step=0.5)

            arrival_col, deadline_col = st.columns(2)
            now_dt = datetime.datetime.now().replace(second=0, microsecond=0)
            default_departure = now_dt + datetime.timedelta(minutes=60)
            with arrival_col:
                in_arrival_time = st.time_input("Arrival time", value=now_dt.time())
            with deadline_col:
                in_departure_time = st.time_input("Departure deadline", value=default_departure.time())
            in_next_day = st.checkbox("Departure is on next day", value=default_departure.date() > now_dt.date())
    
            with st.expander("Advanced options", expanded=False):
                adv1, adv2 = st.columns(2)
                with adv1:
                    in_connector = st.selectbox("Vehicle connector", cat_connectors, index=0)
                with adv2:
                    default_max_kw = cat_dc if ("DC" in in_connector and cat_dc > 0) else cat_ac
                    if default_max_kw <= 0:
                        default_max_kw = 7.2
                    in_max_power = st.number_input("Maximum vehicle charge power (kW; editable)", min_value=1.0, max_value=350.0, value=float(default_max_kw), step=1.0)
    
            # The first station is preselected; its compatible ports are checked before registration.
            # Specs from verified catalogue, with overrides inside Advanced options.
    
            # Real-time requirement calculations
            energy_needed_kwh = round(in_battery * (in_soc_target - in_soc_curr) / 100.0, 3)
            compatible_mask = st_info["connector_type"].apply(lambda c: is_connector_compatible(in_connector, c)) if not st_info.empty else pd.Series(dtype=bool)
            if not st_info.empty:
                compatible_mask &= st_info["charger_status"].astype(str).str.lower().eq("available")
                compatible_mask &= st_info["station_status"].astype(str).str.lower().eq("available")
            compat_cps = st_info[compatible_mask]
            has_compatible = not compat_cps.empty
            _powers = [min(vehicle_power_limit(in_max_power, cat_ac, cat_dc, c), float(r))
                       for c, r in zip(compat_cps["connector_type"], compat_cps["rated_power_kw"])]
            eff_power_kw = max(_powers) if _powers else 0.0
            charger_power_ref = float(compat_cps["rated_power_kw"].max()) if has_compatible else 0.0
            charging_duration_min = round((energy_needed_kwh / max(eff_power_kw * 0.90, 0.1)) * 60.0, 1) if has_compatible else 0.0
    
            t_arr_min = in_arrival_time.hour * 60 + in_arrival_time.minute
            t_dep_min = in_departure_time.hour * 60 + in_departure_time.minute
            window_min = t_dep_min - t_arr_min + (1440 if in_next_day else 0)
            is_feasible = has_compatible and charging_duration_min <= window_min
            buffer_min = window_min - charging_duration_min
    
            st.markdown("<br>", unsafe_allow_html=True)
            feasibility = (f"Compatible · estimated {charging_duration_min:.0f} min charge · {buffer_min:.0f} min window buffer"
                           if is_feasible else (f"No compatible {in_connector} charger at {selected_st_id}"
                                                if not has_compatible else f"Needs {charging_duration_min:.0f} min, window is {window_min} min"))
            st.caption(f"Pre-check: {energy_needed_kwh:.1f} kWh · {eff_power_kw:.1f} kW effective power · {feasibility}")
    
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register EV and add to queue", type="primary", use_container_width=True):
                if not in_reg.strip():
                    st.error("Please provide a valid registration plate.")
                elif any(str(q.get("vehicle_reg_no", "")).strip().upper() == in_reg.strip().upper()
                         and str(q.get("status", "")).strip().lower() not in {
                             "completed", "cancelled", "failed", "duplicate review"
                         }
                         for q in st.session_state["live_queue"]):
                    st.error("Vehicle registration already active in queue.")
                elif in_soc_target <= in_soc_curr:
                    st.error("Target SOC must exceed current SOC.")
                elif window_min <= 0:
                    st.error("Departure deadline must follow arrival time.")
                else:
                    today_str = datetime.date.today().strftime("%Y-%m-%d")
                    arr_dt_str = f"{today_str} {in_arrival_time.strftime('%H:%M:%S')}"
                    dep_date_str = (datetime.date.today() + datetime.timedelta(days=1 if in_next_day else 0)).strftime("%Y-%m-%d")
                    dep_dt_str = f"{dep_date_str} {in_departure_time.strftime('%H:%M:%S')}"
                    request_id, session_id = new_live_request_ids()
    
                    new_req = {
                        "request_id": request_id,
                        "session_id": session_id,
                        "vehicle_reg_no": in_reg.strip().upper(),
                        "manufacturer": in_oem,
                        "model_name": specs["model_name"],
                        "model_variant": in_model,
                        "vehicle_model_id": specs["model_id"],
                        "battery_capacity_kwh": in_battery,
                        "soc_arrival_pct": in_soc_curr,
                        "soc_target_pct": in_soc_target,
                        "energy_required_kwh": energy_needed_kwh,
                        "max_charge_power_kw": in_max_power,
                        "veh_max_ac_kw": cat_ac,
                        "veh_max_dc_kw": cat_dc,
                        "connector_type": in_connector,
                        "station_id": selected_st_id,
                        "arrival_time_IST": arr_dt_str,
                        "departure_time_IST": dep_dt_str,
                        "status": "Queued",
                        "record_type": "registered",
                        "specification_source": "Catalog defaults; editable values are user-configured",
                        "notes": f"Illustrative simulation request ({in_oem} {in_model})",
                    }
    
                    candidate_queue = [copy.deepcopy(q) for q in st.session_state["live_queue"]] + [new_req]
                    try:
                        # Persist the accepted queue entry first. If dispatch fails, the request
                        # remains visible as waiting and prior sessions stay untouched.
                        save_requests(active_facility_id, candidate_queue)
                    except Exception as exc:
                        st.error(f"Registration was not saved: {exc}")
                    else:
                        st.session_state["live_queue"] = candidate_queue
                        st.session_state["live_policy_comparison"] = None
                        attempt_automatic_dispatch(force=True)
                        st.success(f"Registered {in_reg.upper()} and added it to the live queue.")
                        st.rerun()
    
        st.markdown("<br>", unsafe_allow_html=True)
    
    if nav_page == "Live Queue":
        with st.expander("Demonstration / test data (optional)", expanded=False):
            st.caption("This clearly labelled simulated fleet is only for presentations and testing. Normal vehicle registration works without it.")
            st.caption("1. Load the sample fleet  ·  2. Press Play in the simulation clock  ·  3. Observe automatic allocation and handoff  ·  4. Review Analytics")
            demo_scenario_clicked = st.button(
                "Load demonstration scenario", use_container_width=True, key="q_demo")
        if demo_scenario_clicked:
            demo_requests = get_default_live_fleet(selected_st_id, facility_stations)
            candidate_queue, added = merge_live_demo_requests(st.session_state["live_queue"], demo_requests)
            try:
                save_requests(active_facility_id, candidate_queue)
            except Exception as exc:
                st.error(f"Demonstration scenario was not saved: {exc}")
            else:
                st.session_state["live_queue"] = candidate_queue
                st.session_state["live_policy_comparison"] = None
                if added:
                    attempt_automatic_dispatch(force=True)
                st.success(f"Loaded {added} simulated vehicle(s); {len(demo_requests) - added} already active.")
                st.rerun()

        q_btn2, q_btn3 = st.columns([1, 1])
        with q_btn2:
            if st.button("Clear queue", use_container_width=True, key="q_clear"):
                reset_operational_data(active_facility_id)
                st.session_state["live_queue"] = []
                st.session_state["simulation_playing"] = False
                st.session_state["_simulation_last_tick_monotonic"] = None
                st.session_state["live_schedule_results"] = None
                st.session_state["live_dispatch_key"] = None
                st.session_state["live_epoch_anchor"] = None
                st.session_state["operational_snapshot_minute"] = None
                st.success("Queue cleared.")
                st.rerun()

        with q_btn3:
            if st.session_state["live_queue"]:
                csv_buf = pd.DataFrame(st.session_state["live_queue"]).to_csv(index=False).encode("utf-8")
                st.download_button("Export queue (CSV)", data=csv_buf,
                                   file_name=f"queue_{selected_st_id}.csv", mime="text/csv",
                                   use_container_width=True)

        st.markdown("---")
        st.caption("The simulation clock is separate from your computer’s local time.")
        _queue_results = st.session_state.get("live_schedule_results") or []
        _queue_minute = st.session_state.get("operational_snapshot_minute")
        _queue_snapshot = None
        if _queue_results:
            _queue_options = _simulation_timeline_options(
                _queue_results, include_minute=st.session_state.get("operational_snapshot_minute"))
            _queue_start = _queue_options[0]
            _queue_end = _queue_options[-1]
            if _queue_minute is None or not (_queue_start <= _queue_minute <= _queue_end):
                _queue_minute = _queue_start
            if _queue_minute not in _queue_options:
                _queue_minute = min(_queue_options, key=lambda value: abs(value - _queue_minute))
            st.session_state["operational_snapshot_minute"] = _queue_minute

            event_options = [event for event in _simulation_event_minutes(_queue_results)
                             if event > float(_queue_minute) + 1e-6]
            control_a, control_b = st.columns([1, 2])
            with control_a:
                if st.button("Advance to next event", disabled=not event_options,
                             use_container_width=True, key="queue_next_event"):
                    st.session_state["operational_snapshot_minute"] = event_options[0]
                    _queue_minute = event_options[0]
            with control_b:
                _queue_minute = st.select_slider(
                    "Simulation timeline", options=_queue_options, value=_queue_minute,
                    format_func=lambda minute: _format_simulation_time(
                        st.session_state.get("live_epoch_anchor"), minute),
                    key="operational_snapshot_minute",
                )

            if sync_queue_to_simulation(st.session_state["live_queue"], _queue_results, _queue_minute,
                                        facility_stations, st.session_state.get("live_epoch_anchor")):
                save_requests(active_facility_id, st.session_state["live_queue"])
            _queue_snapshot = snapshot_at(_queue_results, _queue_minute)
            _queue_result_by_id = {r.session_id: r for r in _queue_results}
            _charging_ids = {result.session_id for result, _ in _queue_snapshot["charging"].values()}
            _waiting_ids = {result.session_id for result in _queue_snapshot["waiting"]}
            _arrived_count = sum(r.arrival_min <= _queue_minute for r in _queue_results)
            _upcoming_count = sum(r.arrival_min > _queue_minute for r in _queue_results)
        else:
            _queue_result_by_id = {}
            _charging_ids = set()
            _waiting_ids = set()
            _arrived_count = 0
            _upcoming_count = 0

        _computer_clock = datetime.datetime.now().astimezone().strftime("%d %b %Y · %H:%M %Z")
        _simulation_clock = _format_simulation_time(
            st.session_state.get("live_epoch_anchor"), _queue_minute, with_date=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Simulation time", _simulation_clock if _queue_results else "Not started")
        m2.metric("Vehicles arrived", _arrived_count)
        m3.metric("Upcoming arrivals", _upcoming_count)
        m4.metric("Charging now", len(_charging_ids))
        m5.metric("Waiting", len(_waiting_ids) if _queue_results else len(dispatchable_requests(st.session_state["live_queue"])))
        _available_ids = set(facility_stations.loc[
            facility_stations["charger_status"].astype(str).str.lower().eq("available")
            & facility_stations["station_status"].astype(str).str.lower().eq("available"), "charger_id"
        ].astype(str)) if not facility_stations.empty else set()
        _occupied_ids = set(_queue_snapshot["occupied"]) if _queue_snapshot else set()
        m6.metric("Available chargers", len(_available_ids - _occupied_ids))
        st.caption(f"Computer time (local): {_computer_clock} · simulation time follows the replay control above.")

        if not st.session_state["live_queue"]:
            st.info("No vehicles yet. Register an EV to begin automatic dispatch. Optional sample data is available under Demonstration / test data.")
        else:
            _queue = st.session_state["live_queue"]
            _history_statuses = {"completed", "failed", "cancelled", "duplicate review"}
            _charging_rows, _waiting_rows, _upcoming_rows, _history_rows = [], [], [], []
            _technical_rows = []
            for _request in _queue:
                _status = str(_request.get("status", "Queued")).strip().lower()
                _sid = str(_request.get("session_id", ""))
                _result = _queue_result_by_id.get(_sid)
                _reg = _request.get("vehicle_reg_no") or "—"
                _model = _request.get("model_name") or _request.get("model_variant") or "EV"
                _arrival = _request_arrival_label(
                    _request, _result, st.session_state.get("live_epoch_anchor"))
                _charger = (_result.charger_id_assigned if _result is not None
                            else _request.get("assigned_charger_id"))
                _target_soc = float(_request.get("soc_target_pct") or 0.0)
                _start_soc = float(_request.get("soc_arrival_pct") or 0.0)
                _current_soc = float(_request.get("current_soc_pct", _start_soc) or 0.0)
                _energy_required = float(_request.get("energy_required_kwh") or 0.0)
                _energy_delivered = float(_request.get("energy_delivered_kwh") or 0.0)
                _completion = (
                    _format_simulation_time(st.session_state.get("live_epoch_anchor"), _result.charge_end_min)
                    if _result is not None and _result.charge_end_min is not None else "—")

                _technical_rows.append({
                    "Vehicle": _reg, "Request ID": _request.get("request_id"),
                    "Session ID": _request.get("session_id"), "Station": _request.get("station_id"),
                    "Connector": _request.get("connector_type"),
                })
                if _status in _history_statuses:
                    _display_status = {"failed": "Expired / not completed", "cancelled": "Cancelled",
                                       "duplicate review": "Needs review"}.get(_status, "Completed")
                    _history_rows.append({"Vehicle": _reg, "Model": _model,
                                          "Status": _display_status, "Target SOC": f"{_target_soc:.0f}%",
                                          "Energy delivered": f"{_energy_delivered:.1f} kWh"})
                elif _result is not None and _queue_minute is not None and _result.arrival_min > _queue_minute:
                    _upcoming_rows.append({"Vehicle": _reg, "Arrival": _arrival,
                                          "Target SOC": f"{_target_soc:.0f}%", "Model": _model})
                elif _sid in _charging_ids:
                    _progress = min(100.0, max(0.0, 100.0 * _energy_delivered / max(_energy_required, 1e-9)))
                    _charging_rows.append({
                        "Vehicle": _reg, "Model": _model,
                        "Charger": _compact_charger_label(_charger, _request.get("station_id", "")),
                        "Charging progress": _progress,
                        "SOC": f"{_current_soc:.1f}% → {_target_soc:.0f}%",
                        "Energy delivered": f"{_energy_delivered:.1f} / {_energy_required:.1f} kWh",
                        "Est. completion": _completion,
                    })
                else:
                    _wait_minutes = (max(0.0, float(_queue_minute) - float(_result.arrival_min))
                                     if _result is not None and _queue_minute is not None else 0.0)
                    _wait_reason = (_request.get("waiting_reason") or _request.get("duplicate_reason")
                                    or ("Automatic dispatch is pending." if st.session_state.get("_dispatch_error")
                                        else "Waiting for an eligible charger and available power."))
                    if _result is not None and _result.charge_start_min is not None and _result.charge_start_min > _queue_minute:
                        _expected = _compact_charger_label(_charger, _request.get("station_id", ""))
                        _wait_reason = f"Assigned; starts at {_format_simulation_time(st.session_state.get('live_epoch_anchor'), _result.charge_start_min)}."
                    else:
                        _expected = _compact_charger_label(_charger, _request.get("station_id", ""))
                    _waiting_rows.append({"Vehicle": _reg, "Model": _model, "Arrival": _arrival,
                                         "Expected charger": _expected,
                                         "Waiting duration": f"{_wait_minutes:.0f} min",
                                         "Reason": _wait_reason})

            st.markdown(f"### Charging now · {len(_charging_rows)}")
            if _charging_rows:
                st.dataframe(pd.DataFrame(_charging_rows), use_container_width=True, hide_index=True,
                             column_config={"Charging progress": st.column_config.ProgressColumn(
                                 "Charging progress", min_value=0, max_value=100, format="%d%%")})
            else:
                st.caption("No vehicles are charging at this simulation time.")

            st.markdown(f"### Waiting for charger · {len(_waiting_rows)}")
            if _waiting_rows:
                st.dataframe(pd.DataFrame(_waiting_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No arrived vehicles are waiting for a charger.")

            st.markdown(f"### Upcoming arrivals · {len(_upcoming_rows)}")
            if _upcoming_rows:
                st.dataframe(pd.DataFrame(_upcoming_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No vehicles are scheduled to arrive later in this scenario.")

            st.markdown("### Session history")
            with st.expander(f"Completed, expired and cancelled · {len(_history_rows)}", expanded=False):
                if _history_rows:
                    st.dataframe(pd.DataFrame(_history_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("Completed and expired requests will appear here as the simulation advances.")

            with st.expander("Optional technical details", expanded=False):
                st.caption("Operational identifiers and connector records are available here for audit and troubleshooting.")
                st.dataframe(pd.DataFrame(_technical_rows), use_container_width=True, hide_index=True)

            # Preserve queue editing and removal without putting UUIDs in operator-facing labels.
            with st.expander("Modify or cancel requests", expanded=False):
                reg_options = [f"{q.get('vehicle_reg_no', 'EV')} · {q.get('station_id', '')} · {q.get('status', 'Queued')}"
                               for q in _queue]
                sel_target = st.selectbox("Select vehicle", reg_options, key="queue_modify_target")
                sel_idx = reg_options.index(sel_target)
                curr_target = _queue[sel_idx]

                ec1, ec2 = st.columns(2)
                with ec1:
                    new_tgt_soc = st.slider("Update target SOC (%)",
                                            min_value=min(100, int(curr_target.get("soc_arrival_pct", 20)) + 5),
                                            max_value=100,
                                            value=min(100, max(int(curr_target.get("soc_arrival_pct", 20)) + 5,
                                                               int(curr_target.get("soc_target_pct", 80)))),
                                            key="queue_modify_soc")
                with ec2:
                    status_options = ["Queued", "Priority", "Completed", "Cancelled"]
                    if curr_target.get("status") == "Duplicate review":
                        status_options.append("Duplicate review")
                    current_status = curr_target.get("status", "Queued")
                    new_status = st.selectbox("Request status", status_options,
                                              index=status_options.index(current_status) if current_status in status_options else 0,
                                              key="queue_modify_status")

                mod_c1, mod_c2 = st.columns(2)
                with mod_c1:
                    if st.button("Save changes", use_container_width=True, key="queue_save_changes"):
                        candidate_queue = [copy.deepcopy(q) for q in _queue]
                        candidate = candidate_queue[sel_idx]
                        candidate["soc_target_pct"] = new_tgt_soc
                        candidate["energy_required_kwh"] = round(candidate["battery_capacity_kwh"] *
                                                                  (new_tgt_soc - candidate["soc_arrival_pct"]) / 100.0, 3)
                        candidate["status"] = new_status
                        try:
                            save_live_queue(candidate_queue)
                        except Exception as exc:
                            st.error(f"Request update was not saved: {exc}")
                        else:
                            st.session_state["live_queue"] = candidate_queue
                            attempt_automatic_dispatch(force=True)
                            st.success(f"Updated request for {curr_target.get('vehicle_reg_no', 'vehicle')}.")
                            st.rerun()
                with mod_c2:
                    if st.button("Remove request", use_container_width=True, key="queue_remove_request"):
                        candidate_queue = [copy.deepcopy(q) for q in _queue]
                        candidate_queue.pop(sel_idx)
                        try:
                            save_live_queue(candidate_queue)
                        except Exception as exc:
                            st.error(f"Request could not be removed: {exc}")
                        else:
                            st.session_state["live_queue"] = candidate_queue
                            attempt_automatic_dispatch(force=True)
                            st.warning("Request removed from queue; dispatch was recalculated.")
                            st.rerun()
    
    

# ── PAGE 5: ANALYTICS ────────────────────────────────────────────────────────
elif nav_page == "Analytics":
    st.markdown(f"""<div class="page-header"><div>
    <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Benchmark &amp; Telemetry</div>
    <h1 class="page-title">Analytics</h1>
    <p class="page-subtitle">Operational results and research comparisons, clearly separated.</p></div>
    <div class="header-badge">{active_facility['name']}</div></div>""", unsafe_allow_html=True)

    if analytics_view == "Facility activity":
        saved_results = st.session_state.get("live_schedule_results") or []
        saved_anchor = st.session_state.get("live_epoch_anchor")
        if not saved_results:
            st.info("No operational analytics yet. Generate a schedule to see energy delivery, waiting time, completion and charger use.")
        else:
            active_algo = st.session_state.get("live_algo", "HYBRID")
            step_min = float(active_facility.get("simulation_step_min", 1.0))
            result_map = {active_algo: saved_results}
            results_df = results_to_dataframe(result_map)
            summary_df = compute_summary(results_df, result_map, step_min, facility_stations)
            summary = summary_df.iloc[0]
            metric_cols = st.columns(5)
            metric_cols[0].metric("Sessions", int(summary["Sessions"]))
            metric_cols[1].metric("Energy delivered", f"{summary['Total Delivered (kWh)']:.1f} kWh")
            metric_cols[2].metric("Mean wait", f"{summary['Mean Wait (min)']:.1f} min" if pd.notna(summary["Mean Wait (min)"]) else "—")
            metric_cols[3].metric("Completed", f"{summary['Completion Rate (%)']:.1f}%")
            metric_cols[4].metric("Peak facility demand", f"{summary['Peak Facility Demand (kW)']:.1f} kW")
            st.caption("Values are calculated from the latest saved dispatch simulation for this facility.")

            chart_a, chart_b = st.columns(2)
            with chart_a:
                st.plotly_chart(fig_energy_comparison(results_df), use_container_width=True, key="ops_energy")
            with chart_b:
                st.plotly_chart(fig_waiting_time(results_df), use_container_width=True, key="ops_wait")
            chart_c, chart_d = st.columns(2)
            with chart_c:
                st.plotly_chart(fig_completion_rate(summary_df), use_container_width=True, key="ops_completion")
            with chart_d:
                util_df = compute_charger_utilization(result_map, facility_stations, step_min)
                st.plotly_chart(fig_charger_utilization(util_df), use_container_width=True, key="ops_utilization")
            st.plotly_chart(fig_live_power_profile(result_map, active_algo, saved_anchor, facility_grid_cap),
                            use_container_width=True, key="ops_power")

    elif analytics_view == "Algorithm comparison":
        st.caption("Research benchmark uses the bundled, clearly labeled demonstration sessions and station scenarios. These results are separate from this facility's live queue and dispatch history.")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            dataset_mode = st.selectbox("Scenario", ["Combined (96 Sessions)", "Baseline Benchmark (48 Sessions)", "Extended Scenarios (48 Sessions)"])
            dataset_code = "Combined" if "Combined" in dataset_mode else ("Baseline" if "Baseline" in dataset_mode else "Extended")
        with col_sc2:
            env_options = ["All Environments (Full Dataset)", "Residential", "Workplace", "Public", "Highway Fast Hub", "Urban Fleet Depot"]
            if dataset_code == "Baseline":
                env_options = ["All Environments (Full Dataset)", "Residential", "Workplace", "Public"]
            elif dataset_code == "Extended":
                env_options = ["All Environments (Full Dataset)", "Highway Fast Hub", "Urban Fleet Depot"]
            env = st.selectbox("Environment", env_options, index=0)

        all_sessions = load_sessions(dataset_code)
        sessions_env = get_sessions_for_env(env, all_sessions)
        stations_env = get_stations_for_env(env, reference_stations)
        if sessions_env.empty or stations_env.empty:
            st.info("This benchmark selection has no matching sessions or station configurations.")
        else:
            if stations_env["station_id"].nunique() > 1:
                cap_scale = st.slider("Station limits (% of scenario rating)", 25, 200, 100, 5)
                caps = {str(k): float(v) * cap_scale / 100.0 for k, v in
                        stations_env.groupby("station_id")["station_aggregate_power_cap_kw_scenario"].first().items()}
                cap_label = f"station limits at {cap_scale}%"
            else:
                default_cap = float(stations_env["station_aggregate_power_cap_kw_scenario"].iloc[0])
                scenario_cap = st.slider("Scenario station limit (kW)", 0.1, max(default_cap, 0.1), default_cap, 0.5)
                caps = scenario_cap
                cap_label = f"{scenario_cap:g} kW station limit"
            step_min = 1.0
            with st.spinner("Running FCFS, EDF, LAFS and Hybrid on identical scenario inputs..."):
                sim_results = run_all_algorithms(sessions_env, stations_env, caps, step_min)
                results_df = results_to_dataframe(sim_results)
                summary_df = compute_summary(results_df, sim_results, step_min, stations_env)
            b1, b2, b3 = st.columns(3)
            b1.metric("Sessions", len(sessions_env))
            b2.metric("Required energy", f"{sessions_env['energy_required_kwh'].sum():.1f} kWh")
            b3.metric("Constraint", cap_label)
            st.dataframe(humanize_dataframe(summary_df), use_container_width=True, hide_index=True)
            with st.expander("Policy definitions and limitations"):
                st.markdown("- **FCFS:** gives earlier arrivals priority when a compatible charger is scarce.\n- **EDF:** gives earlier departure deadlines priority.\n- **LAFS:** keeps the waiting order deadline-based and shares constrained power proportionally across active sessions.\n- **Hybrid Priority:** ranks the waiting queue with weighted FCFS wait age, EDF urgency and energy-progress deficit, then uses LAFS-style proportional power sharing as a separate stage. Hybrid is not the same queue-ordering policy as LAFS. Results depend on this scenario and do not imply general superiority.")
            chart_a, chart_b = st.columns(2)
            with chart_a:
                st.plotly_chart(fig_energy_comparison(results_df), use_container_width=True, key="bench_energy")
            with chart_b:
                st.plotly_chart(fig_waiting_time(results_df), use_container_width=True, key="bench_wait")
            chart_c, chart_d = st.columns(2)
            with chart_c:
                st.plotly_chart(fig_completion_rate(summary_df), use_container_width=True, key="bench_completion")
            with chart_d:
                util = compute_charger_utilization(sim_results, stations_env, step_min)
                st.plotly_chart(fig_charger_utilization(util), use_container_width=True, key="bench_utilization")
            st.plotly_chart(fig_station_load(sim_results, power_cap_kw=None, step_min=step_min,
                                             epoch_start=sessions_env["epoch_anchor_IST"].iloc[0]),
                            use_container_width=True, key="bench_power")


# ── ANALYTICS SECONDARY VIEW: REFERENCE DATA ─────────────────────────────────
elif nav_page == "Reference data":
    st.markdown(f"""
    <div class="page-header">
        <div>
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Catalogue &amp; Validation</div>
            <h1 class="page-title">Reference data</h1>
            <p class="page-subtitle">Vehicle specifications, research sessions, data quality and exports.</p>
        </div>
        <div class="header-badge">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22C7A5;"></span>
            <span>Dataset & Fleet Audit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_cat, tab_sess, tab_audit, tab_exp = st.tabs([
        "Vehicle catalogue", "Research sessions", "Data quality", "Exports",
    ])

    with tab_cat:
        st.markdown(f"##### Verified Fleet Models ({len(model_catalog)} models across {model_catalog['manufacturer'].nunique()} OEMs)")
        fc1, fc2 = st.columns(2)
        with fc1:
            cat_filt = st.multiselect("Filter by Vehicle Category", sorted(model_catalog["vehicle_category"].dropna().unique().tolist()))
        with fc2:
            oem_filt = st.multiselect("Filter by Manufacturer", sorted(model_catalog["manufacturer"].dropna().unique().tolist()))

        cat_df = model_catalog.copy()
        if cat_filt:
            cat_df = cat_df[cat_df["vehicle_category"].isin(cat_filt)]
        if oem_filt:
            cat_df = cat_df[cat_df["manufacturer"].isin(oem_filt)]

        st.dataframe(
            humanize_dataframe(cat_df[[
                "model_id", "manufacturer", "model_name", "model_variant",
                "vehicle_category", "battery_capacity_kwh_verified",
                "max_ac_charge_power_kw", "max_dc_charge_power_kw", "charging_connector"
            ]]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Vehicle specification inspector")
        sel_insp_model = st.selectbox("Model", model_catalog["model_variant"].tolist(), index=0)
        v_rec = model_catalog[model_catalog["model_variant"] == sel_insp_model].iloc[0]
        vp1, vp2, vp3 = st.columns(3)
        vp1.metric("Battery", f"{v_rec['battery_capacity_kwh_verified']} kWh")
        vp2.metric("AC limit", f"{v_rec['max_ac_charge_power_kw']} kW")
        vp3.metric("DC limit", f"{v_rec['max_dc_charge_power_kw']} kW" if pd.notna(v_rec['max_dc_charge_power_kw']) and v_rec['max_dc_charge_power_kw'] > 0 else "AC only")
        st.caption(f"{v_rec.get('charging_standard', 'Standard not listed')} · {v_rec.get('charging_connector', 'Connector not listed')} · {v_rec.get('compatibility_notes', 'Manufacturer reference')}")

    with tab_sess:
        all_sessions = load_sessions("Combined")
        st.markdown(f"##### Charging Sessions Dataset ({len(all_sessions)} Total Sessions)")
        st.dataframe(humanize_dataframe(all_sessions), use_container_width=True, hide_index=True)

    with tab_audit:
        all_sessions = load_sessions("Combined")
        quality = session_quality_report(all_sessions, reference_stations)
        st.markdown("##### Dataset Quality Findings")
        if quality.empty:
            st.success("No anomalies detected.")
        else:
            st.dataframe(quality["Check"].value_counts().rename_axis("Validation Rule").reset_index(name="Affected Sessions"), use_container_width=True, hide_index=True)
            with st.expander("Show all validation finding records"):
                st.dataframe(quality, use_container_width=True, hide_index=True)

    with tab_exp:
        st.markdown("##### Export Data and Benchmark Reports")
        ec1, ec2 = st.columns(2)
        with ec1:
            all_sessions = load_sessions("Combined")
            csv_data = all_sessions.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Combined Sessions (CSV)", data=csv_data, file_name="combined_sessions.csv", mime="text/csv", use_container_width=True)

        with ec2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                load_comparison_summary().to_excel(writer, sheet_name="Comparison_Summary", index=False)
                model_catalog.to_excel(writer, sheet_name="EV_Model_Catalog", index=False)
                reference_stations.to_excel(writer, sheet_name="Station_Configs", index=False)
            st.download_button(
                "📥 Export Full Excel Workbook (.XLSX)",
                data=excel_buf.getvalue(),
                file_name="VoltFlow_EV_Charging_Export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


# ── PAGE 6: SETTINGS ─────────────────────────────────────────────────────────
elif nav_page == "Settings":
    st.markdown(f"""<div class="page-header"><div>
    <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #22C7A5; letter-spacing: 0.06em; margin-bottom: 4px;">VoltFlow &bull; Configuration</div>
    <h1 class="page-title">Settings</h1>
    <p class="page-subtitle">Facility identity, operating profile and dispatch defaults.</p></div>
    <div class="header-badge">{active_facility['name']}</div></div>""", unsafe_allow_html=True)
    config_mode = st.radio("Facility setup", ["Active facility", "Create facility"], horizontal=True, label_visibility="collapsed")
    existing_facility = active_facility if config_mode == "Active facility" else None
    current = facility_stations.copy() if existing_facility else pd.DataFrame()
    if existing_facility:
        station_rows = current[["station_id", "station_name", "station_location", "station_aggregate_power_cap_kw_scenario", "station_status"]].drop_duplicates("station_id").rename(columns={
            "station_id":"id", "station_name":"name", "station_location":"location",
            "station_aggregate_power_cap_kw_scenario":"power_cap_kw", "station_status":"status"})
        charger_rows = current[["station_id", "charger_id", "connector_type", "power_type", "rated_power_kw", "charger_status"]].rename(columns={"charger_id":"id", "charger_status":"status"})
    else:
        station_rows = pd.DataFrame([{"id":"SITE-01", "name":"Main station", "location":"", "power_cap_kw":100.0, "status":"available"}])
        charger_rows = pd.DataFrame([{"station_id":"SITE-01", "id":"CP-01", "connector_type":"Type-2 AC", "power_type":"AC", "rated_power_kw":7.2, "status":"available"}])

    st.caption("Simulation mode · demonstration vehicle/session records are illustrative. Facility and endpoint settings are local configuration, not live charger connections.")
    with st.form("facility_settings_form"):
        f1, f2 = st.columns(2)
        with f1:
            facility_id_value = st.text_input("Facility ID", value=existing_facility["id"] if existing_facility else "", disabled=bool(existing_facility), help="A stable local identifier for facility-scoped requests and schedules.")
            facility_name_value = st.text_input("Facility name", value=existing_facility["name"] if existing_facility else "")
            location_value = st.text_input("Location", value=existing_facility["location"] if existing_facility else "")
            timezone_value = st.text_input("Timezone (IANA)", value=existing_facility["timezone"] if existing_facility else "Asia/Kolkata")
            grid_value = st.number_input("Facility power limit (kW)", min_value=0.1, value=float(existing_facility["grid_capacity_kw"] if existing_facility else 100.0), step=5.0)
        with f2:
            operating_start_value = st.text_input("Opening time (HH:MM)", value=existing_facility["operating_start"] if existing_facility else "00:00")
            operating_end_value = st.text_input("Closing time (HH:MM)", value=existing_facility["operating_end"] if existing_facility else "23:59")
            tariff_value = st.number_input("Energy tariff per kWh", min_value=0.0, value=float(existing_facility["tariff_per_kwh"] if existing_facility else 0.0), step=0.5, help="Leave at 0 when the tariff has not been configured.")
            currency_value = st.text_input("Currency", value=existing_facility["currency"] if existing_facility else "INR", max_chars=3)
            st.caption("Not Configured" if tariff_value == 0 else f"Configured: {tariff_value:.2f} {currency_value.upper()}/kWh")
            st.caption("Live dispatch: Hybrid Priority. FCFS, EDF and LAFS remain research baselines.")
            algorithm_value = "HYBRID"
            step_value = st.selectbox("Default simulation step (minutes)", [0.5, 1.0, 2.0, 5.0], index=[0.5, 1.0, 2.0, 5.0].index(float(existing_facility["simulation_step_min"])) if existing_facility and float(existing_facility["simulation_step_min"]) in [0.5,1.0,2.0,5.0] else 1)
        save_facility_btn = st.form_submit_button("Save settings", type="primary")
    if save_facility_btn:
        try:
            new_id = existing_facility["id"] if existing_facility else slugify(facility_id_value.strip() or facility_name_value)
            if not existing_facility and get_facility(new_id):
                raise ValueError(f"Facility ID '{new_id}' already exists. Enter a different ID or switch to the active facility.")
            save_configuration({"id":new_id, "name":facility_name_value, "location":location_value, "timezone":timezone_value,
                "status":"active", "grid_capacity_kw":grid_value, "operating_start":operating_start_value,
                "operating_end":operating_end_value, "tariff_per_kwh":tariff_value, "currency":currency_value.upper(),
                "scheduling_algorithm":algorithm_value, "simulation_step_min":step_value,
                "data_origin":"Manually configured" if not existing_facility else existing_facility["data_origin"]},
                station_rows.fillna("").to_dict("records"), charger_rows.fillna("").to_dict("records"))
            st.session_state["simulation_playing"] = False
            st.session_state["_simulation_last_tick_monotonic"] = None
            st.session_state["live_schedule_results"] = None
            st.session_state["live_epoch_anchor"] = None
            st.session_state["live_dispatch_key"] = None
            st.session_state["_dispatch_on_next_rerun"] = True
            st.session_state["switch_to_facility"] = new_id
            st.success("Facility settings saved.")
            st.rerun()
        except (ValueError, KeyError, TypeError) as exc:
            st.error(str(exc))

    if existing_facility:
        cfg1, cfg2 = st.columns(2)
        with cfg1:
            st.markdown("#### Local records")
            record_counts = facility_data_summary(active_facility_id)
            st.write(f"{record_counts['requests']} vehicle requests · {record_counts['schedules']} saved dispatches")
        with cfg2:
            with st.expander("System diagnostics"):
                st.caption(f"Scheduler constraint checks: {TEST_COUNT}")
                if st.button("Run scheduler checks", key="settings_run_tests"):
                    try:
                        _run_tests()
                        st.success("All scheduler checks passed.")
                    except AssertionError as exc:
                        st.error(f"Check failed: {exc}")
        with st.expander("Audit history"):
            _settings_audit = audit_history(active_facility_id)
            if _settings_audit.empty:
                st.info("No changes recorded yet.")
            else:
                st.dataframe(_settings_audit, use_container_width=True, hide_index=True)
        with st.expander("Reset demonstration records"):
            st.caption("Clears this facility's local vehicle queue and saved dispatch runs. Facility, station and charger configuration is retained.")
            reset_confirm = st.checkbox("I understand queue and dispatch records will be removed", key="reset_ops_confirm")
            if st.button("Reset operational records", disabled=not reset_confirm, key="reset_ops_button"):
                reset_operational_data(active_facility_id)
                st.session_state["live_queue"] = []
                st.session_state["simulation_playing"] = False
                st.session_state["_simulation_last_tick_monotonic"] = None
                st.session_state["live_schedule_results"] = None
                st.session_state["live_epoch_anchor"] = None
                st.session_state["live_dispatch_key"] = None
                st.session_state["live_policy_comparison"] = None
                st.success("Operational records cleared. Facility configuration remains.")
                st.rerun()
        with st.expander("Archive this facility"):
            st.caption("Requests, schedules and audit history are retained when a facility is archived.")
            if st.button("Archive facility", key="archive_facility_button"):
                try:
                    archive_facility(active_facility_id)
                    active_remaining = list_facilities()
                    st.session_state["switch_to_facility"] = active_remaining[0]["id"]
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

# ── VoltFlow Global Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 56px; padding-top: 18px; border-top: 1px solid #202D36; display: flex; justify-content: space-between; align-items: center; font-size: 0.74rem; color: #82939B; flex-wrap: wrap; gap: 8px;">
    <div><strong style="color: #E8F0F2;">VoltFlow</strong> &mdash; Intelligent EV Charging &amp; Scheduling Platform</div>
    <div>Enterprise Facility Management &bull; Smart Grid Optimization &bull; Fleet Infrastructure</div>
</div>
""", unsafe_allow_html=True)
