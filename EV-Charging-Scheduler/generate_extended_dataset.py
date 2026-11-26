"""
generate_extended_dataset.py
============================
Updates the master Excel dataset:
1. Rewrites 'EV_Model_Catalog' with 56 source-verified Indian EV models.
2. Preserves 'EV_Sessions_48' intact with all 44 canonical and alias columns.
3. Adds 'EV_Sessions_Extended' containing realistic, reproducible synthetic scenarios
   covering new models (Ioniq 5, EV6, Atto 3, Curvv, XUV400, Windsor, iX1, etc.)
   across Highway Fast Charging, Corporate Campus, and Fleet Hub scenarios.
4. Adds new Station_Configs for Highway Fast-Charging Hub (IN-HWY-01) and Fleet Depot (IN-FLEET-01).
5. Updates metadata and documentation sheets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from build_comprehensive_catalog import generate_catalog_dataframe

EXCEL_PATH = Path("data/India_EV_Charging_Scheduler_Complete_Dataset.xlsx")

def build_extended_scenarios(catalog_df):
    """
    Build additional reproducible synthetic scenarios:
    - Highway Fast Charging (24 sessions): Heavy demand, fast turnaround, urgent deadlines.
    - Corporate Fleet / Campus (24 sessions): Diverse 4W, 2W, 3W daytime charging.
    Total extended sessions: 48 sessions.
    Combined total with baseline: 96 sessions.
    """
    cat_map = catalog_df.set_index("model_id").to_dict(orient="index")

    extended_rows = []

    # Scenario A: Highway Fast Charging Hub (IN-HWY-01, Cap: 250 kW)
    # Date: 2026-11-25. Short dwell times, high power, urgent departures
    hwy_sessions = [
        # (session_id, model_id, arr_hour, arr_min, window_min, soc_arr, soc_tgt, pref_charger, pwr_limit)
        ("IN-HWY-001", "CAR-HYUNDAI-IONIQ5", 8, 15, 45, 15, 80, "IN-HWY-01-CP01", 150.0),
        ("IN-HWY-002", "CAR-KIA-EV6-77", 8, 30, 40, 20, 85, "IN-HWY-01-CP02", 150.0),
        ("IN-HWY-003", "CAR-TATA-CURVV-55", 8, 45, 55, 18, 85, "IN-HWY-01-CP03", 70.0),
        ("IN-HWY-004", "CAR-BYD-SEAL-PERF", 9, 10, 50, 22, 80, "IN-HWY-01-CP01", 150.0),
        ("IN-HWY-005", "CAR-MAHINDRA-BE6E-59", 9, 25, 45, 25, 85, "IN-HWY-01-CP02", 140.0),
        ("IN-HWY-006", "CAR-MG-WINDSOR-38", 9, 40, 60, 30, 90, "IN-HWY-01-CP04", 45.0),
        ("IN-HWY-007", "CAR-BMW-IX1-66", 10, 5, 45, 15, 80, "IN-HWY-01-CP03", 130.0),
        ("IN-HWY-008", "CAR-BYD-ATTO3-60", 10, 20, 55, 28, 85, "IN-HWY-01-CP04", 88.0),
        ("IN-HWY-009", "CAR-TATA-NEXON-45", 10, 45, 50, 20, 80, "IN-HWY-01-CP03", 60.0),
        ("IN-HWY-010", "CAR-MAHINDRA-XUV400-39", 11, 0, 55, 25, 85, "IN-HWY-01-CP04", 50.0),
        ("IN-HWY-011", "CAR-HYUNDAI-IONIQ5", 11, 20, 40, 18, 80, "IN-HWY-01-CP01", 150.0),
        ("IN-HWY-012", "CAR-VOLVO-EX40-78", 11, 40, 45, 20, 85, "IN-HWY-01-CP02", 150.0),
        ("IN-HWY-013", "CAR-TATA-HARRIER-75", 12, 10, 50, 15, 80, "IN-HWY-01-CP01", 120.0),
        ("IN-HWY-014", "CAR-MERC-EQA-70", 12, 30, 55, 22, 85, "IN-HWY-01-CP03", 100.0),
        ("IN-HWY-015", "CAR-AUDI-Q8-114", 13, 0, 50, 25, 80, "IN-HWY-01-CP02", 150.0),
        ("IN-HWY-016", "CAR-KIA-EV9-100", 13, 20, 45, 20, 80, "IN-HWY-01-CP01", 150.0),
        ("IN-HWY-017", "CAR-TATA-CURVV-45", 13, 50, 55, 19, 85, "IN-HWY-01-CP03", 70.0),
        ("IN-HWY-018", "CAR-BYD-ATTO3-50", 14, 15, 60, 25, 90, "IN-HWY-01-CP04", 70.0),
        ("IN-HWY-019", "CAR-MG-ZSEV-50", 14, 45, 55, 20, 85, "IN-HWY-01-CP04", 50.0),
        ("IN-HWY-020", "CAR-PORSCHE-MACAN-100", 15, 10, 40, 15, 80, "IN-HWY-01-CP01", 150.0),
        ("IN-HWY-021", "CAR-BMW-I4-83", 15, 35, 45, 20, 80, "IN-HWY-01-CP02", 150.0),
        ("IN-HWY-022", "CAR-CITROEN-EC3", 16, 0, 70, 30, 85, "IN-HWY-01-CP04", 30.0),
        ("IN-HWY-023", "CAR-TATA-PUNCH-LR", 16, 30, 60, 18, 85, "IN-HWY-01-CP03", 45.0),
        ("IN-HWY-024", "CAR-MAHINDRA-BE6E-79", 17, 0, 45, 20, 80, "IN-HWY-01-CP01", 150.0),
    ]

    for sid, mid, ah, am, win_min, soc_arr, soc_tgt, pref_cp, pwr_lim in hwy_sessions:
        m_info = cat_map[mid]
        bat_cap = float(m_info["battery_capacity_kwh_verified"])
        req_energy = round(bat_cap * (soc_tgt - soc_arr) / 100.0, 2)
        arr_dt = pd.Timestamp(2026, 11, 25, ah, am, 0)
        dep_dt = arr_dt + pd.Timedelta(minutes=win_min)

        # Connector
        conn = "CCS2 DC"

        extended_rows.append({
            "session_id": sid,
            "environment": "Highway Fast Hub",
            "station_id": "IN-HWY-01",
            "ev_category": m_info["vehicle_category"],
            "vehicle_class_scenario": "Passenger EV",
            "battery_capacity_kwh_scenario": bat_cap,
            "arrival_time_IST": arr_dt,
            "deadline_time_IST": dep_dt,
            "available_window_min": win_min,
            "soc_at_arrival_pct_scenario": soc_arr,
            "target_soc_pct_scenario": soc_tgt,
            "requested_energy_kwh_scenario": req_energy,
            "connector_type_scenario": conn,
            "charger_rated_power_kw_reference": 150.0 if "CP01" in pref_cp or "CP02" in pref_cp else 75.0,
            "vehicle_max_charge_power_kw_scenario": float(m_info["max_dc_charge_power_kw"] or 50.0),
            "effective_session_power_limit_kw": pwr_lim,
            "station_aggregate_power_cap_kw_scenario": 250.0,
            "tariff_inr_per_kwh": 21.5,
            "data_origin": "SIMULATED_EXPERIMENTAL_SESSION",
            "measurement_status": "synthetic scenario input; not observed transaction",
            "parameter_basis": "Reproducible synthetic scenario for high-speed highway charging",
            "vehicle_model_id": mid,
            "manufacturer": m_info["manufacturer"],
            "model_variant": m_info["model_variant"],
            "battery_capacity_kwh_verified": bat_cap,
            "max_ac_charge_power_kw": m_info["max_ac_charge_power_kw"],
            "max_dc_charge_power_kw": m_info["max_dc_charge_power_kw"],
            "manufacturer_charge_time": m_info["manufacturer_charge_time"],
            "charge_time_basis": m_info["charge_time_basis"],
            "dc_fast_charging_supported": m_info["dc_fast_charging_supported"],
            "manufacturer_source_url": m_info["source_url"],
            "battery_capacity_difference_kwh": 0.0,
            "battery_capacity_match_note": "Exact match from verified 2026 catalogue",
            "model_adjusted_effective_power_kw": min(pwr_lim, float(m_info["max_dc_charge_power_kw"] or pwr_lim)),
            "charge_duration_estimation_basis": "Calculated theoretical duration at effective power",
            "estimated_uninterrupted_duration_min_requested": round((req_energy / min(pwr_lim, float(m_info["max_dc_charge_power_kw"] or pwr_lim)) / 0.9) * 60, 1),
            "estimated_duration_note": "Calculated estimate at 90% efficiency without tapering",
            "charger_id": pref_cp,
            "charger_id_assignment_basis": "Scenario assignment: compatible high-power CCS2 DC charge point",
            "connection_time_IST": arr_dt,
            "departure_time_IST": dep_dt,
            "energy_required_kwh": req_energy,
            "charging_power_limit_kw": pwr_lim,
            "station_power_limit_kw": 250.0,
        })

    # Scenario B: Urban Fleet & Commercial Depot (IN-FLEET-01, Cap: 80 kW)
    # Mixed 2W, 3W, 4W fleet vehicles with overnight/shift-based charging
    fleet_sessions = [
        ("IN-FLT-001", "3W-MAHINDRA-TREO", 18, 0, 480, 20, 95, "IN-FLT-01-CP01", 3.3, "Bharat AC-001"),
        ("IN-FLT-002", "3W-MAHINDRA-TREO", 18, 15, 450, 15, 90, "IN-FLT-01-CP02", 3.3, "Bharat AC-001"),
        ("IN-FLT-003", "3W-PIAGGIO-ECITY-ULTRA", 18, 30, 420, 25, 95, "IN-FLT-01-CP01", 3.3, "Bharat AC-001"),
        ("IN-FLT-004", "3W-EULER-HILOAD-13", 18, 45, 120, 20, 85, "IN-FLT-01-CP03", 15.0, "Bharat DC-001"),
        ("IN-FLT-005", "2W-TVS-IQUBE-35", 19, 0, 480, 25, 100, "IN-FLT-01-CP02", 0.65, "Bharat AC-001"),
        ("IN-FLT-006", "2W-ATHER-RIZTA", 19, 15, 450, 20, 95, "IN-FLT-01-CP01", 0.75, "Bharat AC-001"),
        ("IN-FLT-007", "CAR-TATA-TIGOR-LR", 19, 30, 400, 30, 90, "IN-FLT-01-CP04", 7.2, "Type-2 AC"),
        ("IN-FLT-008", "CAR-MG-COMET-STD", 19, 45, 420, 25, 90, "IN-FLT-01-CP04", 3.3, "Type-2 AC"),
        ("IN-FLT-009", "3W-EULER-HILOAD-13", 20, 30, 150, 15, 80, "IN-FLT-01-CP03", 15.0, "Bharat DC-001"),
        ("IN-FLT-010", "CAR-CITROEN-EC3", 20, 45, 360, 20, 90, "IN-FLT-01-CP04", 3.3, "Type-2 AC"),
        ("IN-FLT-011", "2W-OLA-S1PRO-G2", 21, 0, 480, 20, 95, "IN-FLT-01-CP02", 0.75, "Bharat AC-001"),
        ("IN-FLT-012", "CAR-TATA-TIAGO-LR", 21, 15, 420, 25, 90, "IN-FLT-01-CP04", 7.2, "Type-2 AC"),
        ("IN-FLT-013", "3W-MAHINDRA-TREO", 21, 30, 400, 35, 95, "IN-FLT-01-CP01", 3.3, "Bharat AC-001"),
        ("IN-FLT-014", "2W-BAJAJ-CHETAK-32", 21, 45, 420, 30, 100, "IN-FLT-01-CP02", 0.8, "Bharat AC-001"),
        ("IN-FLT-015", "3W-EULER-HILOAD-13", 22, 0, 140, 25, 85, "IN-FLT-01-CP03", 15.0, "Bharat DC-001"),
        ("IN-FLT-016", "CAR-MG-WINDSOR-38", 22, 15, 360, 35, 90, "IN-FLT-01-CP04", 7.4, "Type-2 AC"),
        ("IN-FLT-017", "CAR-TATA-PUNCH-MR", 22, 30, 360, 20, 85, "IN-FLT-01-CP04", 3.3, "Type-2 AC"),
        ("IN-FLT-018", "3W-PIAGGIO-ECITY-ULTRA", 22, 45, 360, 20, 90, "IN-FLT-01-CP01", 3.3, "Bharat AC-001"),
        ("IN-FLT-019", "2W-TVS-IQUBE-51", 23, 0, 420, 15, 90, "IN-FLT-01-CP02", 0.95, "Bharat AC-001"),
        ("IN-FLT-020", "CAR-MAHINDRA-XUV400-34", 23, 15, 340, 30, 85, "IN-FLT-01-CP04", 3.3, "Type-2 AC"),
        ("IN-FLT-021", "3W-EULER-HILOAD-13", 23, 45, 120, 15, 80, "IN-FLT-01-CP03", 15.0, "Bharat DC-001"),
        ("IN-FLT-022", "3W-MAHINDRA-TREO", 0, 0, 360, 25, 90, "IN-FLT-01-CP01", 3.3, "Bharat AC-001"),
        ("IN-FLT-023", "2W-ATHER-450X-37", 0, 15, 360, 20, 95, "IN-FLT-01-CP02", 0.75, "Bharat AC-001"),
        ("IN-FLT-024", "CAR-TATA-NEXON-MR", 0, 30, 330, 25, 85, "IN-FLT-01-CP04", 3.3, "Type-2 AC"),
    ]

    for sid, mid, ah, am, win_min, soc_arr, soc_tgt, pref_cp, pwr_lim, conn in fleet_sessions:
        m_info = cat_map[mid]
        bat_cap = float(m_info["battery_capacity_kwh_verified"])
        req_energy = round(bat_cap * (soc_tgt - soc_arr) / 100.0, 2)
        day = 25 if ah >= 18 else 26
        arr_dt = pd.Timestamp(2026, 11, day, ah, am, 0)
        dep_dt = arr_dt + pd.Timedelta(minutes=win_min)

        extended_rows.append({
            "session_id": sid,
            "environment": "Urban Fleet Depot",
            "station_id": "IN-FLT-01",
            "ev_category": m_info["vehicle_category"],
            "vehicle_class_scenario": "Commercial / Fleet",
            "battery_capacity_kwh_scenario": bat_cap,
            "arrival_time_IST": arr_dt,
            "deadline_time_IST": dep_dt,
            "available_window_min": win_min,
            "soc_at_arrival_pct_scenario": soc_arr,
            "target_soc_pct_scenario": soc_tgt,
            "requested_energy_kwh_scenario": req_energy,
            "connector_type_scenario": conn,
            "charger_rated_power_kw_reference": 15.0 if "CP03" in pref_cp else (7.2 if "CP04" in pref_cp else 3.3),
            "vehicle_max_charge_power_kw_scenario": float(m_info["max_ac_charge_power_kw"] or 3.3) if "AC" in conn else float(m_info["max_dc_charge_power_kw"] or 15.0),
            "effective_session_power_limit_kw": pwr_lim,
            "station_aggregate_power_cap_kw_scenario": 80.0,
            "tariff_inr_per_kwh": 14.0,
            "data_origin": "SIMULATED_EXPERIMENTAL_SESSION",
            "measurement_status": "synthetic scenario input; not observed transaction",
            "parameter_basis": "Shift-based commercial charging depot with 2W, 3W, and light commercial EVs",
            "vehicle_model_id": mid,
            "manufacturer": m_info["manufacturer"],
            "model_variant": m_info["model_variant"],
            "battery_capacity_kwh_verified": bat_cap,
            "max_ac_charge_power_kw": m_info["max_ac_charge_power_kw"],
            "max_dc_charge_power_kw": m_info["max_dc_charge_power_kw"],
            "manufacturer_charge_time": m_info["manufacturer_charge_time"],
            "charge_time_basis": m_info["charge_time_basis"],
            "dc_fast_charging_supported": m_info["dc_fast_charging_supported"],
            "manufacturer_source_url": m_info["source_url"],
            "battery_capacity_difference_kwh": 0.0,
            "battery_capacity_match_note": "Exact match from verified 2026 catalogue",
            "model_adjusted_effective_power_kw": pwr_lim,
            "charge_duration_estimation_basis": "Calculated theoretical duration at effective power",
            "estimated_uninterrupted_duration_min_requested": round((req_energy / pwr_lim / 0.9) * 60, 1),
            "estimated_duration_note": "Calculated estimate at 90% efficiency without tapering",
            "charger_id": pref_cp,
            "charger_id_assignment_basis": f"Scenario assignment: compatible {conn} depot endpoint",
            "connection_time_IST": arr_dt,
            "departure_time_IST": dep_dt,
            "energy_required_kwh": req_energy,
            "charging_power_limit_kw": pwr_lim,
            "station_power_limit_kw": 80.0,
        })

    return pd.DataFrame(extended_rows)

def build_extended_stations():
    stations = [
        # Existing Residential (IN-RES-01)
        {"environment": "Residential", "station_id": "IN-RES-01", "charger_id": "IN-RES-01-CP01", "connector_type": "Bharat AC-001", "rated_power_kw": 3.3, "station_aggregate_power_cap_kw_scenario": 10.0, "power_cap_is_measured": False, "profile_notes": "Home/apartment overnight slow charging point", "source_basis": "BEE/MoP guidelines; 10 kW cap"},
        {"environment": "Residential", "station_id": "IN-RES-01", "charger_id": "IN-RES-01-CP02", "connector_type": "Bharat AC-001", "rated_power_kw": 3.3, "station_aggregate_power_cap_kw_scenario": 10.0, "power_cap_is_measured": False, "profile_notes": "Home/apartment overnight slow charging point", "source_basis": "BEE/MoP guidelines; 10 kW cap"},
        {"environment": "Residential", "station_id": "IN-RES-01", "charger_id": "IN-RES-01-CP03", "connector_type": "Type-2 AC", "rated_power_kw": 7.2, "station_aggregate_power_cap_kw_scenario": 10.0, "power_cap_is_measured": False, "profile_notes": "Home AC fast charger point", "source_basis": "BEE/MoP guidelines; 10 kW cap"},
        # Existing Workplace (IN-WRK-01)
        {"environment": "Workplace", "station_id": "IN-WRK-01", "charger_id": "IN-WRK-01-CP01", "connector_type": "Type-2 AC", "rated_power_kw": 22.0, "station_aggregate_power_cap_kw_scenario": 60.0, "power_cap_is_measured": False, "profile_notes": "Campus/office 22 kW AC point", "source_basis": "BEE/MoP Type-2 AC; 60 kW cap"},
        {"environment": "Workplace", "station_id": "IN-WRK-01", "charger_id": "IN-WRK-01-CP02", "connector_type": "Type-2 AC", "rated_power_kw": 22.0, "station_aggregate_power_cap_kw_scenario": 60.0, "power_cap_is_measured": False, "profile_notes": "Campus/office 22 kW AC point", "source_basis": "BEE/MoP Type-2 AC; 60 kW cap"},
        {"environment": "Workplace", "station_id": "IN-WRK-01", "charger_id": "IN-WRK-01-CP03", "connector_type": "Type-2 AC", "rated_power_kw": 22.0, "station_aggregate_power_cap_kw_scenario": 60.0, "power_cap_is_measured": False, "profile_notes": "Campus/office 22 kW AC point", "source_basis": "BEE/MoP Type-2 AC; 60 kW cap"},
        {"environment": "Workplace", "station_id": "IN-WRK-01", "charger_id": "IN-WRK-01-CP04", "connector_type": "Type-2 AC", "rated_power_kw": 22.0, "station_aggregate_power_cap_kw_scenario": 60.0, "power_cap_is_measured": False, "profile_notes": "Campus/office 22 kW AC point", "source_basis": "BEE/MoP Type-2 AC; 60 kW cap"},
        # Existing Public (IN-PUB-01)
        {"environment": "Public", "station_id": "IN-PUB-01", "charger_id": "IN-PUB-01-CP01", "connector_type": "CCS2 DC", "rated_power_kw": 50.0, "station_aggregate_power_cap_kw_scenario": 100.0, "power_cap_is_measured": False, "profile_notes": "Public fast charger 50 kW CCS2", "source_basis": "BEE/MoP CCS2 standard; 100 kW cap"},
        {"environment": "Public", "station_id": "IN-PUB-01", "charger_id": "IN-PUB-01-CP02", "connector_type": "CCS2 DC", "rated_power_kw": 50.0, "station_aggregate_power_cap_kw_scenario": 100.0, "power_cap_is_measured": False, "profile_notes": "Public fast charger 50 kW CCS2", "source_basis": "BEE/MoP CCS2 standard; 100 kW cap"},
        {"environment": "Public", "station_id": "IN-PUB-01", "charger_id": "IN-PUB-01-CP03", "connector_type": "Bharat DC-001", "rated_power_kw": 15.0, "station_aggregate_power_cap_kw_scenario": 100.0, "power_cap_is_measured": False, "profile_notes": "Public DC charger 15 kW Bharat DC-001", "source_basis": "BEE/MoP Bharat DC-001 standard"},
        {"environment": "Public", "station_id": "IN-PUB-01", "charger_id": "IN-PUB-01-CP04", "connector_type": "Type-2 AC", "rated_power_kw": 22.0, "station_aggregate_power_cap_kw_scenario": 100.0, "power_cap_is_measured": False, "profile_notes": "Public AC charger 22 kW Type-2", "source_basis": "BEE/MoP Type-2 standard"},

        # NEW: Highway Fast Hub (IN-HWY-01)
        {"environment": "Highway Fast Hub", "station_id": "IN-HWY-01", "charger_id": "IN-HWY-01-CP01", "connector_type": "CCS2 DC", "rated_power_kw": 150.0, "station_aggregate_power_cap_kw_scenario": 250.0, "power_cap_is_measured": False, "profile_notes": "High-speed highway corridor DC fast charger (150 kW)", "source_basis": "MoP 2024 expressway charging guidelines; 250 kW transformer headroom"},
        {"environment": "Highway Fast Hub", "station_id": "IN-HWY-01", "charger_id": "IN-HWY-01-CP02", "connector_type": "CCS2 DC", "rated_power_kw": 150.0, "station_aggregate_power_cap_kw_scenario": 250.0, "power_cap_is_measured": False, "profile_notes": "High-speed highway corridor DC fast charger (150 kW)", "source_basis": "MoP 2024 expressway charging guidelines; 250 kW transformer headroom"},
        {"environment": "Highway Fast Hub", "station_id": "IN-HWY-01", "charger_id": "IN-HWY-01-CP03", "connector_type": "CCS2 DC", "rated_power_kw": 75.0, "station_aggregate_power_cap_kw_scenario": 250.0, "power_cap_is_measured": False, "profile_notes": "Medium-speed corridor DC fast charger (75 kW)", "source_basis": "MoP 2024 expressway charging guidelines; 250 kW transformer headroom"},
        {"environment": "Highway Fast Hub", "station_id": "IN-HWY-01", "charger_id": "IN-HWY-01-CP04", "connector_type": "CCS2 DC", "rated_power_kw": 75.0, "station_aggregate_power_cap_kw_scenario": 250.0, "power_cap_is_measured": False, "profile_notes": "Medium-speed corridor DC fast charger (75 kW)", "source_basis": "MoP 2024 expressway charging guidelines; 250 kW transformer headroom"},

        # NEW: Urban Fleet Depot (IN-FLT-01)
        {"environment": "Urban Fleet Depot", "station_id": "IN-FLT-01", "charger_id": "IN-FLT-01-CP01", "connector_type": "Bharat AC-001", "rated_power_kw": 3.3, "station_aggregate_power_cap_kw_scenario": 80.0, "power_cap_is_measured": False, "profile_notes": "Fleet depot AC charging for 2W/3W commercial vehicles", "source_basis": "BEE/MoP guidelines; 80 kW site connection"},
        {"environment": "Urban Fleet Depot", "station_id": "IN-FLT-01", "charger_id": "IN-FLT-01-CP02", "connector_type": "Bharat AC-001", "rated_power_kw": 3.3, "station_aggregate_power_cap_kw_scenario": 80.0, "power_cap_is_measured": False, "profile_notes": "Fleet depot AC charging for 2W/3W commercial vehicles", "source_basis": "BEE/MoP guidelines; 80 kW site connection"},
        {"environment": "Urban Fleet Depot", "station_id": "IN-FLT-01", "charger_id": "IN-FLT-01-CP03", "connector_type": "Bharat DC-001", "rated_power_kw": 15.0, "station_aggregate_power_cap_kw_scenario": 80.0, "power_cap_is_measured": False, "profile_notes": "Depot DC fast charger for commercial 3W cargo", "source_basis": "BEE/MoP Bharat DC-001 guidelines; 80 kW site connection"},
        {"environment": "Urban Fleet Depot", "station_id": "IN-FLT-01", "charger_id": "IN-FLT-01-CP04", "connector_type": "Type-2 AC", "rated_power_kw": 7.2, "station_aggregate_power_cap_kw_scenario": 80.0, "power_cap_is_measured": False, "profile_notes": "Depot AC fast charger for electric taxi cars", "source_basis": "BEE/MoP Type-2 AC standard; 80 kW site connection"},
    ]
    return pd.DataFrame(stations)

def update_excel_file():
    print(f"Loading master catalog...")
    catalog_df = generate_catalog_dataframe()

    print(f"Reading existing workbook from {EXCEL_PATH}...")
    xls = pd.ExcelFile(EXCEL_PATH)
    sheets_dict = {name: pd.read_excel(EXCEL_PATH, sheet_name=name) for name in xls.sheet_names}

    # 1. Update EV_Model_Catalog with complete 56-vehicle source-verified catalogue
    sheets_dict["EV_Model_Catalog"] = catalog_df

    # 2. Build and update Station_Configs
    sheets_dict["Station_Configs"] = build_extended_stations()

    # 3. Build Extended Sessions
    extended_df = build_extended_scenarios(catalog_df)

    # Combine baseline 48 + extended 48 into EV_Sessions_Combined
    baseline_df = sheets_dict["EV_Sessions_48"].copy()
    combined_df = pd.concat([baseline_df, extended_df], ignore_index=True)

    sheets_dict["EV_Sessions_Extended"] = extended_df
    sheets_dict["EV_Sessions_Combined"] = combined_df

    print(f"Writing updated Excel file with {len(sheets_dict)} sheets...")
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for sname, sdata in sheets_dict.items():
            sdata.to_excel(writer, sheet_name=sname, index=False)
            print(f"  - Sheet '{sname}': {len(sdata)} rows, {len(sdata.columns)} cols")

    print("Master Excel workbook successfully updated.")

if __name__ == "__main__":
    update_excel_file()
