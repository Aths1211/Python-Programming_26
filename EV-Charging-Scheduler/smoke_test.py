"""Headless smoke test: python smoke_test.py  (run from the project folder)."""
import logging, os, sys, shutil, tempfile, sqlite3, gc, time, math, json
logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pandas as pd
import data_loader as dl
from streamlit.testing.v1 import AppTest

fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail and not ok else ""))
    if not ok: fails.append(name)

def saved_schedule_count(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM schedules").fetchone()[0]
    finally:
        con.close()

# Unit-check the UI restart cursor without launching Streamlit or touching the operator database.
import ast
with open(os.path.join(HERE, "app.py"), encoding="utf-8") as _app_source:
    _app_tree = ast.parse(_app_source.read())
_restore_node = next(node for node in _app_tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_restored_snapshot_minute")
_restore_module = ast.fix_missing_locations(ast.Module(body=[_restore_node], type_ignores=[]))
_restore_ns = {}
exec(compile(_restore_module, "app.py", "exec"), _restore_ns)
_restore_minute = _restore_ns["_restored_snapshot_minute"]
check("restart snapshot follows active requests instead of stale completed history",
      _restore_minute([{"status": "Completed", "simulation_minute": 30},
                       {"status": "Charging", "simulation_minute": 5},
                       {"status": "Waiting", "simulation_minute": 5}]) == 5
      and _restore_minute([{"status": "Completed", "simulation_minute": 30},
                           {"status": "Failed", "simulation_minute": 25}]) == 30)
_clock_node = next(node for node in _app_tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_advance_simulation_clock")
_clock_module = ast.fix_missing_locations(ast.Module(body=[_clock_node], type_ignores=[]))
_clock_ns = {"math": math}
exec(compile(_clock_module, "app.py", "exec"), _clock_ns)
_advance_clock = _clock_ns["_advance_simulation_clock"]
check("simulation clock advances at selected speed, stays monotonic, and stops at fleet end",
      _advance_clock(10.0, 0.5, 2.0, 30.0) == 11.0
      and _advance_clock(10.0, 5.0, 10.0, 30.0) == 30.0
      and _advance_clock(10.0, -1.0, 5.0, 30.0) == 10.0)

# 1. epoch anchor consistency
s = dl.load_sessions("Combined")
ok = ((s["epoch_anchor_IST"] + pd.to_timedelta(s["arrival_epoch_min"], unit="m")) == s["connection_time_IST"]).all()
check("epoch anchor + arrival_epoch_min == connection_time_IST", ok)

# 2. station catalog matches dataset
stations = dl.load_stations()
cat = {d["station_id"] for d in dl.get_station_catalog()}
check("station catalog == dataset stations", cat == set(stations["station_id"]))

# 3. demo fleet: every vehicle has a compatible charger at its station; unique live IDs
for sid in sorted(cat):
    fleet = dl.get_default_live_fleet(sid)
    conns = list(stations.loc[stations["station_id"] == sid, "connector_type"])
    bad = [v["vehicle_reg_no"] for v in fleet
           if not any(dl.is_connector_compatible(v["connector_type"], c) for c in conns)]
    check(f"demo fleet compatible at {sid} ({len(fleet)} EVs)", bool(fleet) and not bad, str(bad))
    if sid == "IN-HWY-01":
        compatible_ports = stations.loc[stations["station_id"] == sid, "charger_id"].nunique()
        check("parallel demo scenario arrives together on configured multi-port infrastructure",
              len(fleet) >= 5 and compatible_ports >= 4
              and len({v["arrival_time_IST"] for v in fleet}) == 1
              and all(v.get("record_type") == "demo" for v in fleet))
q = [{"session_id": "LIVE-EV-001"}, {"session_id": "LIVE-EV-007"}]
check("next_live_session_id is max+1", dl.next_live_session_id(q) == "LIVE-EV-008")
check("AC-only Type-2 EV is not allowed on CCS2 DC-only charger", True if dl.is_connector_compatible("Type-2 AC", "CCS2 DC") is False else False)

# 4. UI runs in a disposable copy so smoke actions cannot alter the operator's queue or facilities.
_temp = tempfile.TemporaryDirectory(prefix="voltflow-ui-smoke-")
_test_root = _temp.name
for _name in ("app.py", "data_loader.py", "facility_store.py", "scheduler.py", "dispatch_service.py", "visualizations.py"):
    shutil.copy2(os.path.join(HERE, _name), os.path.join(_test_root, _name))
os.makedirs(os.path.join(_test_root, "data"), exist_ok=True)
for _name in os.listdir(os.path.join(HERE, "data")):
    if _name.lower().endswith(".xlsx") or _name == "live_queue.json":
        shutil.copy2(os.path.join(HERE, "data", _name), os.path.join(_test_root, "data", _name))
app = os.path.join(_test_root, "app.py")
# AppTest does not implement timer-driven fragment reruns. Keep the same clock
# code path but drive ticks with deterministic full-app test reruns instead.
os.environ["VFLOW_CLOCK_TEST_MODE"] = "1"
at = AppTest.from_file(app, default_timeout=60).run()
check("app loads", not at.exception, str([e.value[:80] for e in at.exception]))
check("fresh facility starts with an empty operational queue",
      not at.exception and not at.session_state.get("live_queue", []),
      str([e.value[:80] for e in at.exception]))
check("simulation controls are available without loading demonstration data",
      any(x.label == "Play" and x.disabled for x in at.button)
      and any(x.label == "Pause" and x.disabled for x in at.button)
      and any(x.label == "Reset" and x.disabled for x in at.button)
      and any(x.label == "Simulation speed" for x in at.selectbox),
      f"buttons={[x.label for x in at.button]}; selectboxes={[x.label for x in at.selectbox]}")
check("interface labels itself as simulation mode",
      any("SIMULATION MODE" in x.value for x in at.markdown))
metric_labels = {x.label.lower() for x in at.metric}
check("overview shows active, available, and occupied charger counts",
      {"vehicles charging", "available chargers", "occupied chargers"}.issubset(metric_labels),
      str(metric_labels))
nav = [x for x in at.radio if x.label == "Primary navigation"]
for _section in ("Register EV", "Live Queue", "Charging Operations", "Analytics", "Settings"):
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value(_section); at.run()
    check(f"navigation opens {_section}", not at.exception, str([e.value[:80] for e in at.exception]))
    if _section == "Settings":
        save = [x for x in at.button if x.label == "Save settings"]
        if save:
            save[0].click(); at.run()
            check("facility settings form saves", not at.exception, str([e.value[:80] for e in at.exception]))
        section = [x for x in at.radio if x.label == "Settings section"]
        if section:
            section[0].set_value("Infrastructure"); at.run()
            save = [x for x in at.button if x.label == "Save infrastructure"]
            if save:
                save[0].click(); at.run()
                check("advanced infrastructure form saves", not at.exception, str([e.value[:80] for e in at.exception]))
            section = [x for x in at.radio if x.label == "Settings section"]
            if section:
                section[0].set_value("Facility"); at.run()
    if _section == "Analytics":
        check("facility analytics has an honest empty state before dispatch",
              any("No operational analytics yet" in x.value for x in at.info))
nav = [x for x in at.radio if x.label == "Primary navigation"]
if nav:
    nav[0].set_value("Live Queue"); at.run()
b = [x for x in at.button if "demonstration scenario" in x.label.lower()]
if b:
    b[0].click(); at.run()
    check("optional demo scenario loads only on request and is labeled simulated",
          not at.exception and any(q.get("record_type") == "demo" for q in at.session_state.get("live_queue", [])),
          str([e.value[:80] for e in at.exception]))
    demo_count = len(at.session_state.get("live_queue", []))
    demo_ids = [(q.get("request_id"), q.get("session_id")) for q in at.session_state.get("live_queue", [])]
    b = [x for x in at.button if "demonstration scenario" in x.label.lower()]
    if b:
        b[0].click(); at.run()
    check("loading demo twice creates no duplicate active request or session",
          not at.exception and len(at.session_state.get("live_queue", [])) == demo_count
          and len({q.get("vehicle_reg_no") for q in at.session_state.get("live_queue", [])}) == demo_count
          and [(q.get("request_id"), q.get("session_id")) for q in at.session_state.get("live_queue", [])] == demo_ids,
          str([e.value[:80] for e in at.exception]))
    check("demo dispatch automatically schedules without the manual button",
          not at.exception and at.session_state.get("live_schedule_results") is not None
          and at.session_state.get("live_dispatch_key") is not None,
          str([e.value[:80] for e in at.exception]))
    check("live queue shows its four operator sections",
          all(any(label.lower() in x.value.lower() for x in at.markdown)
              for label in ("Charging now", "Waiting for charger", "Upcoming arrivals", "Session history")),
          str([x.value for x in at.markdown if "Charging" in x.value or "Waiting" in x.value]))
    check("live queue has an event-aligned simulation timeline",
          any("Simulation timeline" in x.label for x in at.select_slider)
          and any(x.label == "Advance to next event" for x in at.button),
          f"sliders={[x.label for x in at.select_slider]}; buttons={[x.label for x in at.button]}")
    _initial_minute = at.session_state.get("operational_snapshot_minute")
    _next_event = [x for x in at.button if x.label == "Advance to next event"]
    if _next_event:
        _next_event[0].click(); at.run()
    check("event advance follows the scheduler's next real event",
          not at.exception and at.session_state.get("operational_snapshot_minute") is not None
          and at.session_state.get("operational_snapshot_minute") > _initial_minute,
          f"before={_initial_minute}; after={at.session_state.get('operational_snapshot_minute')}")
    speed_control = [x for x in at.selectbox if x.label == "Simulation speed"]
    if speed_control:
        speed_control[0].set_value("5x"); at.run()
    _clock_before_play = float(at.session_state.get("operational_snapshot_minute"))
    _energy_before_play = sum(float(q.get("energy_delivered_kwh", 0.0))
                              for q in at.session_state.get("live_queue", []))
    play = [x for x in at.button if x.label == "Play"]
    if play:
        play[0].click(); at.run()
    check("Play starts the autonomous simulation and selected speed is retained",
          not at.exception and at.session_state.get("simulation_playing") is True
          and at.session_state.get("simulation_speed") == "5x",
          str([e.value[:80] for e in at.exception]))
    time.sleep(1.1)
    at.run()
    _clock_after_play = float(at.session_state.get("operational_snapshot_minute"))
    _energy_after_play = sum(float(q.get("energy_delivered_kwh", 0.0))
                             for q in at.session_state.get("live_queue", []))
    check("autonomous clock rerun advances the saved snapshot and charging energy",
          not at.exception and _clock_after_play > _clock_before_play + 0.5
          and _energy_after_play > _energy_before_play,
          f"minute={_clock_before_play}->{_clock_after_play}; energy={_energy_before_play:.3f}->{_energy_after_play:.3f}")
    _live_results_after_tick = at.session_state.get("live_schedule_results", [])
    _first_finished = next((r for r in _live_results_after_tick
                            if r.charge_end_min is not None and r.charge_end_min <= _clock_after_play), None)
    _handoff_result = next((r for r in _live_results_after_tick
                            if r.charge_start_min is not None
                            and r.charge_start_min > min(x.arrival_min for x in _live_results_after_tick)), None)
    _status_by_session = {q.get("session_id"): q.get("status")
                          for q in at.session_state.get("live_queue", [])}
    check("autonomous playback records completion and hands the released charger to the next EV",
          _first_finished is not None and _handoff_result is not None
          and _status_by_session.get(_first_finished.session_id) == "Completed"
          and _status_by_session.get(_handoff_result.session_id) == "Charging",
          f"completed={_status_by_session.get(getattr(_first_finished, 'session_id', ''))}; handoff={_status_by_session.get(getattr(_handoff_result, 'session_id', ''))}; minute={_clock_after_play}")
    pause = [x for x in at.button if x.label == "Pause"]
    if pause:
        pause[0].click(); at.run()
    check("Pause stops simulation progression",
          not at.exception and at.session_state.get("simulation_playing") is False,
          str([e.value[:80] for e in at.exception]))
    _clock_persisted_minute = at.session_state.get("operational_snapshot_minute")
    restarted_clock = AppTest.from_file(app, default_timeout=60).run()
    check("restart restores autonomous-clock progress paused from the database",
          not restarted_clock.exception and not restarted_clock.session_state.get("simulation_playing")
          and abs(float(restarted_clock.session_state.get("operational_snapshot_minute"))
                  - float(_clock_persisted_minute)) < 1e-4,
          f"saved={_clock_persisted_minute}; restored={restarted_clock.session_state.get('operational_snapshot_minute')}; errors={[e.value[:80] for e in restarted_clock.exception]}")
    del restarted_clock
    reset_clock = [x for x in at.button if x.label == "Reset"]
    if reset_clock:
        reset_clock[0].click(); at.run()
    _clock_first_minute = min(r.arrival_min for r in at.session_state.get("live_schedule_results", []))
    check("Reset returns the persisted simulation and vehicle state to the first arrival",
          not at.exception and at.session_state.get("simulation_playing") is False
          and abs(float(at.session_state.get("operational_snapshot_minute")) - _clock_first_minute) < 1e-6,
          str([e.value[:80] for e in at.exception]))
    _queue_sliders = [x for x in at.select_slider if "Simulation timeline" in x.label]
    if _queue_sliders and at.session_state.get("live_schedule_results"):
        _start_minute = min(r.arrival_min for r in at.session_state["live_schedule_results"])
        _queue_sliders[0].set_value(_start_minute); at.run()
    check("primary queue tables omit full request and session ID columns",
          all(not any(str(column).lower() in ("request id", "session id", "request_id", "session_id")
                      for column in frame.value.columns)
              for frame in at.dataframe if getattr(frame.value, "columns", None) is not None
              and not {"Request ID", "Session ID"}.issubset(set(frame.value.columns))),
          str([list(frame.value.columns) for frame in at.dataframe
               if getattr(frame.value, "columns", None) is not None]))
    check("technical identifiers are behind the optional details expander",
          any("Optional technical details" in x.label for x in at.expander)
          and not any("req_" in x.value or "ses_" in x.value for x in at.markdown),
          str([x.value for x in at.markdown if "req_" in x.value or "ses_" in x.value]))
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value("Register EV"); at.run()
    reg_input = [x for x in at.text_input if x.label == "Vehicle Registration Plate"]
    register_btn = [x for x in at.button if x.label == "Register EV and add to queue"]
    if reg_input and register_btn:
        reg_input[0].set_value("UI-TEST-001"); at.run()
        register_btn = [x for x in at.button if x.label == "Register EV and add to queue"]
        register_btn[0].click(); at.run()
        check("vehicle registration adds a persistent operational request without an app exception",
              not at.exception and any(q.get("vehicle_reg_no") == "UI-TEST-001"
                  and q.get("record_type") == "registered" for q in at.session_state.get("live_queue", [])),
              str([e.value[:80] for e in at.exception]))
        registered = next(q for q in at.session_state.get("live_queue", []) if q.get("vehicle_reg_no") == "UI-TEST-001")
        scheduled_ids = {r.session_id for r in at.session_state.get("live_schedule_results", [])}
        check("registration automatically dispatches and keeps request/session IDs distinct",
              not at.exception and registered.get("request_id") != registered.get("session_id")
              and registered.get("session_id") in scheduled_ids
              and at.session_state.get("live_dispatch_key") is not None,
              str([e.value[:80] for e in at.exception]))
        registered_count = len(at.session_state.get("live_queue", []))
        duplicate = [x for x in at.button if x.label == "Register EV and add to queue"]
        if duplicate:
            duplicate[0].click(); at.run()
        check("duplicate active registration is rejected",
              len(at.session_state.get("live_queue", [])) == registered_count
              and any("already active" in x.value.lower() for x in at.error),
              str([x.value for x in at.error]))
    _db_path = os.path.join(_test_root, "data", "gridcharge.sqlite3")
    _schedules_before_reruns = saved_schedule_count(_db_path)
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value("Charging Operations"); at.run()
    _schedules_after_navigation = saved_schedule_count(_db_path)
    check("ordinary Streamlit navigation reruns do not create another dispatch",
          _schedules_after_navigation == _schedules_before_reruns,
          f"before={_schedules_before_reruns}; after={_schedules_after_navigation}")
    g = [x for x in at.button if x.label == "Run Scheduler"]
    if g: g[0].click(); at.run()
    check("live mode Hybrid dispatch runs", not at.exception and at.session_state.get("live_algo") == "HYBRID",
          str([e.value[:80] for e in at.exception]))
    _schedules_after_manual_rerun = saved_schedule_count(_db_path)
    check("re-running identical inputs does not create duplicate saved schedule sessions",
          _schedules_after_manual_rerun == _schedules_after_navigation,
          f"before={_schedules_after_navigation}; after={_schedules_after_manual_rerun}")
    check("dispatch assigned simultaneous compatible chargers under facility cap",
          not at.exception and at.session_state.get("live_schedule_results") is not None
          and sum(bool(r.timeline and r.timeline[0][0] == min(x.arrival_min for x in at.session_state["live_schedule_results"]))
                  for r in at.session_state["live_schedule_results"]) >= 2,
          str([e.value[:80] for e in at.exception]))
    compare = [x for x in at.button if x.label == "Compare policies"]
    if compare:
        compare[0].click(); at.run()
        comparison = at.session_state.get("live_policy_comparison")
        check("live queue comparison includes all four policies",
              not at.exception and comparison is not None and set(comparison["Algorithm"]) == {"FCFS", "EDF", "LAFS", "HYBRID"},
              str([e.value[:80] for e in at.exception]))
    ops_slider = [x for x in at.select_slider if "Simulation time" in x.label]
    if ops_slider:
        results = at.session_state.get("live_schedule_results", [])
        replay_end = int(max(r.deadline_min for r in results))
        replay_start = int(min(r.arrival_min for r in results))
        mid_minute = min(replay_start + 5, replay_end)
        ops_slider[0].set_value(mid_minute); at.run()
        _mid_saved = {"minute": at.session_state.get("operational_snapshot_minute"),
                      "anchor": str(at.session_state.get("live_epoch_anchor")),
                      "key": at.session_state.get("live_dispatch_key")}
        mid_state = {q.get("session_id"): (q.get("status"), q.get("energy_delivered_kwh", 0.0))
                     for q in at.session_state.get("live_queue", [])}
        restarted_mid = AppTest.from_file(app, default_timeout=60).run()
        _mid_restored = {"minute": restarted_mid.session_state.get("operational_snapshot_minute"),
                         "anchor": str(restarted_mid.session_state.get("live_epoch_anchor")),
                         "key": restarted_mid.session_state.get("live_dispatch_key")}
        mid_restored = {q.get("session_id"): (q.get("status"), q.get("energy_delivered_kwh", 0.0))
                        for q in restarted_mid.session_state.get("live_queue", [])}
        check("restart restores in-progress session status and delivered energy",
              not restarted_mid.exception and mid_state == mid_restored
              and any(status == "Charging" and energy > 0 for status, energy in mid_restored.values()),
              f"saved={mid_state}; restored={mid_restored}; snapshots={_mid_saved}->{_mid_restored}; errors={[e.value[:80] for e in restarted_mid.exception]}")
        ops_slider = [x for x in at.select_slider if "Simulation time" in x.label]
        if ops_slider:
            ops_slider[0].set_value(replay_end); at.run()
        check("simulation replay persists progress to operational requests",
              not at.exception and all(q.get("simulation_minute") == replay_end
                  for q in at.session_state.get("live_queue", []) if q.get("session_id") in
                  {r.session_id for r in results}),
              str([e.value[:80] for e in at.exception]))
        check("end-of-run snapshot records completed and expired states",
              any(q.get("status") == "Completed" for q in at.session_state.get("live_queue", []))
              and all(q.get("status") in ("Completed", "Failed") for q in at.session_state.get("live_queue", [])),
              str([(q.get("vehicle_reg_no"), q.get("status")) for q in at.session_state.get("live_queue", [])]))
        check("simulation clock reports completion and disables further Play",
              any("Complete · Current simulation time" in x.value for x in at.caption)
              and any(x.label == "Play" and x.disabled for x in at.button),
              f"captions={[x.value for x in at.caption if 'Current simulation time' in x.value]}")
        restarted = AppTest.from_file(app, default_timeout=60).run()
        check("application restart restores queue progress and saved schedule",
              not restarted.exception and restarted.session_state.get("live_schedule_results")
              and any(q.get("simulation_minute") == replay_end for q in restarted.session_state.get("live_queue", [])),
              str([e.value[:80] for e in restarted.exception]))
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value("Settings"); at.run()
        section = [x for x in at.radio if x.label == "Settings section"]
        if section:
            section[0].set_value("Infrastructure"); at.run()
        check("endpoint page reflects a simulated charger state", not at.exception,
              str([e.value[:80] for e in at.exception]))
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value("Analytics"); at.run()
        view = [x for x in at.selectbox if x.label == "Analytics view"]
        if view:
            view[0].set_value("Facility activity"); at.run()
            check("facility analytics use the saved dispatch without runtime errors", not at.exception,
                  str([e.value[:80] for e in at.exception]))
            _analytics_db = sqlite3.connect(_db_path)
            try:
                _latest_json = _analytics_db.execute(
                    "SELECT results_json FROM schedules WHERE facility_id=? ORDER BY id DESC LIMIT 1",
                    (at.session_state.get("active_facility_id"),),
                ).fetchone()
            finally:
                _analytics_db.close()
            _database_results = json.loads(_latest_json[0]) if _latest_json else []
            _database_energy = sum(float(result.get("energy_delivered_kwh", 0.0))
                                   for result in _database_results)
            _analytics_metrics = {metric.label: str(metric.value) for metric in at.metric}
            check("H. facility analytics sessions and energy match persisted simulation results",
                  not at.exception and _database_results
                  and _analytics_metrics.get("Sessions") == str(len(_database_results))
                  and _analytics_metrics.get("Energy delivered") == f"{_database_energy:.1f} kWh",
                  f"db_sessions={len(_database_results)}; db_energy={_database_energy:.3f}; metrics={_analytics_metrics}")
            view = [x for x in at.selectbox if x.label == "Analytics view"]
        if view:
            view[0].set_value("Algorithm comparison"); at.run()
            check("algorithm comparison analytics load", not at.exception,
                  str([e.value[:80] for e in at.exception]))
            view = [x for x in at.selectbox if x.label == "Analytics view"]
            if view:
                view[0].set_value("Reference data"); at.run()
                check("reference data view loads", not at.exception, str([e.value[:80] for e in at.exception]))
                check("vehicle specification inspector remains available",
                      any(x.label == "Model" for x in at.selectbox))
        nav = [x for x in at.radio if x.label == "Primary navigation"]
        if nav:
            nav[0].set_value("Overview"); at.run()
            check("overview snapshot control is available after dispatch",
                  any("simulation timeline" in x.label.lower() for x in at.select_slider),
                  f"widgets={[x.label for x in at.select_slider]}; nav={[x.value for x in at.radio if x.label == 'Primary navigation']}; results={at.session_state.get('live_schedule_results') is not None}; anchor={at.session_state.get('live_epoch_anchor')}; exceptions={[getattr(e, 'stack_trace', e.value) for e in at.exception]}")
    seen = set()
    results = at.session_state.get("live_schedule_results", [])
    if results:
        lo = int(min(r.arrival_min for r in results)); hi = int(max(r.deadline_min for r in results))
        minute_options = list(range(lo, hi + 1, 5))
        if hi not in minute_options: minute_options.append(hi)
    else:
        minute_options = []
    for m in (0, 10, 75, 120):
        sliders = [x for x in at.select_slider if "simulation timeline" in x.label.lower()]
        if not sliders or not minute_options:
            break
        sl = sliders[0]   # re-fetch: AppTest handles go stale
        m = min(minute_options, key=lambda option: abs(option - m))
        sl.set_value(m); at.run()
        seen.add({x.label.lower(): x.value for x in at.metric}.get("occupied chargers"))
    check("snapshot slider changes occupancy", len(seen) > 1, str(seen))
    nav = [x for x in at.radio if x.label == "Primary navigation"]
    if nav:
        nav[0].set_value("Live Queue"); at.run()
    clear = [x for x in at.button if x.label == "Clear queue"]
    if clear:
        clear[0].click(); at.run()
        check("queue clear action completes without an app exception", not at.exception,
              str([e.value[:80] for e in at.exception]))
print("\nFAILED: " + ", ".join(fails) if fails else "\nAll smoke checks passed.")
del at
if "restarted" in globals():
    del restarted
gc.collect()
time.sleep(0.2)
_temp.cleanup()
sys.exit(1 if fails else 0)
