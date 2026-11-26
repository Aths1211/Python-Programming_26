# UI changes

- Simplified primary navigation to Overview, Register EV, Live Queue, Charging Operations, Analytics and Settings. Infrastructure editing is a Settings subsection; research reference data remains inside Analytics.
- Overview highlights waiting vehicles, active/available/occupied chargers, completed sessions and facility power utilization.
- Registration keeps model, registration, SOC, battery size and arrival/departure details visible. Connector and vehicle-power overrides stay under Advanced options.
- Demo requests are opt-in, marked simulated and arrive together so the first dispatch snapshot demonstrates multiple compatible endpoints in use.
- Made Hybrid Priority the live-dispatch policy, with expandable weighted-score controls and an inspectable ranked queue. FCFS, EDF and LAFS remain intact as baselines in four-policy comparisons; power sharing stays a separate LAFS-style stage.
- Separated facility dispatch analytics from the bundled research benchmark. Operational charts show an empty state until a dispatch exists. The reference catalog, data-quality view and exports remain available.
- Added a simulation replay control to Charging Operations. Selected progress, SOC, assignment and request status persist across restarts; completed and expired requests are excluded from active dispatch.
- Corrected the displayed allocation to use actual timeline power, clipped energy-completion and off-grid departure intervals, and aligned endpoint/queue statuses to the selected simulation snapshot. Simulation-only status is explicit.
- Reduced card borders, adjusted spacing and typography, added responsive CSS and removed the remote font dependency.
- Rebranded application to **VoltFlow** with the official tagline **"Intelligent EV Charging & Scheduling Platform"**. Updated browser tab title, modern SVG lightning pulse branding in sidebar, header navigation eyebrows, export filenames (`VoltFlow_EV_Charging_Export.xlsx`), and global footer while preserving all underlying scheduling algorithms, facility store schemas, and UI navigation structure.
- Expanded `smoke_test.py` to use a disposable project copy and cover empty startup, duplicate registration, demo loading, parallel dispatch, simulation progress, restart persistence, infrastructure, analytics and queue clearing.
- Registration, demo loads, queue edits and infrastructure updates now trigger automatic dispatch; unchanged Streamlit reruns are guarded by a stable scheduling-input key. Run Scheduler remains an explicit optional re-run.
- Added distinct UUID-backed request and session IDs, database uniqueness/active-registration guards, safe migration of legacy ticket IDs and idempotent demo insertion.
- Queue rows identify Waiting, Assigned, Charging and Completed states, show assignment/progress and explain why a request is waiting.
- Added `dispatch_service.py` to validate dispatch results against request identity, station/connector compatibility, endpoint ratings, station caps, facility capacity and arrival/departure bounds before persistence.
- Added `dispatch_tests.py` for automatic allocation, completion handoff, future arrivals, multi-station isolation, 250 kW limits, repeated demo/scheduler runs, persistence recovery and all four policies.

The redesign preserves the local multi-facility store, dispatch algorithms, queue rules, station/vehicle compatibility checks, grid limits, audit history and benchmark data.
