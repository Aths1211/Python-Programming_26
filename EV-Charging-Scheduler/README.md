# ⚡ VoltFlow — Intelligent EV Charging & Scheduling Platform

**VoltFlow** is an enterprise-grade, dark-themed EV charging management, fleet dispatch, and multi-algorithm optimization platform designed according to Linear, Stripe, and Vercel design principles.

Official Tagline: **Intelligent EV Charging & Scheduling Platform**

Built for charging network operators, fleet managers, and grid research teams to monitor infrastructure, register driver demand, dispatch multi-vehicle fleets under transformer limits, and benchmark optimization algorithms.

---

## 🎨 Design System & Color Palette

The interface strictly adheres to a modern, restrained dark design system:

| Design Token | Hex Code | Purpose |
|---|---|---|
| **Main Background** | `#0B1014` | Application canvas |
| **Sidebar Background** | `#0E151B` | Navigation and facility switcher |
| **Primary Surface** | `#121B22` | Select panels and chart surfaces |
| **Secondary Surface** | `#18232B` | Inputs and secondary controls |
| **Primary Border** | `#2A3943` | Focus and selected-state boundaries |
| **Subtle Border** | `#202D36` | Quiet dividers and table boundaries |
| **Primary Text** | `#E8F0F2` | Headings and values |
| **Secondary Text** | `#A9B8BE` | Labels and table content |
| **Muted Text** | `#82939B` | Captions and units |
| **Jade Accent** | `#22C7A5` | Primary actions and active states |
| **Accent Hover** | `#36DDB3` | Hover states |
| **Secondary Blue** | `#76A9FF` | Active charging and information |
| **Warning State** | `#F4BE5E` | Constraints and partial delivery |
| **Critical / Error** | `#F07878` | Infeasible demand and warnings |
| **Success State** | `#22C7A5` | Available and completed states |

---

## 🧭 Navigation Architecture

The app has six primary sections. Reference datasets remain available as an Analytics subview, so they do not compete with day-to-day operations in the main navigation.

| Section | Purpose |
|---|---|
| **Overview** | Six operational signals, station health, charger counts and recent dispatch activity. |
| **Register EV** | Check in a vehicle, set its SOC window and arrival/departure times. |
| **Live Queue** | Review registered and clearly marked demo requests, progress, assignment and status. |
| **Charging Operations** | See automatic dispatch and replay simulated charging progress; manual re-run is optional. |
| **Analytics** | Review the latest facility dispatch or run a separately labeled research benchmark. |
| **Settings** | Configure facility details and edit infrastructure in the Infrastructure section. |

The interface uses a restrained graphite surface, jade accents, quieter section separators, compact operational summaries and responsive layouts. Advanced vehicle and scheduler controls stay in expandable sections. Overview emphasizes six core signals and a short activity table instead of repeating the same details in multiple cards.

### Existing capabilities retained

- The vehicle catalog still prefills battery and AC/DC specifications; advanced fields allow overrides, connector selection and arrival/departure windows.
- Queue actions still support idempotent demo loading, CSV export, request edits, status changes, removal and clearing.
- Station and charger editors support adding/removing rows and changing names, locations, power, connector, AC/DC type and availability.
- Live dispatch uses Hybrid Priority. Research benchmarks preserve FCFS, EDF and LAFS as independent baselines and run all four policies on identical inputs.
- Analytics retains facility power, energy, waiting, completion and utilization views; the reference-data view keeps the vehicle catalog, research sessions, quality checks and exports.

## 🏢 Facility-Independent Operations

Operational state is stored locally in `data/gridcharge.sqlite3`, independently of the bundled research workbook. Facilities have their own stable IDs, timezone, location, operating hours, currency/tariff, grid limit, dispatch policy, simulation step, stations, connector mix, charger ratings/status, live requests, schedules and audit history. Create and edit this configuration in **Settings**; changing the active facility switches its queue and latest dispatch with it. Archiving preserves local history.

The initial facilities are explicitly marked **Demonstration dataset** and are seeded once from the included reference catalog. Their station configurations and scenario records are illustrative, not real-time Indian charging station information or connected hardware. A newly initialized facility starts with an empty operational queue. Use **Load simulated demo scenario** in Live Queue for an examiner-ready parallel charging example. Demo requests are marked Demo; user entries are marked Registered.

Dispatch considers every configured compatible endpoint in the facility. It checks station and endpoint availability, vehicle/connector limits, station caps and the shared facility grid cap. Requests without an eligible endpoint remain unassigned in the queue. The Overview separates active delivery, available endpoints and occupied/reserved endpoints.

Registering a valid EV saves its queue request and automatically calculates a dispatch. The simulation plans arrivals and charger releases across the whole queue, so future arrivals wait until their configured time and later eligible vehicles start when an endpoint becomes free. A stable input fingerprint prevents Streamlit reruns from creating another schedule for unchanged inputs; **Run Scheduler** remains available for an explicit re-run/optimization. Queue records carry a unique `request_id` and a separate persistent `session_id`. SQLite constraints and upgrade migrations preserve legacy requests while remapping old ticket IDs and saved results.

## Hybrid Priority methodology

Live scheduling combines three normalized queue scores. **FCFS** is accumulated waiting age divided by the oldest request's waiting age. **EDF** maps remaining slack time to 0–1 urgency, with the shortest slack at 1. **Fairness** is accumulated service debt: elapsed share of the arrival window minus delivered-energy progress, clipped to 0–1. This keeps the fairness input useful for unserved requests that have not yet received energy. The combined score is w_fcfs·FCFS + w_edf·EDF + w_fairness·Fairness; weights must be finite, nonnegative and sum to 1. Defaults are exactly one-third each.

Eligible waiting requests are ordered by descending combined score, then earliest departure deadline, earliest arrival, and lexical session ID. The priority list exposes scores, charger compatibility and a plain-language reason. Simulation recalculates scores every step as arrivals, remaining deadline and delivered energy change. The ranked queue decides which waiting requests connect when endpoints are scarce. Once connected, Hybrid uses equal-weight LAFS proportional max-min allocation as a separate stage, redistributing unused capacity when a vehicle reaches its own power limit. Charger ratings, vehicle AC/DC input limits, connector compatibility, station caps and the facility grid cap remain hard limits; infeasible requests receive partial delivery and are recorded as missed energy rather than removed.

Analytics compares **FCFS, EDF, LAFS and HYBRID** on the same scenario inputs. Completion Rate is the share of vehicles that meet their requested energy target; Energy Fulfilment by Deadline is the delivered share of requested kWh. Metrics also include requested/delivered/unmet energy, waiting time, charger utilization, peak station and facility demand, and Jain's fairness index over energy completion fractions. Undefined wait metrics remain blank when no session started. Results are scenario-dependent and do not claim that Hybrid always outperforms a baseline.

Charging Operations uses a deterministic simulation timeline. Move its simulation-time control (or the Overview snapshot control) to replay calculated progress. Queue status distinguishes Waiting, Assigned, Charging and Completed, and includes the reason for waiting. Request status, delivered energy, current SOC, charger assignment and allocation are persisted with the selected simulation snapshot. Completed requests remain in local history but are excluded from later active dispatches. This project has no hardware integration.

## 🧪 Verification & Automated Testing

The application includes automated headless UI tests, persistence checks and constraint verification. The scheduler suite contains 31 checks; facility and dispatch tests cover 3, 5 and 10 vehicle queues, automatic dispatch, ID migration/idempotency, multiple stations, all four policies, endpoint limits and the 250 kW facility limit:

```bash
# 1. Run headless smoke test suite (AppTest UI verification)
python smoke_test.py

# 2. Run scheduler constraint test suite (31 checks, including 3/5/10 vehicle concurrency)
python scheduler.py

# 3. Run facility persistence, isolation and multi-station checks
python facility_tests.py

# 4. Run automatic-dispatch, unique-ID and legacy-migration checks
python dispatch_tests.py
```

### Launch Application

```bash
streamlit run app.py
```

Dataset and database paths are resolved relative to the project files. Local SQLite persistence survives restarts on the same writable machine. Streamlit hosting commonly uses ephemeral storage; use persistent storage or back up data/gridcharge.sqlite3 if operational records must survive a deployment restart. Research data remains bundled separately in the workbook.
