"""
visualizations.py
=================
Plotly chart factories for VoltFlow: Intelligent EV Charging & Scheduling Platform.
All visual elements use 100% human-readable labels (strictly no raw snake_case).
Configured with an executive, modern Dark SaaS palette inspired by Linear, Stripe,
and Vercel design principles with high-contrast slate typography.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color Palette System ──────────────────────────────────────────────────────
# System colors:
# Canvas: #0B1014, Surface: #121B22, Border: #2A3943, Subtle: #202D36
# Primary Accent: #22C7A5, Secondary Blue: #76A9FF, Warning: #F4BE5E, Error: #F07878, Success: #22C7A5
ALGO_COLORS = {
    "FCFS": "#76A9FF",   # Secondary Blue
    "EDF":  "#F4BE5E",   # Warm Amber
    "LAFS": "#22C7A5",   # Emerald Accent
    "HYBRID": "#C4F77A", # Priority policy highlight
}
ALGO_ORDER = ["FCFS", "EDF", "LAFS", "HYBRID"]


def _observed_algorithms(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Algorithm" not in df.columns:
        return []
    observed = list(dict.fromkeys(df["Algorithm"].dropna().astype(str).tolist()))
    return [a for a in ALGO_ORDER if a in observed] + [a for a in observed if a not in ALGO_ORDER]


def _finite_numeric(values: pd.Series) -> pd.Series:
    """Coerce chart inputs to numeric and exclude NaN/infinite values."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric))


def _power_profile(sim_list):
    """Aggregate exact timeline intervals into a facility demand step profile."""
    events = {}
    for result in sim_list:
        for start, end, power in result.timeline:
            if end <= start or power <= 0:
                continue
            events[start] = events.get(start, 0.0) + power
            events[end] = events.get(end, 0.0) - power
    if not events:
        return [], []
    times = sorted(events)
    demand = 0.0
    loads = []
    for time in times:
        demand += events[time]
        loads.append(round(max(0.0, demand), 3))
    return times, loads

DARK_LAYOUT = dict(
    paper_bgcolor="#121B22",
    plot_bgcolor="#121B22",
    font=dict(color="#E8F0F2", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12),
    title_font=dict(size=13, color="#E8F0F2", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
    legend=dict(
        bgcolor="rgba(17, 24, 33, 0.95)",
        bordercolor="#2A3943",
        borderwidth=1,
        font=dict(color="#A9B8BE", size=11),
    ),
    xaxis=dict(
        gridcolor="#202D36",
        zerolinecolor="#2A3943",
        linecolor="#2A3943",
        tickfont=dict(color="#A9B8BE", size=11),
        title_font=dict(color="#E8F0F2", size=12),
        automargin=True,
    ),
    yaxis=dict(
        gridcolor="#202D36",
        zerolinecolor="#2A3943",
        linecolor="#2A3943",
        tickfont=dict(color="#A9B8BE", size=11),
        title_font=dict(color="#E8F0F2", size=12),
        automargin=True,
    ),
    margin=dict(l=45, r=30, t=45, b=40),
)

LIGHT_LAYOUT = DARK_LAYOUT


# ── 1. Gantt Timeline ────────────────────────────────────────────────────────

def fig_gantt(results_df: pd.DataFrame, algo: str, epoch_start) -> go.Figure:
    """
    Horizontal Gantt chart showing charging intervals for one algorithm.
    Safe against NaNs, missing dates, or zero-duration slices.
    """
    if results_df is None or results_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No simulation data available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    sub = results_df[results_df["Algorithm"] == algo].copy()
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No sessions found for algorithm: {algo}", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if epoch_start is None or pd.isna(epoch_start):
        epoch_start = pd.Timestamp("2026-11-24 00:00:00")
    else:
        epoch_start = pd.to_datetime(epoch_start)

    sub = sub.sort_values("Arrival (min)")

    rows = []
    for _, r in sub.iterrows():
        start_val = r.get("Charge Start (min)")
        if start_val is None or pd.isna(start_val):
            continue

        start_min = float(start_val)
        end_val = r.get("Charge End (min)")
        if end_val is not None and not pd.isna(end_val):
            end_min = float(end_val)
        else:
            deadline_val = r.get("Deadline (min)")
            end_min = float(deadline_val) if (deadline_val is not None and not pd.isna(deadline_val)) else start_min

        if end_min <= start_min:
            end_min = start_min + 1.0

        t0 = epoch_start + datetime.timedelta(minutes=start_min)
        t1 = epoch_start + datetime.timedelta(minutes=end_min)

        status = "Completed Target" if bool(r.get("Completed", False)) else "Partial / Incomplete"
        delivered_val = float(r.get("Delivered (kWh)", 0.0) or 0.0)
        requested_val = float(r.get("Requested (kWh)", 0.0) or 0.0)
        charger_assigned = str(r.get("Charger (Assigned)") or r.get("Charger (Dataset)") or "Unassigned")
        model_name = str(r.get("Model Variant") or r.get("Model ID") or "")

        rows.append(dict(
            Session=str(r["Session ID"]),
            Start=t0,
            Finish=t1,
            Status=status,
            Delivered=f"{delivered_val:.2f} kWh",
            Required=f"{requested_val:.2f} kWh",
            Charger=charger_assigned,
            Model=model_name,
        ))

    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No active charging sessions recorded for {algo}",
            showarrow=False, font_size=13, font_color="#82939B", xref="paper", yref="paper", x=0.5, y=0.5
        )
        fig.update_layout(title=f"Charging Schedule Timeline — {algo}", **DARK_LAYOUT)
        return fig

    df_gantt = pd.DataFrame(rows)
    color_map = {"Completed Target": "#22C7A5", "Partial / Incomplete": "#F4BE5E"}
    fig = px.timeline(
        df_gantt, x_start="Start", x_end="Finish", y="Session",
        color="Status", color_discrete_map=color_map,
        hover_data=["Model", "Charger", "Delivered", "Required"],
        title=f"Charging Schedule Timeline — {algo}",
        labels={"Session": "Session Identifier", "Start": "Start Time", "Finish": "End Time"},
    )
    fig.update_layout(
        height=max(360, len(rows) * 26 + 90),
        legend_title_text="Session Status",
        xaxis_title="Timeline (IST)",
        yaxis_title="Charging Session",
        **DARK_LAYOUT
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ── 2. Station Load Over Time ──────────────────────────────────────────────

def fig_station_load(results: dict, power_cap_kw: float | None = None, step_min: float = 1.0, epoch_start=None) -> go.Figure:
    """Aggregate station power demand (kW) over time."""
    fig = go.Figure()

    if epoch_start is not None and not pd.isna(epoch_start):
        epoch_start = pd.to_datetime(epoch_start)
    else:
        epoch_start = None

    for algo in ALGO_ORDER:
        if algo not in results:
            continue
        times, loads = _power_profile(results[algo])
        if not times:
            continue
        xs = [epoch_start + datetime.timedelta(minutes=t) for t in times] if epoch_start is not None else times
        fig.add_trace(go.Scatter(
            x=xs, y=loads,
            mode="lines", line_shape="hv", name=algo,
            line=dict(color=ALGO_COLORS[algo], width=2.5),
        ))

    if power_cap_kw is not None:
        fig.add_hline(
            y=power_cap_kw, line_dash="dash", line_color="#F07878", line_width=1.75,
            annotation_text=f"Facility Grid Limit ({power_cap_kw} kW)",
            annotation_position="top left",
            annotation_font=dict(color="#F07878", size=11),
        )

    if not fig.data:
        fig.add_annotation(text="No delivered-power intervals in the executed results", showarrow=False,
                           font_size=13, font_color="#82939B", xref="paper", yref="paper", x=0.5, y=0.5)

    fig.update_layout(
        title="Facility Power Demand Over Time",
        xaxis_title="Time (IST)" if epoch_start is not None else "Simulation Elapsed Time (minutes)",
        yaxis_title="Total Power Demand (kW)",
        legend_title_text="Algorithm",
        hovermode="x unified",
        **DARK_LAYOUT
    )
    return fig


# ── 3. Energy Comparison ───────────────────────────────────────────────────

def fig_energy_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Required vs Delivered vs Unserved Energy."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No energy data available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if "Requested (kWh)" in df.columns:
        clean = df.copy()
        for column in ("Requested (kWh)", "Delivered (kWh)", "Missed (kWh)"):
            clean[column] = _finite_numeric(clean[column]).fillna(0).clip(lower=0)
        summary = clean.groupby("Algorithm").agg(
            Required=("Requested (kWh)", "sum"),
            Delivered=("Delivered (kWh)", "sum"),
            Unserved=("Missed (kWh)", "sum"),
        )
        order = _observed_algorithms(df)
        summary = summary.reindex(order).reset_index()
    elif "Total Requested (kWh)" in df.columns:
        summary = df.rename(columns={
            "Total Requested (kWh)": "Required",
            "Total Delivered (kWh)": "Delivered",
            "Total Missed (kWh)": "Unserved",
        }).copy()
        for column in ("Required", "Delivered", "Unserved"):
            if column in summary.columns:
                summary[column] = _finite_numeric(summary[column]).fillna(0).clip(lower=0)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Incompatible dataframe schema for energy comparison", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Required Energy", x=summary["Algorithm"], y=summary["Required"],
        marker_color="#82939B", texttemplate="%{y:.1f}", textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Delivered Energy", x=summary["Algorithm"], y=summary["Delivered"],
        marker_color="#22C7A5", texttemplate="%{y:.1f}", textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Unserved Energy", x=summary["Algorithm"], y=summary["Unserved"],
        marker_color="#F07878", texttemplate="%{y:.1f}", textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        title="Energy Allocation Comparison by Algorithm",
        xaxis_title="Scheduling Algorithm",
        yaxis_title="Total Energy (kWh)",
        legend_title_text="Energy Metric",
        **DARK_LAYOUT
    )
    return fig


# ── 4. Waiting Time Distribution ──────────────────────────────────────────

def fig_waiting_time(results_df: pd.DataFrame) -> go.Figure:
    """Box plot of customer waiting times."""
    if results_df is None or results_df.empty or "Waiting (min)" not in results_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No waiting-time results available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    wait_values = _finite_numeric(results_df["Waiting (min)"])
    valid = wait_values.notna()
    if not valid.any():
        fig = go.Figure()
        fig.add_annotation(text="No session started; waiting time is undefined", showarrow=False,
                           font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    clean = results_df.loc[valid].copy()
    clean["Waiting (min)"] = wait_values.loc[valid]
    fig = px.box(
        clean, x="Algorithm", y="Waiting (min)",
        color="Algorithm", color_discrete_map=ALGO_COLORS,
        category_orders={"Algorithm": _observed_algorithms(results_df)},
        title="Waiting Time Distribution by Algorithm",
        labels={"Waiting (min)": "Waiting Time (minutes)", "Algorithm": "Scheduling Algorithm"},
        points="all",
    )
    fig.update_layout(showlegend=False, **DARK_LAYOUT)
    return fig


# ── 5. Completion Rate Comparison ─────────────────────────────────────────

def fig_completion_rate(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing deadline completion rates (%)."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if "Completion Rate (%)" in df.columns:
        summary = df[["Algorithm", "Completion Rate (%)"]].copy()
        summary["Completion Rate (%)"] = _finite_numeric(summary["Completion Rate (%)"]).clip(lower=0, upper=100)
        summary = summary.dropna(subset=["Completion Rate (%)"])
    elif "Completed" in df.columns:
        clean = df[["Algorithm", "Completed"]].copy()
        clean["Completed"] = clean["Completed"].map(lambda value: str(value).strip().lower() in ("true", "1"))
        summary = clean.groupby("Algorithm")["Completed"].mean().mul(100).round(1)
        summary = summary.reindex(_observed_algorithms(df)).reset_index()
        summary.columns = ["Algorithm", "Completion Rate (%)"]
    else:
        fig = go.Figure()
        fig.add_annotation(text="Missing completion columns", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if summary.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid completion data available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    fig = px.bar(
        summary, x="Algorithm", y="Completion Rate (%)",
        color="Algorithm", color_discrete_map=ALGO_COLORS,
        category_orders={"Algorithm": _observed_algorithms(summary)},
        title="Session Completion Rate by Algorithm",
        text="Completion Rate (%)",
        range_y=[0, 115],
        labels={"Completion Rate (%)": "Completion Rate (%)", "Algorithm": "Scheduling Algorithm"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, **DARK_LAYOUT)
    return fig


# ── 6. Unserved Energy Comparison ─────────────────────────────────────────

def fig_missed_energy(df: pd.DataFrame) -> go.Figure:
    """Total unserved energy per algorithm."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if "Missed (kWh)" in df.columns:
        clean = df.copy()
        clean["Missed (kWh)"] = _finite_numeric(clean["Missed (kWh)"]).fillna(0).clip(lower=0)
        summary = clean.groupby("Algorithm")["Missed (kWh)"].sum().round(2)
        summary = summary.reindex(_observed_algorithms(df)).reset_index()
        summary.columns = ["Algorithm", "Unserved Energy (kWh)"]
    elif "Total Missed (kWh)" in df.columns:
        summary = df[["Algorithm", "Total Missed (kWh)"]].rename(columns={"Total Missed (kWh)": "Unserved Energy (kWh)"})
        summary["Unserved Energy (kWh)"] = _finite_numeric(summary["Unserved Energy (kWh)"]).fillna(0).clip(lower=0)
    else:
        fig = go.Figure()
        fig.add_annotation(text="Missing unserved energy columns", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    fig = px.bar(
        summary, x="Algorithm", y="Unserved Energy (kWh)",
        color="Algorithm", color_discrete_map=ALGO_COLORS,
        category_orders={"Algorithm": _observed_algorithms(summary)},
        title="Total Unserved Energy by Algorithm",
        text="Unserved Energy (kWh)",
        labels={"Unserved Energy (kWh)": "Unserved Energy (kWh)", "Algorithm": "Scheduling Algorithm"},
    )
    fig.update_traces(texttemplate="%{text:.2f} kWh", textposition="outside")
    fig.update_layout(showlegend=False, **DARK_LAYOUT)
    return fig


# ── 7. Session Energy Scatter ─────────────────────────────────────────────

def fig_session_scatter(results_df: pd.DataFrame, algo: str = "LAFS") -> go.Figure:
    """Scatter: Required vs Delivered Energy per session."""
    if results_df is None or results_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No session data", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    sub = results_df[results_df["Algorithm"] == algo].copy()
    if sub.empty:
        sub = results_df.copy()
        algo = sub["Algorithm"].iloc[0] if "Algorithm" in sub.columns else "Active"
    sub["Status"] = sub["Completed"].map({True: "100% Fulfilled", False: "Partial Delivery"})

    fig = px.scatter(
        sub, x="Requested (kWh)", y="Delivered (kWh)",
        color="Status", symbol="Status",
        color_discrete_map={"100% Fulfilled": "#22C7A5", "Partial Delivery": "#F07878"},
        hover_data=["Session ID", "Model Variant", "Waiting (min)"],
        title=f"Required vs Delivered Energy per Session — {algo}",
        labels={
            "Requested (kWh)": "Required Energy (kWh)",
            "Delivered (kWh)": "Delivered Energy (kWh)",
            "Status": "Fulfillment Status",
        },
    )
    max_val = max(float(sub["Requested (kWh)"].max() or 10.0), float(sub["Delivered (kWh)"].max() or 10.0)) * 1.08
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", line=dict(dash="dash", color="#82939B", width=1.5),
        name="Target Delivery Boundary",
    ))
    fig.update_layout(**DARK_LAYOUT)
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])
    return fig


# ── 8. Session Timeline Bars ──────────────────────────────────────────────

def fig_session_timeline_bar(results_df: pd.DataFrame) -> go.Figure:
    """Stacked bar: waiting duration and charging duration per session."""
    algorithms = _observed_algorithms(results_df)
    if not algorithms or "Waiting (min)" not in results_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No session timeline results available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    fig = make_subplots(
        rows=1, cols=len(algorithms),
        subplot_titles=[f"Algorithm: {a}" for a in algorithms],
        shared_yaxes=True,
    )
    for col_idx, algo in enumerate(algorithms, start=1):
        sub = results_df[results_df["Algorithm"] == algo].copy()
        sub = sub.sort_values("Arrival (min)").head(16)
        wait = sub["Waiting (min)"]
        started = ~sub["Charge Start (min)"].isna()
        charge_start = sub["Charge Start (min)"]
        charge_end = sub["Charge End (min)"].combine_first(sub["Deadline (min)"])
        charge_dur = (charge_end - charge_start).clip(lower=0).where(started, 0.0).fillna(0)

        fig.add_trace(go.Bar(
            x=sub["Session ID"], y=wait, name="Waiting Time",
            marker_color="#82939B", showlegend=(col_idx == 1),
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            x=sub["Session ID"], y=charge_dur, name="Charging Duration",
            marker_color=ALGO_COLORS[algo], showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

    fig.update_layout(
        barmode="stack",
        title="Session Duration Breakdown: Waiting vs Charging Time",
        yaxis_title="Duration (minutes)",
        height=420,
        **DARK_LAYOUT
    )
    return fig


# ── 9. Radar Summary Chart ────────────────────────────────────────────────

def fig_radar_summary(summary_df: pd.DataFrame) -> go.Figure:
    """Spider/radar chart of normalized algorithm performance."""
    metrics = [
        "Completion Rate",
        "Average Fulfillment",
        "Waiting (relative)",
        "Unserved Energy (relative)",
    ]
    fig = go.Figure()

    if summary_df is None or summary_df.empty:
        fig.add_annotation(text="No benchmark results available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    required = ["Completion Rate (%)", "Avg Completion (%)", "Mean Wait (min)", "Total Missed (kWh)"]
    if any(col not in summary_df.columns for col in required):
        fig.add_annotation(text="Radar omitted because one or more comparison metrics are undefined", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    finite_metrics = summary_df[required].apply(_finite_numeric)
    if finite_metrics.isna().any().any():
        fig.add_annotation(text="Radar omitted because one or more comparison metrics are undefined", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig
    for algo in _observed_algorithms(summary_df):
        row = summary_df[summary_df["Algorithm"] == algo]
        if row.empty:
            continue
        row = row.iloc[0]

        def _rel(col):
            vals = _finite_numeric(summary_df[col]).to_numpy(dtype=float)
            worst, mine = float(vals.max()), float(row[col])
            if worst - float(vals.min()) < 1e-9:
                return 100.0
            return max(0.0, 100.0 * (1.0 - mine / (worst + 1e-6)))
        inv_wait = _rel("Mean Wait (min)")
        inv_missed = _rel("Total Missed (kWh)")

        values = [
            float(row["Completion Rate (%)"]),
            float(row["Avg Completion (%)"]),
            inv_wait,
            inv_missed,
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=metrics + [metrics[0]],
            name=algo,
            line_color=ALGO_COLORS[algo],
            fill="toself",
            opacity=0.25,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], gridcolor="#202D36", linecolor="#2A3943", tickfont=dict(color="#A9B8BE", size=10)),
            angularaxis=dict(gridcolor="#202D36", linecolor="#2A3943", tickfont=dict(color="#E8F0F2", size=11)),
            bgcolor="#121B22",
        ),
        title="Multi-Criteria Performance Overview (Normalized 0–100 Scale)",
        legend_title_text="Algorithm",
        **DARK_LAYOUT
    )
    return fig


# ── 10. Live Station Gantt & Operations Visualization ─────────────────────────

def fig_live_gantt(results_df: pd.DataFrame, algo: str, epoch_start) -> go.Figure:
    """
    Dedicated live station Gantt chart with physical chargers on the Y-axis.
    Allows station operators to visualize charger occupancy, gaps, and EV allocations.
    """
    if results_df is None or results_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No live schedule data available", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    sub = results_df[results_df["Algorithm"] == algo].copy()
    if sub.empty:
        sub = results_df.copy()

    if epoch_start is None or pd.isna(epoch_start):
        epoch_start = pd.Timestamp.now().floor("min")
    else:
        epoch_start = pd.to_datetime(epoch_start)

    rows = []
    for _, r in sub.iterrows():
        start_val = r.get("Charge Start (min)")
        if start_val is None or pd.isna(start_val):
            continue

        start_min = float(start_val)
        end_val = r.get("Charge End (min)")
        if end_val is not None and not pd.isna(end_val):
            end_min = float(end_val)
        else:
            deadline_val = r.get("Deadline (min)")
            end_min = float(deadline_val) if (deadline_val is not None and not pd.isna(deadline_val)) else start_min

        if end_min <= start_min:
            end_min = start_min + 1.0

        t0 = epoch_start + datetime.timedelta(minutes=start_min)
        t1 = epoch_start + datetime.timedelta(minutes=end_min)

        charger = str(r.get("Charger (Assigned)") or r.get("Charger (Dataset)") or "Unassigned")
        reg_no = str(r.get("Vehicle Reg No") or r.get("Session ID") or "EV")
        model = str(r.get("Model Variant") or r.get("Model Name") or "")
        deliv = float(r.get("Delivered (kWh)", 0.0) or 0.0)
        req = float(r.get("Requested (kWh)", 0.0) or 0.0)
        completed = bool(r.get("Completed", False))
        status = "Completed Target" if completed else "Partial / Incomplete"

        rows.append(dict(
            Charger=charger,
            Start=t0,
            Finish=t1,
            Vehicle=f"{reg_no} ({model})",
            RegNo=reg_no,
            Status=status,
            Delivered=f"{deliv:.2f} kWh",
            Requested=f"{req:.2f} kWh",
            Power=f"{r.get('Peak Allocated Power (kW)', 0.0)} kW peak",
        ))

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text="No scheduled charging intervals found", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    plot_df = pd.DataFrame(rows)
    fig = px.timeline(
        plot_df,
        x_start="Start",
        x_end="Finish",
        y="Charger",
        color="Status",
        color_discrete_map={
            "Completed Target": "#22C7A5",
            "Partial / Incomplete": "#F4BE5E",
        },
        hover_data={"Start": "|%H:%M", "Finish": "|%H:%M", "RegNo": True, "Delivered": True, "Power": True, "Charger": True},
        title=f"Live Charger Allocation Timeline ({algo})",
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time of Day (IST)",
        yaxis_title="Physical Charger",
        legend_title_text="Charging Status",
        **DARK_LAYOUT
    )
    return fig


def fig_live_power_profile(sim_results: dict, algo: str, epoch_start, station_cap_kw: float) -> go.Figure:
    """Area chart showing aggregate station load over time for live schedule."""
    if not sim_results or algo not in sim_results:
        fig = go.Figure()
        fig.add_annotation(text="No live power data", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    sim_list = sim_results[algo]
    if not sim_list:
        fig = go.Figure()
        fig.add_annotation(text="No sessions in schedule", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    if epoch_start is None or pd.isna(epoch_start):
        epoch_start = pd.Timestamp.now().floor("min")
    else:
        epoch_start = pd.to_datetime(epoch_start)

    sorted_t, loads = _power_profile(sim_list)
    if not sorted_t:
        fig = go.Figure()
        fig.add_annotation(text="Zero power consumed in schedule", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    times = [epoch_start + datetime.timedelta(minutes=t) for t in sorted_t]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=loads,
        mode="lines",
        line_shape="hv",
        name="Station Demand (kW)",
        line=dict(color="#22C7A5", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(37, 201, 160, 0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=[times[0], times[-1]],
        y=[station_cap_kw, station_cap_kw],
        mode="lines",
        name=f"Power Cap ({station_cap_kw:.0f} kW)",
        line=dict(color="#F07878", width=1.75, dash="dash"),
    ))

    fig.update_layout(
        title=f"Facility Power Profile ({algo})",
        xaxis_title="Time of Day (IST)",
        yaxis_title="Total Power (kW)",
        **DARK_LAYOUT
    )
    return fig


def fig_charger_utilization(util_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart showing physical charger utilization across algorithms."""
    if util_df is None or util_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No charger utilization data", showarrow=False, font_size=13, font_color="#82939B")
        fig.update_layout(**DARK_LAYOUT)
        return fig

    fig = px.bar(
        util_df,
        x="Charger ID",
        y="Utilization (%)",
        color="Algorithm",
        barmode="group",
        color_discrete_map=ALGO_COLORS,
        title="Physical Charger Utilization by Algorithm",
        labels={"Utilization (%)": "Duty Cycle Utilization (%)", "Charger ID": "Dispenser Endpoint"},
        text_auto=".1f",
    )
    fig.update_layout(
        yaxis_range=[0, max(util_df["Utilization (%)"].max() * 1.15, 10.0)],
        **DARK_LAYOUT
    )
    return fig


# Backward-compatibility aliases
fig_power_profile = fig_live_power_profile
fig_live_timeline = fig_live_gantt
fig_radar_comparison = fig_radar_summary
