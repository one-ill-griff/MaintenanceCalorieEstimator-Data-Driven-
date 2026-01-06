import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, List

st.set_page_config(page_title="Maintenance Calories Estimator", layout="wide")
st.title("Maintenance Calories Estimator (Data-Driven)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
default_path = st.text_input("...or load from path", value="data/sample_nutrition_training_log_90d.csv")

@st.cache_data
def load_df(file, path):
    if file is not None:
        return pd.read_csv(file)
    return pd.read_csv(path)

df = load_df(uploaded, default_path)

# -----------------------
# Normalize + standardize schema
# -----------------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

# Map your real sheet columns -> standard names used by the app
rename_map = {
    "calories_consumed": "calories",
    "morning_weight_(lbs)": "weight_lb",
    "protein_(g)": "protein_g",
    "carbs_(g)": "carbs_g",
    "fat_(g)": "fat_g",
    "sodium_(mg)": "sodium_mg",
    "water_(oz)": "water_oz",
    "sleep_(hrs)": "sleep_hrs",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

with st.expander("Detected columns (after normalization/renaming)"):
    st.write(list(df.columns))

# Date handling (supports common variants)
date_candidates = ["date", "day", "timestamp"]
date_col = next((c for c in date_candidates if c in df.columns), None)

if date_col is None:
    st.error("No date column found. Expected one of: date, day, timestamp")
    st.stop()

df["date"] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

# Optional: convert water oz -> liters
if "water_oz" in df.columns and "water_l" not in df.columns:
    df["water_l"] = pd.to_numeric(df["water_oz"], errors="coerce") * 0.0295735

# Validate required fields
required = ["date", "calories", "weight_lb"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns after renaming: {missing}")
    st.stop()

# Coerce numeric columns (handles blanks/strings)
for c in ["calories", "weight_lb"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["calories", "weight_lb"]).copy()

# Proof that full dataset is loaded (not just preview)
st.caption(
    f"Loaded **{len(df):,}** rows after cleaning "
    f"(date range: {df['date'].min().date()} → {df['date'].max().date()})."
)

# -----------------------
# Preview (expandable)
# -----------------------
st.subheader("Preview")
preview_rows = st.slider("Rows to preview", 10, min(200, len(df)), min(50, len(df)))
st.dataframe(df.head(preview_rows), use_container_width=True, height=600)

if st.checkbox("Show full table (may be slow for huge files)"):
    st.dataframe(df, use_container_width=True, height=600)

# -----------------------
# Smoothing for weight trend
# -----------------------
window = st.slider("Weight smoothing window (days)", 3, 21, 7)

# Keep trend usable on sparse/short data
df["weight_trend"] = df["weight_lb"].rolling(window, min_periods=2).mean()
df_trend = df.dropna(subset=["weight_trend"]).copy()

st.caption(f"Rows usable for trend/maintenance after smoothing: **{len(df_trend):,}**")

# -----------------------
# Maintenance estimation helpers
# -----------------------
def estimate_tdee_from_recent(recent_df: pd.DataFrame, kcal_per_lb: float = 3500.0) -> dict:
    """Estimate TDEE from a recent window using linear slope of smoothed weight trend."""
    recent_df = recent_df.sort_values("date")
    t = (recent_df["date"] - recent_df["date"].iloc[0]).dt.days.astype(float).to_numpy()
    w = recent_df["weight_trend"].to_numpy()

    # Need at least 2 distinct days to fit slope
    if len(recent_df) < 2 or np.nanmax(t) == 0:
        return {"tdee": np.nan, "avg_intake": np.nan, "slope_lb_per_day": np.nan}

    slope = np.polyfit(t, w, 1)[0]  # lb/day
    avg_intake = recent_df["calories"].mean()
    tdee = avg_intake - kcal_per_lb * slope
    return {"tdee": float(tdee), "avg_intake": float(avg_intake), "slope_lb_per_day": float(slope)}

def bootstrap_tdee(df_window: pd.DataFrame, n_boot: int = 400, seed: int = 42) -> np.ndarray:
    """Bootstrap TDEE by resampling days WITH replacement inside the window."""
    rng = np.random.default_rng(seed)
    vals = []
    idx = df_window.index.to_numpy()

    # Too few points => bootstrap is meaningless
    if len(idx) < 4:
        return np.array([], dtype=float)

    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        sample = df_window.loc[sample_idx].sort_values("date")
        est = estimate_tdee_from_recent(sample)["tdee"]
        if np.isfinite(est):
            vals.append(est)
    return np.array(vals, dtype=float)

# -----------------------
# Confidence scoring (simple + honest)
# -----------------------
total_days = len(df)
trend_days = len(df_trend)

missing_weight = df["weight_lb"].isna().sum()
missing_cal = df["calories"].isna().sum()

days_score = min(trend_days / 28, 1.0)  # 28 days ~ good baseline

w_std = df["weight_lb"].dropna().std() if df["weight_lb"].notna().sum() >= 3 else np.nan
noise_score = 0.0
if np.isfinite(w_std):
    # <=0.5 lb std is good, >=1.5 lb std is noisy
    noise_score = float(np.clip((1.5 - w_std) / (1.5 - 0.5), 0, 1))

completeness = 1.0 - (missing_weight + missing_cal) / max(total_days * 2, 1)

confidence = 100 * (0.65 * days_score + 0.20 * completeness + 0.15 * noise_score)
confidence = float(np.clip(confidence, 0, 100))

def confidence_label(c: float) -> str:
    if c < 25: return "Very Low"
    if c < 50: return "Low"
    if c < 75: return "Moderate"
    return "High"

st.subheader("Data Quality / Confidence")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Usable days (after smoothing)", trend_days)
c2.metric("Missing weigh-ins", int(missing_weight))
c3.metric("Missing calorie days", int(missing_cal))
c4.metric("Confidence", f"{confidence:.0f}/100 ({confidence_label(confidence)})")

if trend_days < 14:
    st.warning(
        "Very short dataset. Any maintenance estimate will be highly unreliable due to water/sodium/glycogen noise. "
        "Aim for at least 14 days (better: 21–28 days) of daily weigh-ins + calorie entries."
    )

# -----------------------
# Choose window length safely
# -----------------------
available_days = len(df_trend)

if available_days < 2:
    st.error("Not enough distinct weigh-in days to estimate maintenance. Add at least 2 days of weight + calories.")
    st.stop()

if available_days >= 14:
    max_days = min(180, available_days)
    default_days = min(28, max_days)
    N = st.slider("Days to estimate maintenance from", 14, max_days, default_days)
else:
    N = available_days
    st.info(f"Only {available_days} usable days after smoothing — using all available days for estimation.")

recent = df_trend.tail(N).copy()
st.caption(
    f"Maintenance window: last **{N}** usable days "
    f"({recent['date'].min().date()} → {recent['date'].max().date()})."
)

est = estimate_tdee_from_recent(recent)
tdee = est["tdee"]
avg_intake = est["avg_intake"]
slope = est["slope_lb_per_day"]

# Bootstrap confidence interval
boot = np.array([], dtype=float)
ci_low, ci_high = (np.nan, np.nan)

if available_days >= 7:
    n_boot = st.slider("Bootstrap samples (for confidence range)", 100, 1000, 400, step=50)
    boot = bootstrap_tdee(recent, n_boot=n_boot)

if len(boot) >= 30:
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
else:
    # Fallback range when bootstrapping isn't meaningful (short data)
    # Wider when confidence is low; tighter as confidence increases
    spread = 800 - 6 * confidence  # confidence 0 => 800, confidence 100 => 200
    spread = float(np.clip(spread, 200, 900))
    ci_low, ci_high = tdee - spread / 2, tdee + spread / 2

# -----------------------
# Metrics
# -----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg intake (kcal/day)", f"{avg_intake:,.0f}")
col2.metric("Weight trend slope (lb/day)", f"{slope:+.4f}" if np.isfinite(slope) else "N/A")
col3.metric("Estimated maintenance (kcal/day)", f"{tdee:,.0f}" if np.isfinite(tdee) else "N/A")
col4.metric("Likely range", f"{ci_low:,.0f} – {ci_high:,.0f}" if np.isfinite(ci_low) else "N/A")

# -----------------------
# Plots
# -----------------------
st.subheader("Weight and Trend")
fig1 = plt.figure()
plt.plot(df["date"], df["weight_lb"], label="Daily weight")
plt.plot(df["date"], df["weight_trend"], label="Trend")
plt.xlabel("Date")
plt.ylabel("Weight (lb)")
plt.legend()
st.pyplot(fig1)

# -----------------------
# Nutrition & Recovery Signals (Analytics + Insights)
# -----------------------
st.subheader("Nutrition & Recovery Signals (Analytics + Insights)")

def _has(col: str, min_n: int = 3) -> bool:
    return col in df.columns and df[col].notna().sum() >= min_n

def _as_numeric(col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Coerce optional numeric columns if present
optional_cols = [
    "protein_g", "carbs_g", "fat_g",
    "sodium_mg", "water_l", "water_oz",
    "steps", "sleep_hrs",
    "calories_burned", "net_calories"
]
for c in optional_cols:
    _as_numeric(c)

# If only water_oz exists, create water_l (safe)
if "water_oz" in df.columns and "water_l" not in df.columns:
    df["water_l"] = df["water_oz"] * 0.0295735

pretty = {
    "protein_g": "Protein (g)",
    "carbs_g": "Carbs (g)",
    "fat_g": "Fat (g)",
    "sodium_mg": "Sodium (mg)",
    "water_l": "Water (L)",
    "steps": "Steps",
    "sleep_hrs": "Sleep (hrs)",
    "calories_burned": "Calories burned (kcal)",
    "net_calories": "Net calories (kcal)",
}

metric_options = [k for k in pretty.keys() if _has(k)]
if not metric_options:
    st.info("No optional metrics detected (water, sodium, macros, steps, sleep, etc.).")
else:
    selected = st.multiselect(
        "Select metrics to analyze/plot",
        options=metric_options,
        default=metric_options[: min(4, len(metric_options))]
    )

    rows = []
    for k in selected:
        s = df[k].dropna()
        rows.append({
            "Metric": pretty.get(k, k),
            "Days available": int(s.shape[0]),
            "Mean": float(s.mean()),
            "Std dev": float(s.std()) if s.shape[0] >= 2 else np.nan,
            "Min": float(s.min()),
            "Max": float(s.max()),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption("Charts use all available rows; missing days are skipped.")
    for k in selected:
        figm = plt.figure()
        plt.plot(df["date"], df[k])
        plt.xlabel("Date")
        plt.ylabel(pretty.get(k, k))
        st.pyplot(figm)

# -----------------------
# Scale noise analytics: correlate yesterday’s signals with today’s weight change
# -----------------------
st.subheader("Scale Noise: Signals vs Weight Changes")

df["delta_weight_lb"] = df["weight_lb"].diff()
smooth_delta = st.checkbox("Smooth day-to-day weight change (3-day MA)", value=True)
delta_col = "delta_weight_lb_s"
if smooth_delta:
    df[delta_col] = df["delta_weight_lb"].rolling(3, min_periods=2).mean()
else:
    df[delta_col] = df["delta_weight_lb"]

signal_candidates = [c for c in ["sodium_mg", "carbs_g", "water_l"] if _has(c, min_n=4)]

corr_rows = []
for sig in signal_candidates:
    tmp = df[["date", sig, delta_col]].copy()
    tmp["sig_lag1"] = tmp[sig].shift(1)
    tmp = tmp.dropna(subset=["sig_lag1", delta_col])
    r = np.nan
    if len(tmp) >= 6:
        r = np.corrcoef(tmp["sig_lag1"], tmp[delta_col])[0, 1]
    corr_rows.append({
        "Signal (yesterday)": pretty.get(sig, sig),
        "Correlation with today's weight change": float(r) if np.isfinite(r) else np.nan,
        "Samples": int(len(tmp))
    })

if not signal_candidates:
    st.info("Need at least ~4 days of sodium/carbs/water data to run correlation-based scale-noise analysis.")
else:
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)

    sig_choice = st.selectbox(
        "Plot a signal (yesterday) vs today's weight change",
        options=signal_candidates,
        format_func=lambda x: pretty.get(x, x)
    )

    tmp = df[["date", sig_choice, delta_col]].copy()
    tmp["signal_lag1"] = tmp[sig_choice].shift(1)
    tmp = tmp.dropna(subset=["signal_lag1", delta_col])

    if len(tmp) < 6:
        st.info("Not enough data points for a meaningful plot yet.")
    else:
        fig_scatter = plt.figure()
        plt.scatter(tmp["signal_lag1"], tmp[delta_col])
        plt.xlabel(f"{pretty.get(sig_choice, sig_choice)} (yesterday)")
        plt.ylabel("Weight change (lb) today")
        st.pyplot(fig_scatter)

        st.caption(
            "Interpretation: positive correlation means higher values yesterday tend to coincide with higher scale weight today "
            "(often water retention). With short datasets this will be noisy."
        )

# -----------------------
# Actionable Insights (rule-based, transparent)
# -----------------------
st.subheader("Actionable Insights")

with st.expander("Set targets (optional)"):
    protein_target = st.number_input("Protein target (g/day)", min_value=0, value=160, step=5)
    water_target_l = st.number_input("Water target (L/day)", min_value=0.0, value=3.0, step=0.1)
    sodium_soft_cap = st.number_input("Sodium soft cap (mg/day)", min_value=0, value=3000, step=100)
    steps_target = st.number_input("Steps target (per day)", min_value=0, value=8000, step=500)
    sleep_target = st.number_input("Sleep target (hrs/night)", min_value=0.0, value=8.0, step=0.25)

insights: List[tuple] = []

def cv(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 3:
        return np.nan
    m = s.mean()
    if m == 0:
        return np.nan
    return float(s.std() / abs(m))

# Protein insights
if _has("protein_g", min_n=3):
    p = df["protein_g"].dropna()
    p_mean = float(p.mean())
    p_cv = cv(p)
    if p_mean < protein_target * 0.85:
        insights.append(("Protein", f"Your average protein is **{p_mean:.0f}g/day**, below your {protein_target}g target. Raise it by ~20–40g/day (lean meats, Greek yogurt, whey, egg whites)."))
    else:
        insights.append(("Protein", f"Protein looks solid: **{p_mean:.0f}g/day** vs target {protein_target}g. Keep consistency."))
    if np.isfinite(p_cv) and p_cv > 0.25:
        insights.append(("Protein", "Protein intake is pretty inconsistent day-to-day. Try setting a minimum floor (e.g., hit 120g no matter what)."))

# Water insights
if _has("water_l", min_n=3):
    w = df["water_l"].dropna()
    w_mean = float(w.mean())
    if w_mean < water_target_l * 0.8:
        insights.append(("Hydration", f"Average water is **{w_mean:.2f} L/day**, below your {water_target_l:.1f} L target. Increase by ~0.5–1.0 L and keep it consistent to reduce scale noise."))
    else:
        insights.append(("Hydration", f"Hydration looks fine: **{w_mean:.2f} L/day** on average."))
    w_cv = cv(w)
    if np.isfinite(w_cv) and w_cv > 0.25:
        insights.append(("Hydration", "Water intake swings a lot day-to-day. Consistency matters more than pushing extremes."))

# Sodium insights
if _has("sodium_mg", min_n=3):
    s = df["sodium_mg"].dropna()
    s_mean = float(s.mean())
    s_cv = cv(s)
    high_days = int((s > sodium_soft_cap).sum())
    if high_days >= 2:
        insights.append(("Sodium", f"You have **{high_days}** day(s) above {sodium_soft_cap}mg sodium. If scale spikes, this is a prime suspect. Try spreading salty meals out and keeping sodium steadier."))
    if s_mean > sodium_soft_cap:
        insights.append(("Sodium", f"Average sodium is **{s_mean:.0f}mg/day** (above {sodium_soft_cap}mg). If recomp/cut stalls, try tightening sodium consistency before changing calories."))
    if np.isfinite(s_cv) and s_cv > 0.25:
        insights.append(("Sodium", "Sodium is highly variable. Big swings can cause misleading weight fluctuations."))

# Steps and sleep insights
if _has("steps", min_n=3):
    stp = df["steps"].dropna()
    stp_mean = float(stp.mean())
    if stp_mean < steps_target * 0.8:
        insights.append(("Activity", f"Steps average **{stp_mean:,.0f}/day**, below your {steps_target:,}/day target. For recomp, bring this up gradually (+1k/day this week)."))
    else:
        insights.append(("Activity", f"Steps look good: **{stp_mean:,.0f}/day** on average."))

if _has("sleep_hrs", min_n=3):
    sl = df["sleep_hrs"].dropna()
    sl_mean = float(sl.mean())
    if sl_mean < sleep_target * 0.85:
        insights.append(("Recovery", f"Sleep averages **{sl_mean:.1f} hrs/night**, below your {sleep_target:.1f} target. Poor sleep can raise hunger and blur weight signals—prioritize sleep before cutting harder."))
    else:
        insights.append(("Recovery", f"Sleep looks decent: **{sl_mean:.1f} hrs/night** on average."))

# Macros sanity check (if all macros exist)
if _has("protein_g", 3) and _has("carbs_g", 3) and _has("fat_g", 3):
    tmp = df[["protein_g", "carbs_g", "fat_g", "calories"]].dropna()
    if len(tmp) >= 3:
        est_cals = tmp["protein_g"] * 4 + tmp["carbs_g"] * 4 + tmp["fat_g"] * 9
        diff = (tmp["calories"] - est_cals).abs().mean()
        if diff > 250:
            insights.append(("Logging", f"Macro-derived calories differ from logged calories by ~**{diff:.0f} kcal/day** on average. You may be missing items or using inconsistent entries."))

# Water retention pattern detection
if len(signal_candidates) >= 1 and df["delta_weight_lb"].notna().sum() >= 3:
    jump_thresh = st.slider("Scale jump threshold (lb/day)", 0.5, 3.0, 1.0, step=0.1)

    jumps = df[df["delta_weight_lb"] >= jump_thresh].copy()
    if len(jumps) >= 1:
        for sig in signal_candidates:
            med = df[sig].median()
            jumps[f"{sig}_yesterday"] = df[sig].shift(1).loc[jumps.index]
            jumps[f"{sig}_yesterday_high"] = jumps[f"{sig}_yesterday"] > med

        any_high = np.zeros(len(jumps), dtype=bool)
        for sig in signal_candidates:
            any_high |= jumps[f"{sig}_yesterday_high"].fillna(False).to_numpy()

        n_explained = int(any_high.sum())
        if n_explained >= 1:
            insights.append((
                "Scale noise",
                f"On **{n_explained}/{len(jumps)}** scale-jump day(s) (≥{jump_thresh:.1f} lb), yesterday’s sodium/carbs/water were above your typical level. "
                "That pattern often indicates **water retention**, not true fat gain."
            ))

if len(df) < 14:
    insights.append(("Confidence", "You have <14 days of data. Treat all estimates as a starting point. The app gets meaningfully more accurate after ~21–28 days of consistent weigh-ins + calories."))

if not insights:
    st.info("No insights to report yet (either missing metrics or too little data).")
else:
    for area, msg in insights:
        st.write(f"**{area}:** {msg}")

# -----------------------
# Rolling maintenance (only if enough data)
# -----------------------
st.subheader("Rolling Maintenance (TDEE over time)")

if len(df_trend) < 14:
    st.info("Need at least 14 usable days to show a rolling maintenance line.")
else:
    roll_window = st.slider(
        "Rolling window (days)",
        14,
        min(60, len(df_trend)),
        min(21, len(df_trend))
    )

    rolling_vals = []
    rolling_dates = []

    for i in range(roll_window, len(df_trend) + 1):
        win = df_trend.iloc[i - roll_window:i]
        e = estimate_tdee_from_recent(win)["tdee"]
        rolling_vals.append(e)
        rolling_dates.append(win["date"].iloc[-1])

    roll_df = pd.DataFrame({"date": rolling_dates, "tdee": rolling_vals}).dropna()

    if len(roll_df) < 5:
        st.info("Not enough data to display a stable rolling maintenance line.")
    else:
        fig2 = plt.figure()
        plt.plot(roll_df["date"], roll_df["tdee"])
        plt.xlabel("Date")
        plt.ylabel("Estimated Maintenance (kcal/day)")
        st.pyplot(fig2)

# -----------------------
# Goal Planner (works even in low-data mode)
# -----------------------
st.subheader("Goal Planner")
goal_weight = st.number_input("Target weight (lb)", value=float(df["weight_lb"].iloc[-1]))
goal_date = st.date_input("Target date")

# Use trend if available; else fallback to last weight
if len(df_trend) >= 1:
    current_weight = float(df_trend["weight_trend"].iloc[-1])
else:
    current_weight = float(df["weight_lb"].iloc[-1])

days_to_goal = (pd.to_datetime(goal_date) - pd.to_datetime(df["date"].max().date())).days
if days_to_goal <= 0:
    st.info("Pick a future date for the goal.")
else:
    delta_lb = goal_weight - current_weight
    rate_lb_per_day = delta_lb / days_to_goal
    rate_lb_per_week = rate_lb_per_day * 7
    kcal_per_day = 3500 * rate_lb_per_day

    base_maint = tdee if np.isfinite(tdee) else avg_intake
    target_intake = base_maint + kcal_per_day

    st.write(f"Current (trend) weight: **{current_weight:.1f} lb**")
    st.write(f"Change needed: **{delta_lb:+.1f} lb** over **{days_to_goal} days**")
    st.write(f"Required rate: **{rate_lb_per_week:+.2f} lb/week**")
    st.write(f"Implied surplus/deficit: **{kcal_per_day:+.0f} kcal/day**")
    st.success(f"Suggested intake target: **{target_intake:,.0f} kcal/day**")

    # Guardrails
    bw = max(current_weight, 1.0)
    pct_per_week = abs(rate_lb_per_week) / bw * 100
    if pct_per_week > 1.0:
        st.warning("This target rate is aggressive (>~1% bodyweight/week). Consider extending the timeline.")
    elif pct_per_week > 0.5:
        st.info("This target rate is moderate (~0.5–1% bodyweight/week).")

    if confidence < 25:
        st.warning("Because confidence is very low, treat the suggested intake as a rough starting point and adjust as you collect more data.")
