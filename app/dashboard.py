"""ECHO dashboard.  Run from the repo root:  streamlit run app/dashboard.py

Reads the files the pipeline already wrote (results/<subset>/conformal_windows.csv and
results/<subset>/decision_model_final.json). All numbers come from src/decision.py, src/dashboard_logic.py
and (for uploads) src/upload_logic.py. Uploading also needs PyTorch, the saved preprocessing
(data/preproc_<subset>.joblib) and the model checkpoint used for calibration; without them the page still works
on the NASA test engines and says what is missing.
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dashboard_logic import engine_frame, holds_up, latest_table, load_bundle, most_urgent_unit  # noqa: E402
from src.decision import (  # noqa: E402
    DEFAULT_BAND, H_MAX, H_MIN, VALIDATED_HORIZONS, dollar_summary, horizon_is_validated)
from src.upload_logic import (  # noqa: E402
    RAW_COLUMNS, load_runtime, parse_upload, runtime_status, score_upload, settings_warning, upload_bundle)

st.set_page_config(page_title="ECHO", layout="wide")

BLUE, GREY, RED, GREEN = "#2D5BA3", "#8A8A85", "#B3261E", "#1E7B4B"
ADVICE_STYLE = {  # background, text colour
    "WAIT": ("#E4E4E0", "#33332F"), "INVESTIGATE": ("#FBEFD2", "#7A4E00"), "ACT": ("#DCE7F7", "#1F4E8C")}
STAGE_TEXT = {"early": "Early: over 100 cycles left", "mid": "Mid: 30 to 100 cycles left", "late": "Late: under 30 cycles left"}
NASA, UPLOADED = "NASA test engines", "My uploaded file"


@st.cache_resource(show_spinner="Loading results...")
def get_bundle(subset: str):
    return load_bundle(ROOT, subset)


@st.cache_resource(show_spinner="Loading the model...")
def get_runtime(subset: str):
    return load_runtime(ROOT, subset)


def set_horizon(value: int):
    st.session_state["horizon"] = value


# ------------------------------------------------------------------------------------------------ header
st.markdown("## ECHO")
st.caption("Should this engine be serviced now, or can it wait? Predicted remaining life, how sure the model is, "
           "and the maintenance decision that follows.")

with st.expander("How to read this page", expanded=False):
    st.markdown(
        "1. **Pick an engine** and a point in its life (left), or upload your own engines at the bottom.\n"
        "2. **Set how much worse a failure is** with one slider, or add dollar costs to see the advice in money.\n"
        "3. **Read the RUL and the 90% interval together.** The number is the best guess; the interval says how far to trust it.\n"
        "4. **Read the recommendation:** wait, act, or investigate when the call is too close for the model to settle.\n"
        "5. **Check the strip at the bottom** to see how far the numbers can be trusted.")

# ------------------------------------------------------------------------------------------------ sidebar
if st.session_state.pop("goto_upload", False):  # set after a successful upload; must happen before the radio exists
    st.session_state["source"] = UPLOADED

with st.sidebar:
    subset = st.radio("Dataset", ["FD002", "FD001"], horizontal=True)
    ref_bundle = get_bundle(subset)
    if ref_bundle is None:
        st.error(f"No results for {subset} yet. Run the conformal and decision-layer scripts first "
                 f"(results/{subset}/conformal_windows.csv and decision_model_final.json).")
        st.stop()
    saved = st.session_state.get("upload", {}).get(subset)

    source = st.radio("Data source", [NASA, UPLOADED], key="source")
    active = ref_bundle if source == NASA else (saved["bundle"] if saved else None)

    st.markdown("**1. Engine and cycle**")
    if active is None:
        st.info("Upload a file at the bottom of the page first.")
    else:
        default_unit = most_urgent_unit(active)
        unit = st.selectbox("Engine", active["units"], index=active["units"].index(default_unit),
                            label_visibility="collapsed", key=f"engine_{subset}_{source}")
        unit_cycles = active["windows"].loc[active["windows"]["unit"] == unit, "cycle"]
        cycle = st.slider("Cycle", int(unit_cycles.min()), int(unit_cycles.max()), int(unit_cycles.max()),
                          help="A cycle is one flight. The page shows what the model would have said at that "
                               "moment, using only the data up to that cycle.", key=f"cycle_{subset}_{source}_{unit}")

    st.markdown("**2. How much worse is a failure?**")
    ratio_slider = st.slider("Cost ratio (failure cost / planned service cost)", 1, 100, 20)
    with st.expander("Add dollar costs (optional)"):
        st.caption("Enter one or both. With one, the other comes from the ratio slider.")
        planned = st.number_input("Planned maintenance, Cs ($)", min_value=0.0, value=0.0, step=500.0)
        failure_cost = st.number_input("Unplanned failure, Cu ($)", min_value=0.0, value=0.0, step=5000.0)
    if planned > 0 and failure_cost > 0:
        ratio, cs_usd, cu_usd, dollar_note = failure_cost / planned, planned, failure_cost, "set from both dollar amounts"
    elif planned > 0:
        ratio = float(ratio_slider)
        cs_usd, cu_usd, dollar_note = planned, ratio * planned, f"failure cost taken as {ratio:.0f} x your planned cost"
    elif failure_cost > 0:
        ratio = float(ratio_slider)
        cs_usd, cu_usd, dollar_note = failure_cost / ratio, failure_cost, f"planned cost taken as failure cost / {ratio:.0f}"
    else:
        ratio, cs_usd, cu_usd, dollar_note = float(ratio_slider), 0.0, 0.0, ""
    dollars = cs_usd > 0 and cu_usd > 0
    st.info(f"Cost ratio = **{ratio:.1f}**: a failure costs {ratio:.0f} times a planned service."
            + (f" ({dollar_note})" if dollar_note else ""))

    st.markdown("**3. Look-ahead window H (cycles)**")
    st.session_state.setdefault("horizon", 30)
    horizon = st.number_input("H", min_value=int(H_MIN), max_value=int(H_MAX), step=1, key="horizon",
                              label_visibility="collapsed")
    quick = st.columns(len(VALIDATED_HORIZONS))
    for col, h in zip(quick, VALIDATED_HORIZONS):
        col.button(str(h), key=f"h{h}", on_click=set_horizon, args=(h,), width="stretch")
    validated = horizon_is_validated(horizon)
    st.caption(("Validated: cost results were tested at this H." if validated else
                "Not validated: cost results were only tested at 15, 30 and 50.") +
               ' "Fails within H" means H or fewer cycles left.')

    with st.expander("Advanced"):
        band = st.slider("Investigate band", 1.0, 5.0, float(DEFAULT_BAND), 0.25,
                         help="Advice is INVESTIGATE when p x ratio is between 1/band and band.")


# ------------------------------------------------------------------------------------------------ main view
def render_main():
    is_upload = bool(active.get("uploaded"))
    frame = engine_frame(active, unit, float(horizon), ratio, band)
    row = frame[frame["cycle"] == cycle].iloc[0]
    check = holds_up(ref_bundle, row["stage"], float(horizon), ratio, band)  # always checked on the NASA held-out half
    has_truth = bool(frame["true_capped"].notna().any())
    if is_upload:
        st.info(f"Showing uploaded engine {unit} from {saved['name']}. Scored with the {subset} model.")

    # ---- card 1: the estimate and how sure it is
    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            st.caption("Predicted RUL (best guess)")
            st.markdown(f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>{row['yhat']:.0f}</div>",
                        unsafe_allow_html=True)
            st.caption("cycles left")
        with right:
            st.caption("90% conformal interval (how sure the model is)")
            st.markdown(f"<div style='font-size:3rem;font-weight:700;line-height:1.1;color:{BLUE}'>"
                        f"{row['lo']:.0f} to {row['hi']:.0f}</div>", unsafe_allow_html=True)
            st.caption(f"Width {row['hi'] - row['lo']:.0f} cycles. The true RUL should fall inside about 9 times in 10.")
        bar = pd.DataFrame({"lo": [row["lo"]], "hi": [row["hi"]], "yhat": [row["yhat"]], "h": [float(horizon)]})
        domain = [0, max(130, float(row["hi"]) + 5)]
        base_bar = alt.Chart(bar)
        rng = base_bar.mark_rule(size=14, color=BLUE, opacity=0.35).encode(
            x=alt.X("lo:Q", scale=alt.Scale(domain=domain), title="cycles left"), x2="hi:Q")
        dot = base_bar.mark_circle(size=140, color="#1F1F1D").encode(x=alt.X("yhat:Q", scale=alt.Scale(domain=domain)))
        hline = base_bar.mark_tick(color=RED, size=36, thickness=3).encode(
            x=alt.X("h:Q", scale=alt.Scale(domain=domain)))
        st.altair_chart((rng + dot + hline).properties(height=70), width="stretch")
        st.caption("Dot: best guess. Band: 90% interval. Red mark: H. If the band reaches the red mark, "
                   "the engine may fail within the window.")

    # ---- card 2: the recommendation
    with st.container(border=True):
        st.markdown(f"**Recommendation for engine {unit}, cycle {cycle}**")
        strip = "".join(
            f"<span style='flex:1;padding:14px;text-align:center;font-weight:700;font-size:1.3rem;"
            f"background:{ADVICE_STYLE[a][0]};color:{ADVICE_STYLE[a][1]};"
            + ("outline:3px solid " + ADVICE_STYLE[a][1] + ";outline-offset:-3px;" if a == row["advice"] else "opacity:0.55;")
            + f"'>{'ACT NOW' if a == 'ACT' else a}</span>" for a in ("WAIT", "INVESTIGATE", "ACT"))
        st.markdown(f"<div style='display:flex;border-radius:8px;overflow:hidden'>{strip}</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"### p {row['p']:.3f} &times; ratio {ratio:.1f} = {row['score']:.2f}")
            st.markdown(f"Below {1 / band:.2f}: wait. Between {1 / band:.2f} and {band:.2f}: investigate, because "
                        f"acting and waiting cost about the same. Above {band:.2f}: act. Break-even ratio "
                        f"T = 1 / p = {1 / row['p']:.1f}.")
            if check.get("available"):
                st.caption(f"At this ratio and H, {100 * check['investigate_share']:.1f}% of held-out windows land in "
                           "the investigate band.")
        with c2:
            st.info("**In plain words:** p is the chance the engine fails within H cycles. Multiply it by how much "
                    "worse a failure is. Far above 1, servicing is cheaper on average. Far below 1, waiting is. Near 1 "
                    "the model cannot settle it for you, so look at the interval and p and decide by your priorities.")
        if dollars:
            d = dollar_summary(row["p"], cs_usd, cu_usd)
            saver = "servicing now" if d["cheaper"] == "act" else "waiting"
            st.markdown(f"**In dollars:** waiting carries an expected cost of **${d['wait_expected']:,.0f}** (p x Cu) "
                        f"against **${d['act_cost']:,.0f}** to service now (Cs). On average, {saver} is cheaper by "
                        f"**${d['difference']:,.0f}**.")
            if dollar_note and "both" not in dollar_note:
                st.caption(f"Dollar amounts: {dollar_note}.")
        if not row["reliable"]:
            st.warning("p is less reliable here: very few earlier windows had a similar predicted RUL"
                       + (f" (only {int(row['n_nearby'])})." if not np.isnan(row["n_nearby"]) else "."))

    # ---- card row 3: three tiles, each in its own bubble
    if bool(row["echo_act"]) == bool(row["point_act"]):
        verdict = "Same advice"
    else:
        verdict = "ECHO acts earlier" if row["echo_act"] else "ECHO waits longer"
    t1, t2, t3 = st.columns(3)
    with t1.container(border=True):
        st.metric("Stage", str(row["stage"]).capitalize(), help=STAGE_TEXT[row["stage"]] +
                  ". The interval is calibrated separately for each stage.")
    with t2.container(border=True):
        st.metric("Chance of failing within H (p)", f"{row['p']:.3f}",
                  help="Taken from the model's past errors on similar windows.")
    with t3.container(border=True):
        st.metric("ECHO vs the point rule", verdict,
                  help="The point rule acts when the predicted RUL is at or below H and ignores uncertainty and "
                       "cost. A difference between the two is a decision flip.")

    # ---- card row 4: the two charts
    chart_l, chart_r = st.columns([3, 2])
    with chart_l.container(border=True):
        st.markdown("**Cycles left, over the engine's life**")
        base = alt.Chart(frame).encode(x=alt.X("cycle:Q", title="cycle"))
        layers = [base.mark_area(opacity=0.25, color=BLUE).encode(y=alt.Y("lo:Q", title="cycles left"), y2="hi:Q")]
        if has_truth:
            layers.append(base.mark_line(color=GREEN, strokeWidth=2.5, strokeDash=[6, 4]).encode(y="true_capped:Q"))
        layers.append(base.mark_line(color="#1F1F1D").encode(y="yhat:Q"))
        layers.append(alt.Chart(pd.DataFrame({"h": [float(horizon)]})).mark_rule(color=RED, strokeDash=[3, 3]).encode(y="h:Q"))
        layers.append(alt.Chart(pd.DataFrame({"c": [cycle]})).mark_rule(color=BLUE, size=2).encode(x="c:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")
        st.caption("Black: model's guess. Blue band: 90% range. "
                   + ("Green dashed: the real remaining life (NASA test files, capped at 125). " if has_truth else
                      "Uploaded engines have no true RUL to compare with. ")
                   + "Red dashed: H. Blue vertical: the cycle you picked.")
    with chart_r.container(border=True):
        st.markdown("**Chance of failing within H**")
        act_line, wait_line = min(band / ratio, 1.0), min(1 / (ratio * band), 1.0)
        top = min(1.0, max(float(frame["p"].max()), act_line) * 1.25)
        pbase = alt.Chart(frame).encode(x=alt.X("cycle:Q", title="cycle"))
        pline = pbase.mark_line(color="#1F1F1D").encode(y=alt.Y("p:Q", title="p", scale=alt.Scale(domain=[0, top])))
        thresholds = pd.DataFrame({"t": [wait_line, act_line], "kind": ["wait below", "act above"]})
        trules = alt.Chart(thresholds).mark_rule(strokeDash=[3, 3], size=2).encode(
            y="t:Q", color=alt.Color("kind:N", scale=alt.Scale(domain=["wait below", "act above"], range=[GREY, BLUE]),
                                     legend=alt.Legend(title=None, orient="bottom")))
        pnow = alt.Chart(frame[frame["cycle"] == cycle]).mark_circle(size=110, color=BLUE).encode(x="cycle:Q", y="p:Q")
        st.altair_chart((pline + trules + pnow).properties(height=280), width="stretch")
        st.caption("Dot: the cycle you picked. Above the blue line: act. Below the grey line: wait. Between: "
                   "investigate. The axis stops just above the act line, so a line hugging the bottom means the "
                   "engine is far from acting.")

    # ---- card 5: does it hold up
    with st.container(border=True):
        st.markdown("**Does it hold up?**")
        if check.get("available"):
            h1, h2, h3 = st.columns(3)
            h1.metric(f"Interval coverage, {row['stage']} stage",
                      f"{100 * check['coverage']:.1f}%" if check["stage_windows"] else "n/a",
                      help="How often the real RUL landed inside the 90% range on NASA engines held out of "
                           "calibration. It should be near 90.")
            h2.metric("Typical model error (RMSE)", f"{check['rmse']:.1f} cycles" if check["stage_windows"] else "n/a",
                      help=f"Root mean squared error on held-out {row['stage']}-stage windows.")
            h3.metric("Cost vs the point rule",
                      f"{check['change_pct']:+.0f}%" if not np.isnan(check["change_pct"]) else "n/a",
                      help="Average cost per held-out window, ECHO against acting only when the predicted RUL is at "
                           "or below H.")
            st.caption(f"Checked on {check['windows']:,} windows from {check['engines']} NASA test engines the "
                       f"intervals and p were not fitted on ({check['stage_windows']:,} in the {row['stage']} stage). "
                       "This is a single split with no confidence interval; the report has repeated splits"
                       + ("." if validated else ", and only for H = 15, 30 and 50."))
            if is_upload:
                st.caption("Uploaded engines have no true RUL, so coverage cannot be checked on them. The intervals "
                           "and p were calibrated on the NASA test engines; a different fleet or different operating "
                           "conditions may not be covered at the stated 90%.")
            else:
                st.caption("The interval and p shown above use all test engines, including this one, so they look "
                           "slightly better than they will on a new engine. The numbers in this strip are the fair ones.")
        else:
            st.caption("This dataset has no held-out split in its results, so no checks are shown.")


if active is not None:
    render_main()
else:
    st.info("No uploaded engines for this dataset yet. Use the upload section below, or switch the data source to "
            "the NASA test engines.")


# ------------------------------------------------------------------------------------------------ upload
def render_upload():
    with st.container(border=True):
        _render_upload_body()


def _render_upload_body():
    st.markdown("### Upload your own engine data")
    st.caption("Optional. Scores engines that are not in the NASA test set with the same model and calibration as "
               f"above ({subset}). Predictions run on the CPU.")
    ready, why = runtime_status(ROOT, subset)
    if not ready:
        st.info(f"Uploading is not available on this machine. {why}")
        return
    runtime = get_runtime(subset)
    st.markdown(f"**File format:** one row per engine and cycle, no header, space or comma separated, "
                f"{len(RAW_COLUMNS)} numeric columns: engine id, cycle, three operating settings, then 21 sensors "
                "(the raw NASA layout). Every engine needs at least "
                f"{runtime.window_size} cycles.")
    file = st.file_uploader("Drop a file here", type=["txt", "csv"], key=f"file_{subset}")
    if file is not None:
        frame, check = parse_upload(file.getvalue(), runtime.window_size)
        st.markdown("**Checks before predicting**")
        for message in check.errors:
            st.error(message)
        if frame is not None:
            st.success(f"Layout and numbers look right: {check.engines} engines, {check.rows:,} rows, "
                       f"{check.windows:,} windows to score.")
            for message in check.warnings:
                st.warning(message)
            odd = settings_warning(frame, runtime.preproc)
            if odd:
                st.warning(odd)
            if st.button("Run predictions", type="primary"):
                with st.spinner("Scoring engines..."):
                    windows = score_upload(frame, runtime)
                st.session_state.setdefault("upload", {})[subset] = {
                    "bundle": upload_bundle(ref_bundle, windows), "name": file.name}
                st.session_state["goto_upload"] = True
                st.rerun()
    results = st.session_state.get("upload", {}).get(subset)
    if results:
        st.markdown(f"**Review** (file: {results['name']}). The table uses the ratio, H and band chosen on the left.")
        table = latest_table(results["bundle"], float(horizon), ratio, band)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", table.to_csv(index=False).encode(), "echo_uploaded_results.csv",
                           "text/csv")
        st.caption("Pick \"My uploaded file\" under Data source on the left to open any of these engines above.")
    st.caption("Limit: the model and the intervals were built from NASA's simulated engines. A different fleet or "
               "different operating conditions may not be covered at the stated 90%. Uploaded engines have no true "
               "RUL, so coverage cannot be checked on them.")


render_upload()

with st.expander("Glossary"):
    st.markdown(
        "- **RUL**: remaining useful life, the cycles until the engine fails.\n"
        "- **Cycle**: one flight, the time step in the data.\n"
        "- **90% interval**: a range built so the true RUL lands inside it about 90% of the time (conformal "
        "prediction), sized from the model's past errors.\n"
        "- **Horizon H**: how far ahead you plan. A failure counts if it happens within H cycles.\n"
        "- **p**: the estimated chance of failing within H, from how wrong the model was on similar past windows.\n"
        "- **Cost ratio**: unplanned failure cost divided by planned service cost.\n"
        "- **Break-even T**: 1 / p. Act when the cost ratio is above T.\n"
        "- **Investigate**: p x ratio is close to 1, so acting and waiting cost about the same.\n"
        "- **Point rule**: the simple baseline. Act when the predicted RUL is at or below H, ignoring uncertainty and cost.\n"
        "- **Decision flip**: a case where ECHO and the point rule give different advice.")