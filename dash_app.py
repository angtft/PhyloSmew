#!/usr/bin/env python3
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import base64, io

# NEW: multiple-testing correction for Wilcoxon
from statsmodels.stats.multitest import multipletests

APP_TITLE = "Phylogenetic Inference Benchmark Dashboard"

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
CORR_LLH_TOL = 0.05
CONSEL_P_THRESHOLD = 0.05  # pass if p > 0.05

# Variables users can correlate
VAR_OPTIONS = [
    {"label": "Difficulty", "value": "difficulty"},
    {"label": "RF distance to TRUE", "value": "rf"},
    {"label": "NTD distance to TRUE", "value": "ntd"},
    {"label": "Δ log-likelihood (tool − true)", "value": "llh"},
    {"label": "CONSEL p-value", "value": "consel"},
]

CATEGORY_OPTIONS = [
    {"label": "Δ log-likelihood vs TRUE (tool − true)", "value": "llh"},
    {"label": "RF distance to TRUE", "value": "rf"},
    {"label": "NTD distance to TRUE", "value": "ntd"},
    {"label": "Relative runtime vs RAxML1 (median per bin + speedup labels)", "value": "runtime"},
    {"label": "CONSEL: pass rate (%)", "value": "consel"},
    {"label": "CONSEL: corrected pass rate (%)", "value": "consel_corrected"},
    {"label": "Relative runtime (accumulated; sum times → speedup)", "value": "runtime_accum"},
    {"label": "Relative runtime (ignore reference < 60 s)", "value": "runtime_gt1min"},
    {"label": "Correlation (scatter + trendline + R²)", "value": "corr"},
    {"label": "RF distribution similarity (violin + ANOVA/Wilcoxon)", "value": "rf_dist"},
]

def nice_tool_label(name: str) -> str:
    return (name.replace("_", " ").replace("-", " ").upper()
            .replace("RAXML", "RAxML").replace("IQTREE", "IQTREE")
            .replace("FASTTREE", "FastTree").replace("BIGRAXML", "bigRAxML")
            .replace("  ", " "))

def bin_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["difficulty_bin"] = pd.cut(
        out["difficulty"].astype(float),
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return out

# ---------- CONSEL helpers (suffix-aware) ----------
def get_consel_tests(df: pd.DataFrame) -> list[str]:
    tests = set()
    for c in df.columns:
        if c.startswith("consel_"):
            rest = c[len("consel_"):]
            if "_" in rest:
                test = rest.rsplit("_", 1)[1]
                tests.add(test)
    return sorted(tests)

def pick_default_consel_test(df: pd.DataFrame) -> str | None:
    tests = get_consel_tests(df)
    if not tests:
        return None
    return "au" if "au" in tests else tests[0]

def consel_colname(tool: str, test: str) -> str:
    return f"consel_{tool}_{test}"

def detect_tools(df: pd.DataFrame, category: str) -> list[str]:
    tools = set()
    if category in ("rf", "rf_dist", "corr"):
        for c in df.columns:
            if c.startswith("rf_true_"):
                tools.add(c.split("rf_true_", 1)[1])
        if category == "corr":
            for c in df.columns:
                if c.startswith("llh_") and c != "llh_true":
                    tools.add(c.split("llh_", 1)[1])
                if c.startswith("consel_"):
                    rest = c[len("consel_"):]
                    tool = rest.rsplit("_", 1)[0] if "_" in rest else rest
                    if tool and tool != "true":
                        tools.add(tool)
                if c.startswith("ntd_true_"):
                    tools.add(c.split("ntd_true_", 1)[1])
    elif category == "ntd":
        for c in df.columns:
            if c.startswith("ntd_true_"):
                tools.add(c.split("ntd_true_", 1)[1])
    elif category == "llh":
        for c in df.columns:
            if c.startswith("llh_") and c != "llh_true":
                tools.add(c.split("llh_", 1)[1])
    elif category == "runtime":
        for c in df.columns:
            if c.startswith("abs_time_") and c != "abs_time_raxml1":
                tools.add(c.split("abs_time_", 1)[1])
    elif category in ("consel", "consel_corrected"):
        for c in df.columns:
            if c.startswith("consel_"):
                rest = c[len("consel_"):]
                tool = rest.rsplit("_", 1)[0] if "_" in rest else rest
                if tool and tool != "true":
                    tools.add(tool)
    elif category in ("runtime_accum", "runtime_gt1min"):
        for c in df.columns:
            if c.startswith("abs_time_") and c != "abs_time_raxml1":
                tools.add(c.split("abs_time_", 1)[1])
    return sorted(tools)

# ---------- runtime reference picker ----------
def choose_runtime_reference(dfx: pd.DataFrame, selected_tools: list[str] | None):
    if "abs_time_raxml1" in dfx.columns and dfx["abs_time_raxml1"].notna().any():
        return "abs_time_raxml1", "RAxML1"
    for t in (selected_tools or []):
        col = f"abs_time_{t}"
        if col in dfx.columns:
            return col, nice_tool_label(t)
    for c in sorted([c for c in dfx.columns if c.startswith("abs_time_")]):
        return c, nice_tool_label(c.split("abs_time_", 1)[1])
    return None, None

# ---------- corrected CONSEL (p>threshold, with LLH correction) ----------
def build_corrected_consel_long(df: pd.DataFrame, selected_tools: list[str], consel_test: str, llh_tol: float = CORR_LLH_TOL) -> pd.DataFrame:
    dfx = bin_difficulty(df)
    rows = []

    def pass_flag(row, tool):
        col = consel_colname(tool, consel_test)
        p = row.get(col)
        if pd.isna(p):
            p = row.get(f"consel_{tool}")  # legacy
        if pd.isna(p):
            return False
        try:
            return float(p) > CONSEL_P_THRESHOLD
        except Exception:
            return False

    for _, row in dfx.iterrows():
        passed_tools = [t for t in selected_tools if pass_flag(row, t)]
        llh_vals = {t: row.get(f"llh_{t}") for t in selected_tools if f"llh_{t}" in dfx.columns and pd.notna(row.get(f"llh_{t}"))}
        if passed_tools:
            best = None
            if llh_vals:
                cand = [(t, llh_vals.get(t, -np.inf)) for t in passed_tools if t in llh_vals]
                if cand:
                    best = max(cand, key=lambda kv: kv[1])[0]
            passed_set = {best} if best else set(passed_tools)
        else:
            passed_set = set()

        for t in selected_tools:
            corrected = 1.0 if t in passed_set else 0.0
            if corrected == 0.0 and passed_set and t in llh_vals:
                diffs = [abs(llh_vals[t] - llh_vals.get(pt, -np.inf)) for pt in passed_set if pt in llh_vals]
                if diffs and np.min(diffs) <= llh_tol:
                    corrected = 1.0
            rows.append({
                "exp_id": row["exp_id"],
                "difficulty_bin": row["difficulty_bin"],
                "tool": t,
                "value": corrected,
            })
    out = pd.DataFrame(rows)
    return out.dropna(subset=["difficulty_bin"])

# ---------- runtime aggregations ----------
def aggregate_runtime_accum(df: pd.DataFrame, selected_tools: list[str]) -> pd.DataFrame:
    dfx = bin_difficulty(df).dropna(subset=["difficulty_bin"])
    ref_col, _ref_label = choose_runtime_reference(dfx, selected_tools)
    if not ref_col:
        return pd.DataFrame(columns=["difficulty_bin", "Tool", "rel", "speedup_str"])
    dfx = dfx.copy()
    dfx["_ref_time"] = dfx[ref_col].astype(float)
    g = dfx.groupby("difficulty_bin", observed=True)
    sum_ref = g["_ref_time"].sum().replace(0, np.nan)
    out = []
    for t in selected_tools:
        col = f"abs_time_{t}"
        if col not in dfx.columns:
            continue
        sum_tool = g[col].sum()
        rel = sum_tool / sum_ref
        for b, v in rel.items():
            if pd.isna(v) or np.isinf(v):
                continue
            out.append({"difficulty_bin": b, "Tool": nice_tool_label(t), "rel": v, "speedup_str": f"{(1.0/v):.2f}×"})
    return pd.DataFrame(out)

def aggregate_runtime_filtered(df: pd.DataFrame, selected_tools: list[str], min_ref_seconds: float = 60.0) -> pd.DataFrame:
    dfx = bin_difficulty(df).dropna(subset=["difficulty_bin"])
    ref_col, _ref_label = choose_runtime_reference(dfx, selected_tools)
    if not ref_col:
        return pd.DataFrame(columns=["difficulty_bin", "Tool", "rel", "speedup_str"])
    dfx = dfx[dfx[ref_col] >= min_ref_seconds]
    if dfx.empty:
        return pd.DataFrame(columns=["difficulty_bin", "Tool", "rel", "speedup_str"])
    denom = dfx[ref_col].astype(float).replace(0, np.nan)
    out = []
    for t in selected_tools:
        col = f"abs_time_{t}"
        if col not in dfx.columns:
            continue
        rel = dfx[col].astype(float) / denom
        med = rel.groupby(dfx["difficulty_bin"], observed=True).median()
        for b, v in med.items():
            if pd.isna(v):
                continue
            out.append({"difficulty_bin": b, "Tool": nice_tool_label(t), "rel": v, "speedup_str": f"{(1.0/v):.2f}×"})
    return pd.DataFrame(out)

# ---------- long-form melt ----------
def melt_for_category(df: pd.DataFrame, category: str, tools: list[str], consel_test: str | None = None) -> pd.DataFrame:
    df = bin_difficulty(df)
    pieces = []
    if category == "llh":
        if "llh_true" not in df.columns:
            return pd.DataFrame()
        for t in tools:
            col = f"llh_{t}"
            if col in df.columns:
                tmp = df[["exp_id", "difficulty_bin"]].copy()
                tmp["tool"] = t
                tmp["value"] = df[col] - df["llh_true"]
                pieces.append(tmp)
    elif category in ("rf", "rf_dist", "ntd"):
        prefix = "rf_true_" if category != "ntd" else "ntd_true_"
        extra_cols = ["difficulty"] if category != "ntd" else []
        for t in tools:
            col = f"{prefix}{t}"
            if col in df.columns:
                tmp = df[["exp_id", "difficulty_bin"] + extra_cols].copy()
                tmp["tool"] = t
                tmp["value"] = df[col]
                pieces.append(tmp)
    elif category == "runtime":
        ref_col, _ref_label = choose_runtime_reference(df, tools)
        if not ref_col:
            return pd.DataFrame()
        denom = df[ref_col].astype(float).replace(0, np.nan)
        for t in tools:
            col = f"abs_time_{t}"
            if col in df.columns:
                tmp = df[["exp_id", "difficulty_bin"]].copy()
                tmp["tool"] = t
                tmp["value"] = df[col].astype(float) / denom
                pieces.append(tmp)
    elif category == "consel":
        if consel_test is None:
            consel_test = pick_default_consel_test(df)
        for t in tools:
            col = consel_colname(t, consel_test)
            if col not in df.columns:
                col = f"consel_{t}"
                if col not in df.columns:
                    continue
            tmp = df[["exp_id", "difficulty_bin"]].copy()
            tmp["tool"] = t
            vals = df[col].astype(float)
            tmp["value"] = (vals > CONSEL_P_THRESHOLD).astype(float)
            pieces.append(tmp)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out = out.dropna(subset=["difficulty_bin", "value"])
    return out

def _strip_facet_prefixes(fig):
    for ann in fig.layout.annotations or []:
        if ann.text and "=" in ann.text:
            ann.text = ann.text.split("=", 1)[1]

def _counts_by_bin_from_df(df: pd.DataFrame) -> dict:
    dfx = bin_difficulty(df).dropna(subset=["difficulty_bin"])
    s = dfx.groupby("difficulty_bin", observed=True)["exp_id"].nunique()
    return {str(k): int(v) for k, v in s.items()}

def _apply_counts_to_facet_titles(fig, counts_map: dict):
    for ann in fig.layout.annotations or []:
        if not ann.text:
            continue
        label = ann.text.split("=")[-1]
        n = counts_map.get(label)
        if n is not None:
            ann.text = f"{label} ({n})"

# -------- Correlation, per-bin Wilcoxon (Holm-adjusted), and ANOVA --------
def pick_default_consel_test_for_corr(df: pd.DataFrame) -> str | None:
    return pick_default_consel_test(df)

def _get_series(df: pd.DataFrame, var: str, tool: str | None):
    if var == "difficulty":
        return df["difficulty"].astype(float)
    if tool is None:
        return None
    if var == "rf":
        col = f"rf_true_{tool}"
        return df[col].astype(float) if col in df.columns else None
    if var == "ntd":
        col = f"ntd_true_{tool}"
        return df[col].astype(float) if col in df.columns else None
    if var == "llh":
        col = f"llh_{tool}"
        if col in df.columns and "llh_true" in df.columns:
            return (df[col] - df["llh_true"]).astype(float)
        return None
    if var == "consel":
        test = pick_default_consel_test_for_corr(df)
        if not test:
            return None
        col = consel_colname(tool, test)
        if col not in df.columns:
            col = f"consel_{tool}"
            if col not in df.columns:
                return None
        return df[col].astype(float)
    return None

def figure_corr_scatter(df: pd.DataFrame, selected_tools: list[str], xvar: str, yvar: str):
    traces = []
    for t in selected_tools:
        x = _get_series(df, xvar, t)
        y = _get_series(df, yvar, t)
        if x is None or y is None:
            continue
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        if sub.empty:
            continue
        xs = sub["x"].values
        ys = sub["y"].values
        label = nice_tool_label(t)
        traces.append(go.Scatter(x=xs, y=ys, mode="markers", name=label, legendgroup=t))
        if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
            a, b = np.polyfit(xs, ys, 1)
            xfit = np.linspace(float(xs.min()), float(xs.max()), 50)
            yfit = a * xfit + b
            r = np.corrcoef(xs, ys)[0, 1]
            r2 = float(r * r)
            traces.append(go.Scatter(
                x=xfit, y=yfit, mode="lines",
                name=f"{label} fit (R²={r2:.3f})", legendgroup=t, showlegend=True
            ))
    label_map = {
        "rf": "RF distance to TRUE",
        "ntd": "NTD distance to TRUE",
        "llh": "Δ log-likelihood (tool − true)",
        "consel": "CONSEL p-value",
        "difficulty": "Difficulty",
    }
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Correlation: {label_map.get(xvar)} vs {label_map.get(yvar)} (per tool) — linear fit & R²",
        xaxis_title=label_map.get(xvar), yaxis_title=label_map.get(yvar),
        margin=dict(t=60, r=10, l=10, b=40),
    )
    return fig

def wilcoxon_heatmap_per_bin_figure(long_df: pd.DataFrame, selected_tools: list[str], counts_map: dict):
    """
    For each difficulty bin:
      1) Align paired values by exp_id for every tool pair
      2) Compute two-sided Wilcoxon signed-rank test p-values on paired differences
      3) Apply Holm FWER correction *within the bin* across all unique tool pairs
      4) Plot a per-bin heatmap of Holm-adjusted p-values (symmetric; diag=1)
    """
    dfp = long_df.copy()
    dfp = dfp[dfp["tool"].isin(selected_tools)]
    bins_present = [b for b in BIN_LABELS if b in dfp["difficulty_bin"].unique().tolist()]
    if not bins_present:
        return px.imshow(np.zeros((1, 1)), title="No data for Wilcoxon heatmaps")

    tool_order = sorted(dfp["tool"].unique())
    ticktext = [nice_tool_label(t) for t in tool_order]
    n_tools = len(tool_order)

    fig = make_subplots(
        rows=1, cols=len(bins_present),
        subplot_titles=[f"{b} ({counts_map.get(str(b), 0)})" for b in bins_present],
        horizontal_spacing=0.06,
        column_widths=[1/len(bins_present)] * len(bins_present),
    )

    try:
        from scipy.stats import wilcoxon
        from itertools import combinations

        for idx, b in enumerate(bins_present, start=1):
            sub = dfp[dfp["difficulty_bin"] == b]
            pivot = sub.pivot_table(index="exp_id", columns="tool", values="value", aggfunc="first")

            # Collect unique pairwise p-values (i<j)
            pair_indices = []
            pvals = []
            ns = []
            for i, j in combinations(range(n_tools), 2):
                ta, tb = tool_order[i], tool_order[j]
                if ta in pivot.columns and tb in pivot.columns:
                    paired = pivot[[ta, tb]].dropna()
                    n = paired.shape[0]
                    pval = np.nan
                    if n >= 1:
                        diffs = (paired[ta] - paired[tb]).values
                        if np.allclose(diffs, 0):
                            pval = 1.0
                        else:
                            try:
                                res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
                                pval = float(res.pvalue)
                            except Exception:
                                pval = np.nan
                    pair_indices.append((i, j))
                    pvals.append(1.0 if pd.isna(pval) else pval)
                    ns.append(n)

            # Holm adjust within this bin
            if pvals:
                _, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
            else:
                p_adj = []

            # Build symmetric z-matrix of adjusted p-values
            z = np.full((n_tools, n_tools), np.nan)
            for i in range(n_tools):
                z[i, i] = 1.0
            for (i, j), p in zip(pair_indices, p_adj):
                z[i, j] = p
                z[j, i] = p

            # Optional: customdata for hover (n of pairs)
            custom = np.empty((n_tools, n_tools, 1))
            custom[:] = np.nan
            for (i, j), n in zip(pair_indices, ns):
                custom[i, j, 0] = n
                custom[j, i, 0] = n

            heat = go.Heatmap(
                z=z, x=ticktext, y=ticktext, zmin=0, zmax=1,
                showscale=(idx == len(bins_present)),
                colorbar=dict(title="p (Holm)", len=0.8),
                customdata=custom,
                hovertemplate="%(a)s vs %(b)s<br>n pairs=%{customdata[0]:.0f}<br>p (Holm)=%{z:.3g}<extra></extra>"
                    .replace("%(a)s", "%{y}")
                    .replace("%(b)s", "%{x}")
            )
            fig.add_trace(heat, row=1, col=idx)
            fig.update_xaxes(tickangle=45, row=1, col=idx)

    except Exception as e:
        fig.add_annotation(
            text=f"Wilcoxon heatmaps unavailable (SciPy/statsmodels not available: {e})",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    fig.update_layout(
        title="Pairwise Wilcoxon signed-rank p-values (Holm-adjusted) for RF — per difficulty bin",
        margin=dict(t=80, r=10, l=10, b=40),
    )
    return fig

def anova_text_block(long_df: pd.DataFrame, selected_tools: list[str], response_label: str) -> str:
    dfp = long_df.copy()
    dfp = dfp[dfp["tool"].isin(selected_tools)].dropna(subset=["value", "difficulty_bin"])
    if dfp.empty:
        return "No data for ANOVA."
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        dfp["tool"] = dfp["tool"].astype("category")
        dfp["difficulty_bin"] = dfp["difficulty_bin"].astype("category")
        model = smf.ols("value ~ C(tool) + C(difficulty_bin) + C(tool):C(difficulty_bin)", data=dfp).fit()
        table = anova_lm(model, typ=2)
        hdr = (
            f"Two-way ANOVA on {response_label}\n"
            "Model: value ~ C(tool) + C(difficulty_bin) + C(tool):C(difficulty_bin)\n"
            "Type-II sums of squares."
        )
        return hdr + "\n" + table.to_string()
    except Exception as e:
        return "Two-way ANOVA unavailable (install `statsmodels`). Details: " + str(e)

# ---------------- Figure factory ----------------
def make_figure(df: pd.DataFrame, long_df: pd.DataFrame, category: str, selected_tools: list[str], counts_map: dict, consel_test: str | None):
    if category in ("llh", "rf", "ntd"):
        if long_df.empty:
            return px.scatter(title="No data to display (check your selections & columns).")
        dfp = long_df.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        titles = {
            "llh": ("Δ Log-likelihood (tool − true)", "Δ log-likelihood"),
            "rf": ("RF Distance to TRUE", "RF distance"),
            "ntd": ("NTD Distance to TRUE", "NTD distance"),
        }
        title, y_title = titles[category]
        fig = px.box(
            dfp, x="Tool", y="value", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            points=False,
        )
        fig.update_layout(
            title=f"{title} — split by Difficulty Bin",
            xaxis_title="Tool", yaxis_title=y_title,
            boxmode="group", legend_title="Tool",
            margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    if category == "runtime":
        if long_df.empty:
            return px.bar(title="No runtime data to display.")
        ref_col, ref_label = choose_runtime_reference(bin_difficulty(df), selected_tools)
        ref_label = ref_label or "reference"
        dfp = long_df.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        agg = (dfp.groupby(["difficulty_bin", "Tool"], observed=True)["value"]
               .median().reset_index(name="median_rel"))
        agg["speedup_str"] = (1.0 / agg["median_rel"]).map(lambda x: f"{x:.2f}×")
        agg["difficulty_bin"] = pd.Categorical(agg["difficulty_bin"], categories=BIN_LABELS, ordered=True)
        fig = px.bar(
            agg, x="Tool", y="median_rel", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="speedup_str",
        )
        fig.update_traces(textposition="outside")
        ymax = (agg["median_rel"].max() or 1.0) * 1.18
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, ymax])
        fig.update_layout(
            title=f"Relative Runtime vs {ref_label} (median per bin) — labels show speedup = 1 / runtime",
            xaxis_title="Tool", yaxis_title=f"Median relative runtime (× {ref_label})",
            legend_title="Tool", margin=dict(t=70, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    if category == "consel":
        if long_df.empty:
            return px.bar(title="No CONSEL data to display.")
        dfp = long_df.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        agg = (dfp.groupby(["difficulty_bin", "Tool"], observed=True)["value"]
               .mean().reset_index(name="pass_rate"))
        agg["percent"] = (agg["pass_rate"] * 100).round(1)
        agg["difficulty_bin"] = pd.Categorical(agg["difficulty_bin"], categories=BIN_LABELS, ordered=True)
        test_lbl = (consel_test or pick_default_consel_test(df) or "").upper()
        fig = px.bar(
            agg, x="Tool", y="percent", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="percent",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, 105])
        fig.update_layout(
            title=f"CONSEL ({test_lbl}) Pass Rate (%) by Difficulty Bin  —  pass = p > {CONSEL_P_THRESHOLD}",
            xaxis_title="Tool", yaxis_title="Pass rate (%)",
            legend_title="Tool", margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    if category == "consel_corrected":
        if long_df.empty:
            return px.bar(title="No data for corrected CONSEL (need consel_*_* and llh_* columns).")
        dfp = long_df.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        agg = (dfp.groupby(["difficulty_bin", "Tool"], observed=True)["value"]
               .mean().reset_index(name="pass_rate"))
        agg["percent"] = (agg["pass_rate"] * 100).round(1)
        agg["difficulty_bin"] = pd.Categorical(agg["difficulty_bin"], categories=BIN_LABELS, ordered=True)
        test_lbl = (consel_test or pick_default_consel_test(df) or "").upper()
        fig = px.bar(
            agg, x="Tool", y="percent", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="percent",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, 105])
        fig.update_layout(
            title=f"Corrected CONSEL ({test_lbl}) Pass Rate (%) by Difficulty Bin  —  pass = (p > {CONSEL_P_THRESHOLD}) or within ΔLLH ≤ {CORR_LLH_TOL}",
            xaxis_title="Tool", yaxis_title="Pass rate (%)",
            legend_title="Tool", margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    if category == "runtime_accum":
        agg = aggregate_runtime_accum(df, selected_tools)
        if agg.empty:
            return px.bar(title="No data to compute accumulated runtime.")
        ref_col, ref_label = choose_runtime_reference(bin_difficulty(df), selected_tools)
        ref_label = ref_label or "reference"
        fig = px.bar(
            agg, x="Tool", y="rel", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="speedup_str",
        )
        fig.update_traces(textposition="outside")
        ymax = (agg["rel"].max() or 1.0) * 1.18
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, ymax])
        fig.update_layout(
            title=f"Accumulated Relative Runtime vs {ref_label} (sum times per bin) — labels show speedup = 1 / relative",
            xaxis_title="Tool", yaxis_title=f"Sum(tool) / Sum({ref_label})",
            legend_title="Tool", margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    if category == "runtime_gt1min":
        agg = aggregate_runtime_filtered(df, selected_tools, min_ref_seconds=60.0)
        if agg.empty:
            return px.bar(title="No data after filtering by reference < 60 s.")
        ref_col, ref_label = choose_runtime_reference(bin_difficulty(df), selected_tools)
        ref_label = ref_label or "reference"
        fig = px.bar(
            agg, x="Tool", y="rel", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="speedup_str",
        )
        fig.update_traces(textposition="outside")
        ymax = (agg["rel"].max() or 1.0) * 1.18
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, ymax])
        fig.update_layout(
            title=f"Relative Runtime vs {ref_label} (median per bin; ignoring {ref_label} < 60 s) — labels show speedup",
            xaxis_title="Tool", yaxis_title=f"Median relative runtime (× {ref_label})",
            legend_title="Tool", margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig

    return px.scatter(title="Unsupported category.")

def parse_uploaded(contents: str, filename: str) -> pd.DataFrame | None:
    try:
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        text = decoded.decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"Failed to parse {filename}: {e}")
        return None

app = Dash(__name__, title=APP_TITLE)
server = app.server

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "padding": "1rem 2rem"},
    children=[
        html.H1(APP_TITLE, style={"marginBottom": "0.25rem"}),
        html.P("Upload your CSV, choose a category, and select tools. Plots are split into 5 difficulty bins (facets)."),
        dcc.Upload(
            id="upload",
            children=html.Div(["Drag & drop or ", html.A("select CSV")]),
            style={
                "width": "100%", "height": "64px", "lineHeight": "64px",
                "borderWidth": "1px", "borderStyle": "dashed",
                "borderRadius": "8px", "textAlign": "center", "marginBottom": "1rem",
            },
            multiple=False,
        ),
        dcc.Store(id="data-store"),
        html.Div(id="file-info", style={"marginBottom": "0.75rem", "color": "#444"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "0.75rem", "alignItems": "start"},
            children=[
                html.Div([
                    html.Label("Category", htmlFor="category"),
                    dcc.Dropdown(id="category", options=CATEGORY_OPTIONS, value="llh", clearable=False),
                ]),
                html.Div([
                    html.Label("Tools"),
                    dcc.Checklist(
                        id="tool-select", options=[], value=[], inline=True,
                        style={"display": "flex", "flexWrap": "wrap", "gap": "0.5rem"},
                        inputStyle={"marginRight": "0.35rem"},
                    ),
                ]),
            ],
        ),

        html.Div([
            html.Label("Threads"),
            dcc.RadioItems(
                id="thread-radio",
                options=[], value=None, inline=True,
                style={"display": "flex", "gap": "0.75rem", "flexWrap": "wrap"},
                inputStyle={"marginRight": "0.35rem"},
            ),
        ], style={"marginTop": "0.25rem"}),

        # Correlation controls (bigger, no wrapping)
        html.Div(
            id="corr-controls",
            style={"display": "none", "gap": "0.75rem", "alignItems": "end", "flexWrap": "wrap"},
            children=[
                html.Div(style={"minWidth": "520px", "maxWidth": "680px"}, children=[
                    html.Label("X variable"),
                    dcc.Dropdown(id="xvar", options=VAR_OPTIONS, value="difficulty", clearable=False, style={"width": "100%"}),
                ]),
                html.Div(style={"minWidth": "520px", "maxWidth": "680px"}, children=[
                    html.Label("Y variable"),
                    dcc.Dropdown(id="yvar", options=VAR_OPTIONS, value="rf", clearable=False, style={"width": "100%"}),
                ]),
            ],
        ),

        # CONSEL test selector
        html.Div(
            id="consel-controls",
            style={"display": "none", "marginTop": "0.5rem"},
            children=[
                html.Label("CONSEL test"),
                dcc.RadioItems(
                    id="consel-test",
                    options=[],
                    value=None,
                    inline=True,
                    style={"display": "flex", "gap": "0.75rem", "flexWrap": "wrap"},
                    inputStyle={"marginRight": "0.35rem"},
                ),
                html.Div(style={"color": "#666", "marginTop": "0.25rem"}, children=f"Pass criterion: p > {CONSEL_P_THRESHOLD}")
            ],
        ),

        dcc.Graph(id="main-graph", style={"height": "60vh", "marginTop": "0.5rem"}),

        # Only shown in rf_dist
        html.Div(
            id="anova-container",
            style={"display": "none", "marginTop": "0.5rem"},
            children=[
                dcc.Graph(id="extra-graph", style={"height": "45vh"}),
                html.Pre(id="stats-text", style={"marginTop": "0.5rem", "whiteSpace": "pre-wrap"}),
            ],
        ),

        html.Details(open=False, style={"marginTop": "0.5rem"}, children=[
            html.Summary("Notes & assumptions"),
            html.Ul([
                html.Li("ΔLLH = llh_tool − llh_true (0 means equal to TRUE; larger is better)."),
                html.Li("RF uses columns like rf_true_TOOL. NTD uses ntd_true_TOOL."),
                html.Li("Runtime uses median(abs_time_TOOL / abs_time_REF) per bin; labels show speedup = 1 / median runtime. REF is RAxML1 if present, else the first selected tool."),
                html.Li("CONSEL pass = p-value > 0.05. When 'corrected', a fail may pass if ΔLLH ≤ tolerance to a passing tool."),
                html.Li("RF Wilcoxon heatmaps show Holm-adjusted two-sided p-values per bin."),
                html.Li("Difficulty bins: [0.0,0.2], (0.2,0.4], (0.4,0.6], (0.6,0.8], (0.8,1.0]."),
                html.Li("Facet titles include (n) = number of unique experiments in that bin."),
            ])
        ]),
    ]
)

@app.callback(
    Output("data-store", "data"),
    Output("file-info", "children"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def load_data(contents, filename):
    df = parse_uploaded(contents, filename)
    if df is None or df.empty:
        return None, html.Span("Failed to read the uploaded file. Please upload a valid CSV with the expected header.")
    missing = [c for c in ("exp_id", "difficulty") if c not in df.columns]
    if missing:
        return None, html.Span(f"Missing required column(s): {', '.join(missing)}")
    keep_prefixes = ("llh_", "rf_true_", "ntd_true_", "abs_time_", "consel_", "exp_id", "difficulty", "thread_num")
    cols = [c for c in df.columns if c.startswith(keep_prefixes) or c in ("exp_id", "difficulty")]
    df = df[cols].copy()
    return df.to_json(date_format="iso", orient="split"), html.Span(f"Loaded: {filename}  •  rows={len(df)}")

@app.callback(
    Output("thread-radio", "options"),
    Output("thread-radio", "value"),
    Input("data-store", "data"),
)
def update_thread_radio(df_json):
    if not df_json:
        return [], None
    df = pd.read_json(df_json, orient="split")
    if "thread_num" not in df.columns:
        return [], None
    vals = sorted(v for v in df["thread_num"].dropna().unique())
    options = [{"label": str(int(v)), "value": int(v)} for v in vals]
    default = options[0]["value"] if options else None
    return options, default

@app.callback(
    Output("tool-select", "options"),
    Output("tool-select", "value"),
    Input("data-store", "data"),
    Input("category", "value"),
    Input("thread-radio", "value"),
)
def update_tool_options(df_json, category, thread_selected):
    if not df_json:
        return [], None
    df = pd.read_json(df_json, orient="split")
    if "thread_num" in df.columns and thread_selected is not None:
        try:
            df = df[df["thread_num"] == int(thread_selected)]
        except Exception:
            pass
    tools = detect_tools(df, category)
    options = [{"label": nice_tool_label(t), "value": t} for t in tools]
    return options, tools  # default: select all

@app.callback(
    Output("corr-controls", "style"),
    Input("category", "value"),
)
def toggle_corr_controls(category):
    return {"display": "flex", "gap": "0.75rem", "alignItems": "end", "flexWrap": "wrap"} if category == "corr" else {"display": "none"}

@app.callback(
    Output("consel-controls", "style"),
    Input("category", "value"),
)
def toggle_consel_controls(category):
    return {"display": "block", "marginTop": "0.5rem"} if category in ("consel", "consel_corrected") else {"display": "none"}

@app.callback(
    Output("consel-test", "options"),
    Output("consel-test", "value"),
    Input("data-store", "data"),
    Input("category", "value"),
)
def populate_consel_tests(df_json, category):
    if not df_json:
        return [], None
    df = pd.read_json(df_json, orient="split")
    tests = get_consel_tests(df)
    opts = [{"label": t.upper(), "value": t} for t in tests]
    default = "au" if "au" in tests else (tests[0] if tests else None)
    return opts, default

@app.callback(
    Output("anova-container", "style"),
    Input("category", "value"),
)
def toggle_anova_container(category):
    return {"display": "block", "marginTop": "0.5rem"} if category == "rf_dist" else {"display": "none"}

@app.callback(
    Output("main-graph", "figure"),
    Output("extra-graph", "figure"),
    Output("stats-text", "children"),
    Input("data-store", "data"),
    Input("category", "value"),
    Input("tool-select", "value"),
    Input("thread-radio", "value"),
    Input("xvar", "value"),
    Input("yvar", "value"),
    Input("consel-test", "value"),
)
def update_graph(df_json, category, selected_tools, thread_selected, xvar, yvar, consel_test):
    blank = px.scatter(title="")
    if not df_json:
        return px.scatter(title="Upload a CSV to begin."), blank, ""
    df = pd.read_json(df_json, orient="split")
    if "thread_num" in df.columns and thread_selected is not None:
        df = df[df["thread_num"] == int(thread_selected)]

    avail = set(detect_tools(df, category))
    selected_tools = [t for t in (selected_tools or []) if t in avail]

    counts_map = _counts_by_bin_from_df(df)

    if category in ("llh", "rf", "ntd", "runtime", "consel", "rf_dist", "consel_corrected"):
        melt_cat = category if category != "rf_dist" else "rf"
        if category == "consel_corrected":
            long_df = pd.DataFrame()
        else:
            long_df = melt_for_category(df, melt_cat, selected_tools, consel_test=consel_test)
    else:
        long_df = pd.DataFrame()

    if category == "corr":
        xv = xvar or "difficulty"
        yv = yvar or "rf"
        fig = figure_corr_scatter(df, selected_tools, xv, yv)
        return fig, blank, ""

    if category == "rf_dist":
        if long_df.empty:
            return px.scatter(title="No RF data to display."), blank, "No RF data to display."
        dfp = long_df.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        fig_violin = px.violin(
            dfp, x="Tool", y="value", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            box=True, points="all"
        )
        fig_violin.update_layout(
            title="RF distance distributions by Tool (violin+box), faceted by difficulty bin",
            xaxis_title="Tool", yaxis_title="RF distance to TRUE",
            margin=dict(t=60, r=10, l=10, b=40),
        )
        _strip_facet_prefixes(fig_violin)
        _apply_counts_to_facet_titles(fig_violin, counts_map)

        fig_heat = wilcoxon_heatmap_per_bin_figure(long_df, selected_tools, counts_map)
        stats_text = anova_text_block(long_df, selected_tools, "RF distance")
        return fig_violin, fig_heat, stats_text

    if category == "consel":
        main = make_figure(df, long_df, category, selected_tools, counts_map, consel_test)
        return main, blank, ""

    if category == "consel_corrected":
        if not selected_tools:
            return px.bar(title="Select at least one tool."), blank, ""
        long_corr = build_corrected_consel_long(df, selected_tools, consel_test or pick_default_consel_test(df))
        if long_corr.empty:
            return px.bar(title="No data for corrected CONSEL (need consel_*_* and llh_* columns)."), blank, ""
        dfp = long_corr.copy()
        dfp["Tool"] = dfp["tool"].map(nice_tool_label)
        agg = (dfp.groupby(["difficulty_bin", "Tool"], observed=True)["value"]
               .mean().reset_index(name="pass_rate"))
        agg["percent"] = (agg["pass_rate"] * 100).round(1)
        agg["difficulty_bin"] = pd.Categorical(agg["difficulty_bin"], categories=BIN_LABELS, ordered=True)
        test_lbl = (consel_test or pick_default_consel_test(df) or "").upper()
        fig = px.bar(
            agg, x="Tool", y="percent", color="Tool",
            facet_col="difficulty_bin", category_orders={"difficulty_bin": BIN_LABELS},
            text="percent",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        for ax in fig.layout:
            if ax.startswith("yaxis"):
                fig.layout[ax].update(range=[0, 105])
        fig.update_layout(
            title=f"Corrected CONSEL ({test_lbl}) Pass Rate (%) by Difficulty Bin  —  pass = (p > {CONSEL_P_THRESHOLD}) or within ΔLLH ≤ {CORR_LLH_TOL}",
            xaxis_title="Tool", yaxis_title="Pass rate (%)",
            legend_title="Tool", margin=dict(t=60, r=10, l=10, b=40),
            uniformtext_minsize=10, uniformtext_mode="hide",
        )
        fig.update_xaxes(tickangle=45)
        _strip_facet_prefixes(fig)
        _apply_counts_to_facet_titles(fig, counts_map)
        return fig, blank, ""

    main = make_figure(df, long_df, category, selected_tools, counts_map, consel_test)
    return main, blank, ""

if __name__ == "__main__":
    app.run(debug=True)
