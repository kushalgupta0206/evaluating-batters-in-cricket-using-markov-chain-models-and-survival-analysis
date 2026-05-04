"""
Shiny dashboard: run-expectation batter radar (entry stage vs play stage).

Run from project root: ``python -m shiny run app.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

STAGES = ["start", "middle", "end"]

_DATA_PATH = Path(__file__).resolve().parent / "data" / "run_expectation_model_batter_summary.csv"

_df = pd.read_csv(_DATA_PATH, dtype={"Event": str})
_df["Event"] = _df["Event"].str.zfill(4)
_df["is_wkt"] = _df["Event"].str[0] == "1"
_df["Batter.Runs"] = _df["Batter.Runs"].astype(np.int32)
_df["exp_runs_prev"] = _df["exp_runs_prev"].astype(np.float64)
_df["exp_runs_post"] = _df["exp_runs_post"].astype(np.float64)
_df["rva"] = _df["Batter.Runs"] + _df["exp_runs_post"] - _df["exp_runs_prev"]

_BATTERS = sorted(_df["Batter"].unique().tolist())


def _stage_list(x: object) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(s) for s in list(x)]


def compute_metrics(sub: pd.DataFrame) -> dict:
    """Aggregate metrics for filtered ball-by-ball rows."""
    balls = len(sub)
    matches = int(sub["Match.ID"].nunique()) if balls else 0
    runs = int(sub["Batter.Runs"].sum())
    dis = int(sub["is_wkt"].sum())
    avg_runs = runs / max(dis, 1)
    strike_rate = (runs / balls * 100.0) if balls else 0.0
    rva_mean = float(sub["rva"].mean()) if balls else 0.0
    if balls:
        dots = float(((sub["Batter.Runs"] == 0) & (~sub["is_wkt"])).mean() * 100.0)
        pct_12 = float(sub["Batter.Runs"].isin([1, 2]).mean() * 100.0)
        pct_46 = float(sub["Batter.Runs"].isin([4, 6]).mean() * 100.0)
    else:
        dots = pct_12 = pct_46 = 0.0
    return {
        "balls": balls,
        "matches": matches,
        "runs": runs,
        "dismissals": dis,
        "avg_runs": avg_runs,
        "strike_rate": strike_rate,
        "rva": rva_mean,
        "pct_dots": dots,
        "pct_12": pct_12,
        "pct_46": pct_46,
    }


def filter_batter_stages(
    df: pd.DataFrame,
    batter: str,
    entry_stages: list[str],
    play_stages: list[str],
) -> pd.DataFrame:
    if not entry_stages or not play_stages:
        return df.iloc[0:0].copy()
    return df[
        (df["Batter"] == batter)
        & (df["Entry.Stage"].isin(entry_stages))
        & (df["Match.Stage"].isin(play_stages))
    ]


def filter_all_batters_stages(
    df: pd.DataFrame,
    entry_stages: list[str],
    play_stages: list[str],
) -> pd.DataFrame:
    if not entry_stages or not play_stages:
        return df.iloc[0:0].copy()
    return df[
        (df["Entry.Stage"].isin(entry_stages))
        & (df["Match.Stage"].isin(play_stages))
    ]


def global_metric_bounds(
    df: pd.DataFrame,
    entry_stages: list[str],
    play_stages: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Min/max of each radar metric across every batter with ≥1 ball under the
    given entry and play stage filters.
    """
    sub = filter_all_batters_stages(df, entry_stages, play_stages)
    if len(sub) == 0:
        return None
    rows: list[list[float]] = []
    for b in sub["Batter"].unique():
        m = compute_metrics(sub[sub["Batter"] == b])
        if m["balls"] == 0:
            continue
        rows.append(
            [
                m["avg_runs"],
                m["strike_rate"],
                m["rva"],
                m["pct_dots"],
                m["pct_12"],
                m["pct_46"],
            ]
        )
    if not rows:
        return None
    M = np.array(rows, dtype=float)
    return M.min(axis=0), M.max(axis=0)


RADAR_LABELS = [
    "Avg runs / dismissal",
    "Strike rate",
    "Run value added (mean)",
    "% Dot balls",
    "% 1s & 2s",
    "% 4s & 6s",
]

# Main chart size in browser (pixels)
RADAR_FIG_WIDTH = 1400
RADAR_FIG_HEIGHT = 1100


def raw_matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [
            [
                r["avg_runs"],
                r["strike_rate"],
                r["rva"],
                r["pct_dots"],
                r["pct_12"],
                r["pct_46"],
            ]
            for r in rows
        ],
        dtype=float,
    )


def scale_matrix_with_bounds(
    M: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Map each column linearly to [0, 1] using fixed per-metric lo/hi; clip; degenerate → 0.5."""
    out = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        l, h = float(lo[j]), float(hi[j])
        if not np.isfinite(l) or not np.isfinite(h) or h <= l:
            out[:, j] = 0.5
        else:
            out[:, j] = np.clip((M[:, j] - l) / (h - l), 0.0, 1.0)
    return out


def _fmt_hover_raw(k: int, v: float) -> str:
    if k == 2:
        return f"{v:.6f}"
    return f"{v:.4f}"


def _fmt_tick_raw(k: int, v: float) -> str:
    if k == 2:
        return f"{v:.4f}"
    if k == 0:
        return f"{v:.2f}"
    if k in (3, 4, 5):
        return f"{v:.1f}"
    return f"{v:.1f}"


def _radar_angles(n: int) -> np.ndarray:
    """Angles (radians); first axis at 12 o'clock, then clockwise."""
    return np.array([np.pi / 2 - 2 * np.pi * j / n for j in range(n)])


def _radar_plot_title_html(
    entry_stages: list[str], play_stages: list[str], extra_line: str | None = None
) -> str:
    es = ", ".join(entry_stages) if entry_stages else "—"
    ps = ", ".join(play_stages) if play_stages else "—"
    sub = f"Entry Stages: {es} | Play Stages: {ps}"
    parts = [
        "<b>Batter Radar First Innings</b><br>",
        f"<span style='font-size:15px;font-weight:400;color:#333'>{sub}</span>",
    ]
    if extra_line:
        parts.append(
            f"<br><span style='font-size:13px;font-weight:400;color:#666'>{extra_line}</span>"
        )
    return "".join(parts)


def build_radar_figure(
    rows: list[dict],
    col_mins: np.ndarray | None,
    col_maxs: np.ndarray | None,
    entry_stages: list[str],
    play_stages: list[str],
) -> go.Figure:
    """
    Cartesian radar: each spoke uses global [min, max] for that metric across **all**
    batters (with ≥1 ball) under the selected entry/play stages. Chart batters are
    drawn against those scales.
    """
    if not entry_stages or not play_stages:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=_radar_plot_title_html(
                    entry_stages,
                    play_stages,
                    extra_line="Select at least one entry stage and one play stage.",
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=17),
            ),
            width=RADAR_FIG_WIDTH,
            height=RADAR_FIG_HEIGHT,
            margin=dict(t=120, b=80, l=80, r=80),
        )
        return fig

    if not rows:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=_radar_plot_title_html(
                    entry_stages,
                    play_stages,
                    extra_line="Add batters to see the radar.",
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=17),
            ),
            width=RADAR_FIG_WIDTH,
            height=RADAR_FIG_HEIGHT,
            margin=dict(t=120, b=80, l=80, r=80),
        )
        return fig

    if col_mins is None or col_maxs is None:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=_radar_plot_title_html(
                    entry_stages,
                    play_stages,
                    extra_line="No data for this stage combination (cannot compute bounds).",
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=17),
            ),
            width=RADAR_FIG_WIDTH,
            height=RADAR_FIG_HEIGHT,
            margin=dict(t=120, b=80, l=80, r=80),
        )
        return fig

    M = raw_matrix(rows)
    S = scale_matrix_with_bounds(M, col_mins, col_maxs)
    n = M.shape[1]
    angles = _radar_angles(n)
    R = 1.0
    fig = go.Figure()

    # Spokes and web (relative radius only for geometry; labels show raw)
    for j in range(n):
        c, s = np.cos(angles[j]), np.sin(angles[j])
        fig.add_trace(
            go.Scatter(
                x=[0, R * c],
                y=[0, R * s],
                mode="lines",
                line=dict(color="rgba(180,180,180,0.8)", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for frac in (0.25, 0.5, 0.75):
        wx, wy = [], []
        for j in range(n):
            c, s = np.cos(angles[j]), np.sin(angles[j])
            wx.append(frac * R * c)
            wy.append(frac * R * s)
        wx.append(wx[0])
        wy.append(wy[0])
        fig.add_trace(
            go.Scatter(
                x=wx,
                y=wy,
                mode="lines",
                line=dict(color="rgba(200,200,200,0.5)", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Raw-value tick labels along each spoke (min slightly inset so spokes stay distinct)
    tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ann: list[dict] = []
    lo, hi = col_mins, col_maxs
    for j in range(n):
        c, s = np.cos(angles[j]), np.sin(angles[j])
        span = hi[j] - lo[j]
        for tf in tick_fracs:
            raw_v = lo[j] + tf * span if span > 0 else lo[j]
            # Avoid stacking every "min" at (0,0): place t=0 labels at small radius on each spoke
            r_plot = (0.06 * R if tf == 0 else tf * R) + 0.02 * R
            px = r_plot * c
            py = r_plot * s
            ann.append(
                dict(
                    x=px,
                    y=py,
                    xref="x",
                    yref="y",
                    text=_fmt_tick_raw(j, float(raw_v)),
                    showarrow=False,
                    font=dict(size=11, color="#555"),
                    xanchor="center",
                    yanchor="middle",
                )
            )
    # Metric names outside the chart
    label_r = R * 1.18
    for j in range(n):
        c, s = np.cos(angles[j]), np.sin(angles[j])
        mn = _fmt_tick_raw(j, float(lo[j]))
        mx = _fmt_tick_raw(j, float(hi[j]))
        ann.append(
            dict(
                x=label_r * c,
                y=label_r * s,
                xref="x",
                yref="y",
                text=(
                    f"<b>{RADAR_LABELS[j]}</b><br>"
                    f"<span style='font-size:12px'>scale {mn} … {mx}</span>"
                ),
                showarrow=False,
                font=dict(size=15),
                xanchor="center",
                yanchor="middle",
            )
        )

    # Player polygons (position on spoke = raw value mapped linearly to [0,R])
    for i, r in enumerate(rows):
        raw_row = M[i]
        S_row = S[i]
        xs, ys = [], []
        hover_texts = []
        for j in range(n):
            t = float(S_row[j])
            c, s = np.cos(angles[j]), np.sin(angles[j])
            xs.append(t * R * c)
            ys.append(t * R * s)
            hover_texts.append(
                f"{RADAR_LABELS[j]}: {_fmt_hover_raw(j, float(raw_row[j]))} "
                f"(global range for stages [{lo[j]:g}, {hi[j]:g}])"
            )
        xs.append(xs[0])
        ys.append(ys[0])
        hover_texts.append(hover_texts[0])
        label = r["label_line"]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                fill="toself",
                name=label,
                opacity=0.35,
                line=dict(width=3),
                marker=dict(size=10),
                hovertext=hover_texts,
                hoverinfo="text",
            )
        )

    pad = 1.38
    fig.update_layout(
        xaxis=dict(
            visible=False,
            range=[-pad, pad],
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(visible=False, range=[-pad, pad]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        margin=dict(t=110, b=130, l=90, r=90),
        title=dict(
            text=_radar_plot_title_html(entry_stages, play_stages),
            x=0.5,
            xanchor="center",
            font=dict(size=17),
        ),
        width=RADAR_FIG_WIDTH,
        height=RADAR_FIG_HEIGHT,
        annotations=ann,
        plot_bgcolor="white",
    )
    return fig


_SECTION_CARD_HEAD = "fs-5 fw-semibold mb-0"
_SECTION_HEAD = "fs-5 fw-semibold"
_BODY_HELP = "text-muted small mb-0"


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div(
            ui.div("Batters", class_=f"{_SECTION_HEAD} mb-3"),
            ui.p(
                "Choose entry and play stages in the main panel first. "
                "Then add batters; they all use those stages.",
                class_=f"{_BODY_HELP} mb-3",
            ),
            ui.input_action_button(
                "open_add_modal",
                ui.HTML("&#43; Add batter"),
                class_="btn-primary w-100 mb-4",
            ),
            ui.div("On chart", class_=f"{_SECTION_HEAD} mb-2"),
            ui.output_ui("player_table"),
            ui.input_select(
                "remove_choice",
                "Remove batter",
                choices={},
            ),
            ui.input_action_button(
                "remove_selected",
                "Remove",
                class_="btn-outline-danger w-100 mt-2",
            ),
            class_="px-1",
        ),
    ),
    ui.div(
        ui.card(
            ui.card_header(ui.div("Stages (fixed for all batters)", class_=_SECTION_CARD_HEAD)),
            ui.card_body(
                ui.layout_columns(
                    ui.input_selectize(
                        "entry_stages",
                        "Entry stages",
                        choices=STAGES,
                        multiple=True,
                        selected=["start"],
                        options={"plugins": ["remove_button"]},
                    ),
                    ui.input_selectize(
                        "play_stages",
                        "Play stages",
                        choices=STAGES,
                        multiple=True,
                        selected=["start"],
                        options={"plugins": ["remove_button"]},
                    ),
                    col_widths=(6, 6),
                ),
                ui.p(
                    "Balls must match both: batter entry stage (when they joined the innings) "
                    "and play stage (match phase while facing). Multiple selections include "
                    "all matching balls.",
                    class_=f"{_BODY_HELP} mt-3",
                ),
            ),
            class_="mb-5 shadow-sm border",
        ),
        ui.card(
            ui.card_header(ui.div("Batter Radar", class_=_SECTION_CARD_HEAD)),
            ui.card_body(
                ui.output_ui("radar_headers"),
                ui.div(
                    output_widget("radar_run_exp"),
                    style=(
                        f"width: min(100%, {RADAR_FIG_WIDTH}px); max-width: 100%; "
                        "margin: 0 auto;"
                    ),
                    class_="mt-3",
                ),
            ),
            class_="mb-5 shadow-sm border",
        ),
        ui.card(
            ui.card_header(ui.div("Raw Metrics by Batter", class_=_SECTION_CARD_HEAD)),
            ui.card_body(
                ui.output_ui("stage_caption"),
                ui.div(
                    ui.output_table(
                        "metrics_table",
                        class_="table table-sm table-striped table-hover align-middle",
                    ),
                    class_="mt-2",
                ),
            ),
            class_="shadow-sm border",
        ),
        class_="px-3 px-lg-4 py-3",
    ),
)


def server(input, output, session):
    batters = reactive.Value([])

    @reactive.effect
    @reactive.event(input.open_add_modal)
    def _show_modal():
        m = ui.modal(
            ui.div(
                ui.input_selectize(
                    "add_batter",
                    "Search batter",
                    choices=_BATTERS,
                    selected=_BATTERS[0],
                    multiple=False,
                    options={"plugins": ["remove_button"]},
                    width="100%",
                ),
                class_="mb-4",
            ),
            ui.div(
                ui.input_action_button(
                    "confirm_add",
                    "Add to chart",
                    class_="btn-primary",
                ),
                class_="d-grid",
            ),
            title="Add Batter",
            size="l",
            easy_close=True,
            footer=ui.modal_button("Close", class_="btn-outline-secondary"),
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.confirm_add)
    def _confirm_add():
        b = input.add_batter()
        if not b:
            return
        cur = list(batters())
        cur.append(b)
        batters.set(cur)
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.remove_selected)
    def _remove():
        key = input.remove_choice()
        if not key or key == "":
            return
        cur = list(batters())
        try:
            idx = int(key)
            if 0 <= idx < len(cur):
                cur.pop(idx)
                batters.set(cur)
        except ValueError:
            pass

    @reactive.effect
    def _sync_remove_choices():
        cur = batters()
        if not cur:
            ui.update_select(
                "remove_choice",
                choices={"": "(no batters)"},
                selected="",
            )
        else:
            choices = {str(i): name for i, name in enumerate(cur)}
            ui.update_select("remove_choice", choices=choices, selected=None)

    @render.ui
    def player_table():
        cur = batters()
        if not cur:
            return ui.p(
                "No batters yet. Click + Add batter.",
                class_=f"{_BODY_HELP} fst-italic",
            )
        rows = []
        for name in cur:
            rows.append(ui.div(ui.tags.b(name), class_="mb-1 fs-6"))
        return ui.TagList(*rows)

    @reactive.calc
    def radar_rows():
        entry = _stage_list(input.entry_stages())
        play = _stage_list(input.play_stages())
        if not entry or not play:
            return []
        out = []
        for b in batters():
            sub = filter_batter_stages(_df, b, entry, play)
            m = compute_metrics(sub)
            out.append(
                {
                    **m,
                    "batter": b,
                    "label_line": f"{b} — {m['balls']:,} balls, {m['matches']:,} matches",
                }
            )
        return out

    @render.ui
    def radar_headers():
        entry = _stage_list(input.entry_stages())
        play = _stage_list(input.play_stages())
        cur = batters()
        if not entry or not play:
            return ui.div(
                ui.p(
                    "Select at least one entry stage and one play stage above.",
                    class_="text-warning",
                )
            )
        if not cur:
            return ui.div()
        blocks = []
        for b in cur:
            sub = filter_batter_stages(_df, b, entry, play)
            m = compute_metrics(sub)
            blocks.append(
                ui.div(
                    f"{b} — {m['balls']:,} balls, {m['matches']:,} matches",
                    class_="fs-6 mb-2",
                )
            )
        return ui.div(*blocks)

    @render.ui
    def stage_caption():
        entry = _stage_list(input.entry_stages())
        play = _stage_list(input.play_stages())
        return ui.p(
            f"Filtered by entry stages {entry or '—'} and play stages {play or '—'}.",
            class_=f"{_BODY_HELP}",
        )

    @reactive.calc
    def stage_global_bounds() -> tuple[np.ndarray, np.ndarray] | None:
        entry = _stage_list(input.entry_stages())
        play = _stage_list(input.play_stages())
        if not entry or not play:
            return None
        return global_metric_bounds(_df, entry, play)

    @render_plotly
    def radar_run_exp():
        entry = _stage_list(input.entry_stages())
        play = _stage_list(input.play_stages())
        rows = radar_rows()
        bnds = stage_global_bounds()
        if bnds is None:
            return build_radar_figure(rows, None, None, entry, play)
        mins, maxs = bnds
        return build_radar_figure(rows, mins, maxs, entry, play)

    @render.table
    def metrics_table():
        rows = radar_rows()
        if not rows:
            return pd.DataFrame()
        data = []
        for r in rows:
            data.append(
                {
                    "Batter": r["batter"],
                    "Balls": r["balls"],
                    "Matches": r["matches"],
                    "Dismissals": r["dismissals"],
                    "Avg runs / dismissal": round(r["avg_runs"], 2),
                    "Strike rate": round(r["strike_rate"], 2),
                    "Mean RVA": round(r["rva"], 4),
                    "% Dots": round(r["pct_dots"], 2),
                    "% 1s & 2s": round(r["pct_12"], 2),
                    "% 4s & 6s": round(r["pct_46"], 2),
                }
            )
        return pd.DataFrame(data)


app = App(app_ui, server)
