"""
Beam Load Simulator — Dash application.

Run with:
    python app.py
Then open http://127.0.0.1:8050 in a browser.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html, no_update

from beam_analysis import (
    compute_envelopes,
    solve_simply_supported,
    vehicle_positions_at_offset,
)

# ---------------------------------------------------------------------------
# Preset vehicles — EN 1991-2 Traffic load models
# ---------------------------------------------------------------------------
# LM1 Tandem System characteristic axle loads (EN 1991-2 Table 4.2, α = 1.0).
# LM1 also requires a UDL component (9 kN/m² Lane 1, 2.5 kN/m² Lanes 2-3);
# multiply by the notional lane width (typically 3.0 m) and apply via the UDL
# input below.
#
# LM3 SV80, SV100, SV196 axle layouts match UK NA to BS EN 1991-2.
# The variable inter-bogie gap (1.2, 5.0, 9.0 m) is automatically swept
# and the critical (worst-case) gap is selected for each span.
PRESET_VEHICLES = {
    # ── EN 1991-2 §4.3.2  Load Model 1 – Tandem System ─────────────────
    "LM1 – Lane 1 Tandem  (2 × 300 kN)": {
        "axle_loads": [300, 300],
        "axle_spacings": [1.2],
    },
    "LM1 – Lane 2 Tandem  (2 × 200 kN)": {
        "axle_loads": [200, 200],
        "axle_spacings": [1.2],
    },
    "LM1 – Lane 3 Tandem  (2 × 100 kN)": {
        "axle_loads": [100, 100],
        "axle_spacings": [1.2],
    },
    # ── EN 1991-2 §4.3.3  Load Model 2 – Single Axle ───────────────────
    "LM2 – Single Axle  (400 kN)": {
        "axle_loads": [400],
        "axle_spacings": [],
    },
    # ── EN 1991-2 §4.3.4 / UK NA  Load Model 3 – Special Vehicles ───────
    # SV80 per UK NA: 6 axles of 130 kN (780 kN total), two groups of 3
    # at 1.2 m; variable gap between groups auto-selected from 1.2/5.0/9.0 m.
    "LM3 – SV80  (6 × 130 kN)": {
        "axle_loads": [130, 130, 130, 130, 130, 130],
        "axle_spacings": [1.2, 1.2, 1.2, 1.2, 1.2],
        "variable_gap": {"index": 2, "values": [1.2, 5.0, 9.0]},
    },
    # SV100 per UK NA: 6 axles of 165 kN (990 kN total), same layout.
    "LM3 – SV100  (6 × 165 kN)": {
        "axle_loads": [165, 165, 165, 165, 165, 165],
        "axle_spacings": [1.2, 1.2, 1.2, 1.2, 1.2],
        "variable_gap": {"index": 2, "values": [1.2, 5.0, 9.0]},
    },
    # SV196 per UK NA: 8 × 165 kN trailer + 180, 180, 100 kN tractor (1780 kN).
    # Variable gap between the two 4-axle trailer bogies.
    "LM3 – SV196  (1780 kN)": {
        "axle_loads": [165, 165, 165, 165, 165, 165, 165, 165, 180, 180, 100],
        "axle_spacings": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 4.0, 1.6, 4.4],
        "variable_gap": {"index": 3, "values": [1.2, 5.0, 9.0]},
    },
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
CLR_BEAM = "#4a4a4a"
CLR_SHEAR = "#1f77b4"
CLR_SHEAR_MIN = "#aec7e8"
CLR_MOMENT = "#d62728"
CLR_MOMENT_MIN = "#f5a0a0"
CLR_DEFL = "#2ca02c"
CLR_DEFL_MIN = "#98df8a"
CLR_AXLE = "#e377c2"
CLR_UDL = "rgba(255, 165, 0, 0.35)"
CLR_UDL_LINE = "rgb(255, 140, 0)"
CLR_REACTION = "#17becf"
CLR_BG = "#fafafa"
CLR_ENVELOPE_FILL = "rgba(31,119,180,0.10)"
CLR_MOMENT_FILL = "rgba(214,39,40,0.10)"
CLR_DEFL_FILL = "rgba(44,160,44,0.10)"
CLR_CRITICAL = "rgba(0,0,0,0.08)"


# ---------------------------------------------------------------------------
# Layout helper functions (must be defined before app.layout)
# ---------------------------------------------------------------------------

def _input_style():
    return {"width": "100%", "padding": "4px 8px", "borderRadius": "4px",
            "border": "1px solid #ccc", "boxSizing": "border-box"}


def _section(title, children):
    return html.Div(
        style={"background": "#fff", "border": "1px solid #e0e0e0",
               "borderRadius": "8px", "padding": "12px 14px",
               "marginBottom": "12px"},
        children=[html.H4(title, style={"margin": "0 0 8px 0", "fontSize": "15px"})] + children,
    )


def _labelled_input(label, id_, value, mn, mx, step, tooltip=None):
    children = [
        html.Label(label, style={"fontWeight": "600", "fontSize": "13px"},
                    title=tooltip),
        dcc.Input(id=id_, type="number", value=value, min=mn, max=mx, step=step,
                  style=_input_style(), debounce=True),
    ]
    return html.Div(children, style={"marginBottom": "6px"})



# ---------------------------------------------------------------------------
# App & layout
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Beam Load Simulator"

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, sans-serif", "margin": "0 auto",
           "maxWidth": "1400px", "padding": "20px"},
    children=[
        html.H2("Simply Supported Beam — Moving Load Envelope",
                 style={"textAlign": "center", "marginBottom": "4px"}),
        html.P("EN 1991-2 load models — define a vehicle and UDL to see envelopes and critical load positions.",
               style={"textAlign": "center", "color": "#666", "marginTop": "0"}),

        html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[

            # ---- LEFT PANEL ----
            html.Div(style={"flex": "0 0 340px", "minWidth": "300px"}, children=[

                _section("Beam", [
                    _labelled_input("Span (m)", "beam-span", 20, 1, 200, 0.5),
                    _labelled_input("EI (kN·m²)", "beam-ei", 1e6, 1e3, 1e12, 1e3,
                                    tooltip="Flexural rigidity — set to 0 to hide deflection"),
                ]),

                _section("UDL (uniformly distributed load)", [
                    _labelled_input("Intensity (kN/m)", "udl-w", 0, 0, 500, 0.5),
                    _labelled_input("Start (m)", "udl-start", 0, 0, 200, 0.5),
                    _labelled_input("End (m)", "udl-end", 20, 0, 200, 0.5),
                ]),

                _section("Vehicle", [
                    html.Label("Load model", style={"fontWeight": "600", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="vehicle-preset",
                        options=[{"label": k, "value": k} for k in PRESET_VEHICLES],
                        value="LM1 – Lane 1 Tandem  (2 × 300 kN)",
                        clearable=False,
                        style={"marginBottom": "8px"},
                    ),
                    html.Div(id="vehicle-summary",
                             style={"fontSize": "12px", "color": "#555",
                                    "marginTop": "6px"}),
                    html.Div(
                        "LM1 also requires a UDL: Lane 1 → 27 kN/m, "
                        "Lanes 2-3 → 7.5 kN/m (9 or 2.5 kN/m² × 3 m lane width). "
                        "Apply via the UDL panel above.",
                        id="lm1-udl-hint",
                        style={"fontSize": "11px", "color": "#888",
                               "marginTop": "4px", "fontStyle": "italic"},
                    ),
                ]),

                _section("Envelope resolution", [
                    _labelled_input("Vehicle positions", "n-steps", 300, 50, 2000, 50,
                                    tooltip="Number of vehicle positions used to build the envelope"),
                ]),
            ]),

            # ---- RIGHT PANEL ----
            html.Div(style={"flex": "1 1 700px", "minWidth": "500px"}, children=[
                dcc.Loading(
                    dcc.Graph(id="main-graph", style={"height": "90vh"},
                              config={"displayModeBar": False}),
                    type="circle",
                ),
            ]),
        ]),
    ],
)


# ---------------------------------------------------------------------------
# Callback: vehicle summary + LM1 hint visibility
# ---------------------------------------------------------------------------
@app.callback(
    Output("vehicle-summary", "children"),
    Output("lm1-udl-hint", "style"),
    Input("vehicle-preset", "value"),
)
def update_vehicle_info(preset_name):
    _hidden = {"display": "none"}
    _visible = {"fontSize": "11px", "color": "#888",
                "marginTop": "4px", "fontStyle": "italic"}
    if not preset_name or preset_name not in PRESET_VEHICLES:
        return "No vehicle selected", _hidden
    v = PRESET_VEHICLES[preset_name]
    loads = v["axle_loads"]
    spacings = v["axle_spacings"]
    n = len(loads)
    total = sum(loads)
    length = sum(spacings)
    vgap = v.get("variable_gap")
    summary = f"{n} axle(s) | {total:.0f} kN total | {length:.1f} m long"
    if vgap:
        gaps = ", ".join(str(g) for g in vgap["values"])
        summary += f" | auto-gap [{gaps}] m"
    show_hint = _visible if preset_name.startswith("LM1") else _hidden
    return summary, show_hint


# ---------------------------------------------------------------------------
# Callback: main graph
# ---------------------------------------------------------------------------
@app.callback(
    Output("main-graph", "figure"),
    Input("beam-span", "value"),
    Input("beam-ei", "value"),
    Input("udl-w", "value"),
    Input("udl-start", "value"),
    Input("udl-end", "value"),
    Input("vehicle-preset", "value"),
    Input("n-steps", "value"),
)
def update_graph(span, ei, udl_w, udl_a, udl_b,
                 preset_name, n_steps):
    span = float(span or 20)
    ei = float(ei or 0)
    udl_w = float(udl_w or 0)
    udl_a = float(udl_a or 0)
    udl_b = float(udl_b or span)
    n_steps = int(n_steps or 300)

    preset = PRESET_VEHICLES.get(preset_name) if preset_name else None
    if preset:
        axle_loads = list(preset["axle_loads"])
        axle_spacings = list(preset["axle_spacings"])
    else:
        axle_loads = [100]
        axle_spacings = []

    udl_segments = []
    if udl_w > 0 and udl_a < udl_b:
        udl_segments.append((max(udl_a, 0), min(udl_b, span), udl_w))

    n_pts = 501
    EI_val = ei if ei > 0 else None

    # --- Build list of spacing variants to sweep ---
    # For SV vehicles with a variable inter-bogie gap, run all gap values
    # and keep the worst-case envelope.
    vgap = preset.get("variable_gap") if preset else None

    spacing_variants = []
    if vgap is not None:
        gap_idx = vgap["index"]
        for gap in vgap["values"]:
            variant = list(axle_spacings)
            variant[gap_idx] = gap
            spacing_variants.append(variant)
    else:
        spacing_variants.append(list(axle_spacings))

    # --- Compute envelopes (worst-case across all gap variants) ---
    x = s_max = s_min = m_max = m_min = d_max = d_min = None
    crit_shear_fx = crit_moment_fx = 0.0
    best_shear_spacings = best_moment_spacings = axle_spacings
    best_shear_abs = 0.0
    best_moment_val = 0.0

    for sp in spacing_variants:
        (xi, sv_max, sv_min, mv_max, mv_min, dv_max, dv_min,
         cs_fx, cm_fx) = compute_envelopes(
            span, sp, axle_loads, udl_segments,
            n_points=n_pts, n_steps=n_steps, EI=EI_val,
        )
        if x is None:
            x = xi
            s_max, s_min = sv_max, sv_min
            m_max, m_min = mv_max, mv_min
            d_max, d_min = dv_max, dv_min
            crit_shear_fx, crit_moment_fx = cs_fx, cm_fx
            best_shear_spacings = best_moment_spacings = sp
            best_shear_abs = max(abs(sv_max.max()), abs(sv_min.min()))
            best_moment_val = mv_max.max()
        else:
            # Element-wise worst-case envelopes
            s_max = np.maximum(s_max, sv_max)
            s_min = np.minimum(s_min, sv_min)
            m_max = np.maximum(m_max, mv_max)
            m_min = np.minimum(m_min, mv_min)
            if dv_max is not None and d_max is not None:
                d_max = np.maximum(d_max, dv_max)
                d_min = np.minimum(d_min, dv_min)
            # Track which gap produced the worst shear / moment
            shear_abs = max(abs(sv_max.max()), abs(sv_min.min()))
            if shear_abs > best_shear_abs:
                best_shear_abs = shear_abs
                crit_shear_fx = cs_fx
                best_shear_spacings = sp
            moment_val = mv_max.max()
            if moment_val > best_moment_val:
                best_moment_val = moment_val
                crit_moment_fx = cm_fx
                best_moment_spacings = sp

    # --- Solve the two critical load cases ---
    def _solve_at(front_x, spacings):
        axles = vehicle_positions_at_offset(spacings, axle_loads, front_x)
        on_beam = [(p, P) for p, P in axles if 0 <= p <= span]
        return axles, solve_simply_supported(
            span, on_beam, udl_segments, n_points=n_pts, EI=EI_val,
        )

    shear_axles, (_, shear_v, shear_m, _, shear_RA, shear_RB) = \
        _solve_at(crit_shear_fx, best_shear_spacings)
    moment_axles, (_, moment_v, moment_m, _, moment_RA, moment_RB) = \
        _solve_at(crit_moment_fx, best_moment_spacings)

    # --- Build figure ---
    show_deflection = ei > 0
    # Rows: beam@shear, SFD, shear envelope, beam@moment, BMD, moment envelope
    #        + optionally deflection envelope
    n_rows = 7 if show_deflection else 6
    row_titles = [
        "Vehicle position for max shear",
        "Shear force diagram (critical case)",
        "Shear force envelope",
        "Vehicle position for max moment",
        "Bending moment diagram (critical case)",
        "Bending moment envelope",
    ]
    if show_deflection:
        row_titles.append("Deflection envelope (mm)")

    heights = [0.12, 0.15, 0.15, 0.12, 0.15, 0.15, 0.16] if show_deflection \
        else [0.13, 0.17, 0.17, 0.13, 0.20, 0.20]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=heights,
        subplot_titles=row_titles,
        vertical_spacing=0.04,
    )

    # ================================================================
    # SHEAR CRITICAL CASE — rows 1 & 2
    # ================================================================
    _draw_beam_with_vehicle(fig, row=1, span=span,
                            all_axles=shear_axles,
                            udl_a=udl_a, udl_b=udl_b, udl_w=udl_w,
                            R_A=shear_RA, R_B=shear_RB)

    fig.add_trace(go.Scatter(
        x=x, y=shear_v, mode="lines", line=dict(color=CLR_SHEAR, width=2),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.15)", name="V (critical)",
        hovertemplate="x=%{x:.2f} m<br>V=%{y:.1f} kN<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=2, col=1)
    _annotate_peak(fig, x, shear_v, "V_max", CLR_SHEAR, "kN", row=2)
    _annotate_peak(fig, x, shear_v, "V_min", CLR_SHEAR_MIN, "kN", row=2, use_min=True)

    # ================================================================
    # SHEAR ENVELOPE — row 3
    # ================================================================
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([s_max, s_min[::-1]]),
        fill="toself", fillcolor=CLR_ENVELOPE_FILL,
        line=dict(width=0), hoverinfo="skip",
        name="Shear envelope", showlegend=True,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=s_max, mode="lines", line=dict(color=CLR_SHEAR, width=2),
        name="V_max",
        hovertemplate="x=%{x:.2f} m<br>V_max=%{y:.1f} kN<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=s_min, mode="lines", line=dict(color=CLR_SHEAR_MIN, width=2),
        name="V_min",
        hovertemplate="x=%{x:.2f} m<br>V_min=%{y:.1f} kN<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=3, col=1)

    # ================================================================
    # MOMENT CRITICAL CASE — rows 4 & 5
    # ================================================================
    _draw_beam_with_vehicle(fig, row=4, span=span,
                            all_axles=moment_axles,
                            udl_a=udl_a, udl_b=udl_b, udl_w=udl_w,
                            R_A=moment_RA, R_B=moment_RB)

    # Negate moments for UK convention (sagging drawn below axis)
    fig.add_trace(go.Scatter(
        x=x, y=-moment_m, mode="lines", line=dict(color=CLR_MOMENT, width=2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)", name="M (critical)",
        customdata=moment_m,
        hovertemplate="x=%{x:.2f} m<br>M=%{customdata:.1f} kN·m<extra></extra>",
        showlegend=False,
    ), row=5, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=5, col=1)
    _annotate_peak(fig, x, -moment_m, "M_max", CLR_MOMENT, "kN·m", row=5,
                   use_min=True, negate_label=True)

    # ================================================================
    # MOMENT ENVELOPE — row 6
    # ================================================================
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([-m_max, -m_min[::-1]]),
        fill="toself", fillcolor=CLR_MOMENT_FILL,
        line=dict(width=0), hoverinfo="skip",
        name="Moment envelope", showlegend=True,
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=-m_max, mode="lines", line=dict(color=CLR_MOMENT, width=2),
        name="M_max", customdata=m_max,
        hovertemplate="x=%{x:.2f} m<br>M_max=%{customdata:.1f} kN·m<extra></extra>",
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=-m_min, mode="lines", line=dict(color=CLR_MOMENT_MIN, width=2),
        name="M_min", customdata=m_min,
        hovertemplate="x=%{x:.2f} m<br>M_min=%{customdata:.1f} kN·m<extra></extra>",
    ), row=6, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=6, col=1)

    # ================================================================
    # DEFLECTION ENVELOPE — row 7 (optional)
    # ================================================================
    if show_deflection:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([d_max * 1000, d_min[::-1] * 1000]),
            fill="toself", fillcolor=CLR_DEFL_FILL,
            line=dict(width=0), hoverinfo="skip",
            name="Deflection envelope", showlegend=True,
        ), row=7, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=d_max * 1000, mode="lines",
            line=dict(color=CLR_DEFL, width=2), name="d_max",
            hovertemplate="x=%{x:.2f} m<br>d_max=%{y:.3f} mm<extra></extra>",
        ), row=7, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=d_min * 1000, mode="lines",
            line=dict(color=CLR_DEFL_MIN, width=2), name="d_min",
            hovertemplate="x=%{x:.2f} m<br>d_min=%{y:.3f} mm<extra></extra>",
        ), row=7, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=7, col=1)
        fig.update_yaxes(title_text="mm", row=7, col=1)

    # ---- Axis labels ----
    fig.update_yaxes(title_text="kN", row=2, col=1)
    fig.update_yaxes(title_text="kN", row=3, col=1)
    fig.update_yaxes(title_text="kN·m", row=5, col=1)
    fig.update_yaxes(title_text="kN·m", row=6, col=1)
    fig.update_xaxes(title_text="Position along beam (m)", row=n_rows, col=1)

    # ---- Global layout ----
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor=CLR_BG,
        paper_bgcolor="#fff",
        font=dict(size=12),
    )

    for i in range(1, n_rows + 1):
        fig.update_xaxes(range=[0, span], row=i, col=1)

    return fig


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_beam_with_vehicle(fig, row, span, all_axles,
                            udl_a, udl_b, udl_w, R_A, R_B):
    """Draw beam, supports, UDL, vehicle axles and reactions on a given row."""
    beam_y = 0
    xref = f"x{row}" if row > 1 else "x"
    yref = f"y{row}" if row > 1 else "y"

    # Beam line
    fig.add_trace(go.Scatter(
        x=[0, span], y=[beam_y, beam_y], mode="lines",
        line=dict(color=CLR_BEAM, width=6), hoverinfo="skip",
        showlegend=False,
    ), row=row, col=1)

    # Supports
    _draw_triangle(fig, 0, beam_y, size=0.6, span=span, row=row)
    _draw_triangle(fig, span, beam_y, size=0.6, span=span, roller=True, row=row)

    # UDL
    if udl_w > 0 and udl_a < udl_b:
        a = max(udl_a, 0)
        b = min(udl_b, span)
        arrow_h = _arrow_height(udl_w, span)
        n_arrows = max(int((b - a) / (span / 30)), 3)
        xs = np.linspace(a, b, n_arrows)

        fig.add_trace(go.Scatter(
            x=[a, a, b, b],
            y=[beam_y, beam_y + arrow_h, beam_y + arrow_h, beam_y],
            fill="toself", fillcolor=CLR_UDL,
            line=dict(color=CLR_UDL_LINE, width=1),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)

        for xi in xs:
            fig.add_annotation(
                x=xi, y=beam_y, ax=xi, ay=beam_y + arrow_h,
                xref=xref, yref=yref, axref=xref, ayref=yref,
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5,
                arrowcolor=CLR_UDL_LINE,
            )

    # Axle loads
    for pos, P in all_axles:
        on = 0 <= pos <= span
        colour = CLR_AXLE if on else "rgba(200,200,200,0.5)"
        arrow_h = _arrow_height(P, span)
        fig.add_annotation(
            x=pos, y=beam_y, ax=pos, ay=beam_y + arrow_h,
            xref=xref, yref=yref, axref=xref, ayref=yref,
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor=colour,
        )
        fig.add_annotation(
            x=pos, y=beam_y + arrow_h + _arrow_height(10, span) * 0.2,
            text=f"{P:.0f} kN", showarrow=False,
            font=dict(size=10, color=colour), xref=xref, yref=yref,
        )
        fig.add_trace(go.Scatter(
            x=[pos], y=[beam_y], mode="markers",
            marker=dict(size=8, color=colour, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)

    # Reaction labels
    r_offset = _arrow_height(50, span) * 0.5
    fig.add_annotation(
        x=0, y=beam_y - r_offset,
        text=f"R_A = {R_A:.1f} kN", showarrow=False,
        font=dict(size=11, color=CLR_REACTION), xref=xref, yref=yref,
    )
    fig.add_annotation(
        x=span, y=beam_y - r_offset,
        text=f"R_B = {R_B:.1f} kN", showarrow=False,
        font=dict(size=11, color=CLR_REACTION), xref=xref, yref=yref,
    )

    fig.update_yaxes(
        visible=False, range=[-2, 5], fixedrange=True, row=row, col=1,
    )


def _annotate_peak(fig, x, y, label, colour, unit, row,
                   use_min=False, negate_label=False):
    """Add an annotation at the peak (max or min) of a curve.

    If negate_label is True the displayed value is -val (useful when the
    plotted y-data has been negated for the UK moment convention but the
    label should show the true positive value).
    """
    if use_min:
        idx = int(np.argmin(y))
    else:
        idx = int(np.argmax(y))
    val = y[idx]
    display_val = -val if negate_label else val
    xref = f"x{row}" if row > 1 else "x"
    yref = f"y{row}" if row > 1 else "y"
    fig.add_annotation(
        x=x[idx], y=val,
        text=f"{label} = {display_val:.1f} {unit} @ x = {x[idx]:.2f} m",
        showarrow=True, arrowhead=2, arrowcolor=colour,
        font=dict(size=10, color=colour),
        bgcolor="white", bordercolor=colour, borderwidth=1, borderpad=3,
        ax=0, ay=30,
        xref=xref, yref=yref,
    )


def _draw_triangle(fig, x0, y0, size, span, roller=False, row=1):
    s = size * span / 20
    h = s * 1.2
    xs = [x0 - s / 2, x0, x0 + s / 2, x0 - s / 2]
    ys = [y0 - h, y0, y0 - h, y0 - h]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor="rgba(100,100,100,0.3)", line=dict(color=CLR_BEAM, width=1.5),
        hoverinfo="skip", showlegend=False,
    ), row=row, col=1)
    if roller:
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0 - h - s * 0.25], mode="markers",
            marker=dict(size=6, color=CLR_BEAM, symbol="circle-open", line_width=1.5),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=1)


def _arrow_height(load, span):
    return max(0.5, min(3.5, 1.0 + 2.0 * abs(load) / 200)) * span / 20


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
