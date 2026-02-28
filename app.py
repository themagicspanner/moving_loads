"""
Beam Load Simulator — Dash application.

Run with:
    python app.py
Then open http://127.0.0.1:8050 in a browser.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html, no_update

from beam_analysis import (
    compute_envelopes,
    solve_simply_supported,
    vehicle_positions_at_offset,
)

# ---------------------------------------------------------------------------
# Preset vehicles
# ---------------------------------------------------------------------------
PRESET_VEHICLES = {
    "Single Axle": {
        "axle_loads": [100],
        "axle_spacings": [],
    },
    "Two-Axle Truck": {
        "axle_loads": [80, 120],
        "axle_spacings": [4.5],
    },
    "Three-Axle Truck": {
        "axle_loads": [60, 100, 100],
        "axle_spacings": [3.6, 1.2],
    },
    "HL-93 (Truck)": {
        "axle_loads": [35, 145, 145],
        "axle_spacings": [4.3, 4.3],
    },
    "Five-Axle Semi": {
        "axle_loads": [50, 80, 80, 90, 90],
        "axle_spacings": [3.6, 1.2, 6.0, 1.2],
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


def _parse_csv_floats(text: str) -> list[float]:
    """Parse a comma-separated string of numbers."""
    if not text or not text.strip():
        return []
    parts = [s.strip() for s in text.split(",") if s.strip()]
    return [float(p) for p in parts]


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
        html.P("Define a vehicle and UDL to see envelopes and critical load positions.",
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
                    html.Label("Preset", style={"fontWeight": "600", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="vehicle-preset",
                        options=[{"label": k, "value": k} for k in PRESET_VEHICLES],
                        value="Three-Axle Truck",
                        clearable=False,
                        style={"marginBottom": "8px"},
                    ),
                    html.Label("Axle loads (kN) — comma separated",
                               style={"fontWeight": "600", "fontSize": "13px"}),
                    dcc.Input(id="axle-loads", type="text", value="60, 100, 100",
                              style=_input_style(), debounce=True),
                    html.Label("Axle spacings (m) — comma separated",
                               style={"fontWeight": "600", "fontSize": "13px",
                                       "marginTop": "6px"}),
                    dcc.Input(id="axle-spacings", type="text", value="3.6, 1.2",
                              style=_input_style(), debounce=True),
                    html.Div(id="vehicle-summary",
                             style={"fontSize": "12px", "color": "#555",
                                    "marginTop": "6px"}),
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
# Callback: populate axle fields from preset dropdown
# ---------------------------------------------------------------------------
@app.callback(
    Output("axle-loads", "value"),
    Output("axle-spacings", "value"),
    Input("vehicle-preset", "value"),
)
def apply_preset(preset_name):
    if preset_name and preset_name in PRESET_VEHICLES:
        v = PRESET_VEHICLES[preset_name]
        return (
            ", ".join(str(x) for x in v["axle_loads"]),
            ", ".join(str(x) for x in v["axle_spacings"]),
        )
    return no_update, no_update


# ---------------------------------------------------------------------------
# Callback: vehicle summary text
# ---------------------------------------------------------------------------
@app.callback(
    Output("vehicle-summary", "children"),
    Input("axle-loads", "value"),
    Input("axle-spacings", "value"),
)
def update_vehicle_summary(loads_text, spacings_text):
    try:
        loads = _parse_csv_floats(loads_text or "")
        spacings = _parse_csv_floats(spacings_text or "")
    except ValueError:
        return "Invalid input"
    if not loads:
        return "No axles defined"
    n = len(loads)
    total = sum(loads)
    length = sum(spacings)
    return f"{n} axle(s) | total load {total:.0f} kN | vehicle length {length:.1f} m"


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
    Input("axle-loads", "value"),
    Input("axle-spacings", "value"),
    Input("n-steps", "value"),
)
def update_graph(span, ei, udl_w, udl_a, udl_b,
                 loads_text, spacings_text, n_steps):
    span = float(span or 20)
    ei = float(ei or 0)
    udl_w = float(udl_w or 0)
    udl_a = float(udl_a or 0)
    udl_b = float(udl_b or span)
    n_steps = int(n_steps or 300)

    try:
        axle_loads = _parse_csv_floats(loads_text or "100")
        axle_spacings = _parse_csv_floats(spacings_text or "")
    except ValueError:
        axle_loads = [100]
        axle_spacings = []

    if len(axle_spacings) < len(axle_loads) - 1:
        axle_spacings += [0.0] * (len(axle_loads) - 1 - len(axle_spacings))
    axle_spacings = axle_spacings[: len(axle_loads) - 1]

    udl_segments = []
    if udl_w > 0 and udl_a < udl_b:
        udl_segments.append((max(udl_a, 0), min(udl_b, span), udl_w))

    n_pts = 501

    # --- Compute envelopes + critical positions ---
    (x, s_max, s_min, m_max, m_min, d_max, d_min,
     crit_shear_fx, crit_moment_fx) = compute_envelopes(
        span, axle_spacings, axle_loads, udl_segments,
        n_points=n_pts, n_steps=n_steps,
        EI=ei if ei > 0 else None,
    )

    # --- Solve the two critical load cases ---
    def _solve_at(front_x):
        axles = vehicle_positions_at_offset(axle_spacings, axle_loads, front_x)
        on_beam = [(p, P) for p, P in axles if 0 <= p <= span]
        return axles, solve_simply_supported(
            span, on_beam, udl_segments, n_points=n_pts,
            EI=ei if ei > 0 else None,
        )

    shear_axles, (_, shear_v, shear_m, _, shear_RA, shear_RB) = _solve_at(crit_shear_fx)
    moment_axles, (_, moment_v, moment_m, _, moment_RA, moment_RB) = _solve_at(crit_moment_fx)

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

    fig.add_trace(go.Scatter(
        x=x, y=moment_m, mode="lines", line=dict(color=CLR_MOMENT, width=2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)", name="M (critical)",
        hovertemplate="x=%{x:.2f} m<br>M=%{y:.1f} kN·m<extra></extra>",
        showlegend=False,
    ), row=5, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=5, col=1)
    _annotate_peak(fig, x, moment_m, "M_max", CLR_MOMENT, "kN·m", row=5)

    # ================================================================
    # MOMENT ENVELOPE — row 6
    # ================================================================
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([m_max, m_min[::-1]]),
        fill="toself", fillcolor=CLR_MOMENT_FILL,
        line=dict(width=0), hoverinfo="skip",
        name="Moment envelope", showlegend=True,
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=m_max, mode="lines", line=dict(color=CLR_MOMENT, width=2),
        name="M_max",
        hovertemplate="x=%{x:.2f} m<br>M_max=%{y:.1f} kN·m<extra></extra>",
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=m_min, mode="lines", line=dict(color=CLR_MOMENT_MIN, width=2),
        name="M_min",
        hovertemplate="x=%{x:.2f} m<br>M_min=%{y:.1f} kN·m<extra></extra>",
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


def _annotate_peak(fig, x, y, label, colour, unit, row, use_min=False):
    """Add an annotation at the peak (max or min) of a curve."""
    if use_min:
        idx = int(np.argmin(y))
    else:
        idx = int(np.argmax(y))
    val = y[idx]
    xref = f"x{row}" if row > 1 else "x"
    yref = f"y{row}" if row > 1 else "y"
    fig.add_annotation(
        x=x[idx], y=val,
        text=f"{label} = {val:.1f} {unit} @ x = {x[idx]:.2f} m",
        showarrow=True, arrowhead=2, arrowcolor=colour,
        font=dict(size=10, color=colour),
        bgcolor="white", bordercolor=colour, borderwidth=1, borderpad=3,
        ax=0, ay=-30,
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
