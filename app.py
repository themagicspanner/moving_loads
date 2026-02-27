"""
Beam Load Simulator — Dash application.

Run with:
    python app.py
Then open http://127.0.0.1:8050 in a browser.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

from beam_analysis import solve_simply_supported, vehicle_positions_at_offset

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
CLR_MOMENT = "#d62728"
CLR_DEFL = "#2ca02c"
CLR_AXLE = "#e377c2"
CLR_UDL = "rgba(255, 165, 0, 0.35)"
CLR_UDL_LINE = "rgb(255, 140, 0)"
CLR_REACTION = "#17becf"
CLR_BG = "#fafafa"


# ---------------------------------------------------------------------------
# Layout helper functions (must be defined before app.layout)
# ---------------------------------------------------------------------------

def _input_style():
    return {"width": "100%", "padding": "4px 8px", "borderRadius": "4px",
            "border": "1px solid #ccc", "boxSizing": "border-box"}


def _btn_style(colour):
    return {"padding": "6px 16px", "border": "none", "borderRadius": "4px",
            "background": colour, "color": "#fff", "cursor": "pointer",
            "fontWeight": "600", "fontSize": "13px"}


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
        html.H2("Simply Supported Beam — Moving Load Simulator",
                 style={"textAlign": "center", "marginBottom": "4px"}),
        html.P("Define a vehicle and UDL, then animate it crossing the beam.",
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

                _section("Animation", [
                    _labelled_input("Speed (steps/s)", "anim-speed", 20, 1, 60, 1),
                    html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px"},
                             children=[
                                 html.Button("▶  Play", id="btn-play", n_clicks=0,
                                             style=_btn_style("#2196F3")),
                                 html.Button("⏸  Pause", id="btn-pause", n_clicks=0,
                                             style=_btn_style("#FF9800")),
                                 html.Button("⏮  Reset", id="btn-reset", n_clicks=0,
                                             style=_btn_style("#9E9E9E")),
                             ]),
                ]),

                html.Div(style={"marginTop": "12px"}, children=[
                    html.Label("Vehicle front-axle position",
                               style={"fontWeight": "600", "fontSize": "13px"}),
                    dcc.Slider(id="position-slider", min=-20, max=40, step=0.1,
                               value=-10, marks=None,
                               tooltip={"placement": "bottom", "always_visible": True}),
                ]),
            ]),

            # ---- RIGHT PANEL ----
            html.Div(style={"flex": "1 1 700px", "minWidth": "500px"}, children=[
                dcc.Graph(id="main-graph", style={"height": "85vh"},
                          config={"displayModeBar": False}),
            ]),
        ]),

        dcc.Interval(id="anim-interval", interval=50, n_intervals=0, disabled=True),
        dcc.Store(id="anim-state", data={"playing": False, "position": -10}),
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
# Callback: update slider range when span or vehicle changes
# ---------------------------------------------------------------------------
@app.callback(
    Output("position-slider", "min"),
    Output("position-slider", "max"),
    Input("beam-span", "value"),
    Input("axle-spacings", "value"),
)
def update_slider_range(span, spacings_text):
    span = float(span or 20)
    try:
        spacings = _parse_csv_floats(spacings_text or "")
    except ValueError:
        spacings = []
    vehicle_length = sum(spacings)
    margin = max(vehicle_length + 2, 5)
    return -margin, span + margin


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
# Callbacks: play / pause / reset — control the Interval
# ---------------------------------------------------------------------------
@app.callback(
    Output("anim-interval", "disabled"),
    Output("anim-interval", "interval"),
    Output("anim-state", "data"),
    Output("position-slider", "value"),
    Input("btn-play", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("anim-interval", "n_intervals"),
    State("anim-state", "data"),
    State("anim-speed", "value"),
    State("position-slider", "min"),
    State("position-slider", "max"),
    State("position-slider", "value"),
    State("beam-span", "value"),
    prevent_initial_call=True,
)
def control_animation(play_clicks, pause_clicks, reset_clicks,
                      n_intervals, anim_data, speed,
                      slider_min, slider_max, slider_val, span):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    speed = int(speed or 20)
    interval_ms = max(int(1000 / speed), 16)
    span = float(span or 20)
    step = span / 200

    if trigger == "btn-play":
        pos = anim_data.get("position", slider_val)
        if pos >= slider_max:
            pos = slider_min
        return False, interval_ms, {"playing": True, "position": pos}, pos

    if trigger == "btn-pause":
        return True, interval_ms, {"playing": False,
                                   "position": anim_data.get("position", slider_val)}, no_update

    if trigger == "btn-reset":
        return True, interval_ms, {"playing": False, "position": slider_min}, slider_min

    if trigger == "anim-interval":
        pos = anim_data.get("position", slider_min) + step
        if pos > slider_max:
            return True, interval_ms, {"playing": False, "position": slider_max}, slider_max
        return False, interval_ms, {"playing": True, "position": pos}, pos

    return no_update, no_update, no_update, no_update


# ---------------------------------------------------------------------------
# Callback: main graph
# ---------------------------------------------------------------------------
@app.callback(
    Output("main-graph", "figure"),
    Input("position-slider", "value"),
    Input("beam-span", "value"),
    Input("beam-ei", "value"),
    Input("udl-w", "value"),
    Input("udl-start", "value"),
    Input("udl-end", "value"),
    Input("axle-loads", "value"),
    Input("axle-spacings", "value"),
)
def update_graph(front_x, span, ei, udl_w, udl_a, udl_b,
                 loads_text, spacings_text):
    span = float(span or 20)
    ei = float(ei or 0)
    udl_w = float(udl_w or 0)
    udl_a = float(udl_a or 0)
    udl_b = float(udl_b or span)
    front_x = float(front_x if front_x is not None else 0)

    try:
        axle_loads = _parse_csv_floats(loads_text or "100")
        axle_spacings = _parse_csv_floats(spacings_text or "")
    except ValueError:
        axle_loads = [100]
        axle_spacings = []

    if len(axle_spacings) < len(axle_loads) - 1:
        axle_spacings += [0.0] * (len(axle_loads) - 1 - len(axle_spacings))
    axle_spacings = axle_spacings[: len(axle_loads) - 1]

    all_axles = vehicle_positions_at_offset(axle_spacings, axle_loads, front_x)
    on_beam = [(pos, P) for pos, P in all_axles if 0 <= pos <= span]

    udl_segments = []
    if udl_w > 0 and udl_a < udl_b:
        udl_segments.append((max(udl_a, 0), min(udl_b, span), udl_w))

    x, shear, moment, deflection, R_A, R_B = solve_simply_supported(
        span, on_beam, udl_segments, n_points=501,
        EI=ei if ei > 0 else None,
    )

    show_deflection = ei > 0
    n_rows = 4 if show_deflection else 3
    row_titles = ["Beam diagram", "Shear force (kN)", "Bending moment (kN·m)"]
    if show_deflection:
        row_titles.append("Deflection (mm)")

    heights = [0.30, 0.23, 0.23, 0.24] if show_deflection else [0.34, 0.33, 0.33]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=heights,
        subplot_titles=row_titles,
        vertical_spacing=0.06,
    )

    _draw_beam_diagram(fig, span, all_axles, on_beam, udl_a, udl_b, udl_w, R_A, R_B)

    fig.add_trace(go.Scatter(
        x=x, y=shear, mode="lines", line=dict(color=CLR_SHEAR, width=2),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.15)", name="Shear",
        hovertemplate="x=%{x:.2f} m<br>V=%{y:.1f} kN<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=moment, mode="lines", line=dict(color=CLR_MOMENT, width=2),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.15)", name="Moment",
        hovertemplate="x=%{x:.2f} m<br>M=%{y:.1f} kN·m<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=3, col=1)

    if show_deflection:
        fig.add_trace(go.Scatter(
            x=x, y=deflection * 1000, mode="lines",
            line=dict(color=CLR_DEFL, width=2),
            fill="tozeroy", fillcolor="rgba(44,160,44,0.15)", name="Deflection",
            hovertemplate="x=%{x:.2f} m<br>δ=%{y:.3f} mm<extra></extra>",
        ), row=4, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#aaa", row=4, col=1)
        fig.update_yaxes(title_text="mm", row=4, col=1)

    fig.update_yaxes(title_text="kN", row=2, col=1)
    fig.update_yaxes(title_text="kN·m", row=3, col=1)
    fig.update_xaxes(title_text="Position along beam (m)", row=n_rows, col=1)

    fig.update_layout(
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=40),
        plot_bgcolor=CLR_BG,
        paper_bgcolor="#fff",
        font=dict(size=12),
    )

    x_margin = max(sum(axle_spacings) + 3, 5)
    for i in range(1, n_rows + 1):
        fig.update_xaxes(range=[-x_margin, span + x_margin], row=i, col=1)

    return fig


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_beam_diagram(fig, span, all_axles, on_beam, udl_a, udl_b, udl_w,
                       R_A, R_B):
    beam_y = 0

    fig.add_trace(go.Scatter(
        x=[0, span], y=[beam_y, beam_y], mode="lines",
        line=dict(color=CLR_BEAM, width=6), hoverinfo="skip", name="Beam",
    ), row=1, col=1)

    _draw_triangle(fig, 0, beam_y, size=0.6, span=span)
    _draw_triangle(fig, span, beam_y, size=0.6, span=span, roller=True)

    if udl_w > 0 and udl_a < udl_b:
        a = max(udl_a, 0)
        b = min(udl_b, span)
        arrow_h = _arrow_height(udl_w, span)
        n_arrows = max(int((b - a) / (span / 30)), 3)
        xs = np.linspace(a, b, n_arrows)

        fig.add_trace(go.Scatter(
            x=[a, a, b, b], y=[beam_y, beam_y + arrow_h, beam_y + arrow_h, beam_y],
            fill="toself", fillcolor=CLR_UDL, line=dict(color=CLR_UDL_LINE, width=1),
            hoverinfo="skip", name="UDL",
        ), row=1, col=1)

        for xi in xs:
            fig.add_annotation(
                x=xi, y=beam_y, ax=xi, ay=beam_y + arrow_h,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5,
                arrowcolor=CLR_UDL_LINE,
            )

        fig.add_annotation(
            x=(a + b) / 2, y=beam_y + arrow_h + _arrow_height(20, span) * 0.3,
            text=f"w = {udl_w} kN/m", showarrow=False,
            font=dict(size=11, color=CLR_UDL_LINE),
            xref="x", yref="y",
        )

    for pos, P in all_axles:
        on = 0 <= pos <= span
        colour = CLR_AXLE if on else "rgba(200,200,200,0.5)"
        arrow_h = _arrow_height(P, span)
        fig.add_annotation(
            x=pos, y=beam_y, ax=pos, ay=beam_y + arrow_h,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor=colour,
        )
        fig.add_annotation(
            x=pos, y=beam_y + arrow_h + _arrow_height(10, span) * 0.2,
            text=f"{P:.0f} kN", showarrow=False,
            font=dict(size=10, color=colour), xref="x", yref="y",
        )
        fig.add_trace(go.Scatter(
            x=[pos], y=[beam_y], mode="markers",
            marker=dict(size=8, color=colour, symbol="circle"),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    r_offset = _arrow_height(50, span) * 0.5
    fig.add_annotation(
        x=0, y=beam_y - r_offset,
        text=f"R_A = {R_A:.1f} kN", showarrow=False,
        font=dict(size=11, color=CLR_REACTION), xref="x", yref="y",
    )
    fig.add_annotation(
        x=span, y=beam_y - r_offset,
        text=f"R_B = {R_B:.1f} kN", showarrow=False,
        font=dict(size=11, color=CLR_REACTION), xref="x", yref="y",
    )

    fig.update_yaxes(
        visible=False, range=[-2, 5], fixedrange=True, row=1, col=1,
    )


def _draw_triangle(fig, x0, y0, size, span, roller=False):
    s = size * span / 20
    h = s * 1.2
    xs = [x0 - s / 2, x0, x0 + s / 2, x0 - s / 2]
    ys = [y0 - h, y0, y0 - h, y0 - h]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor="rgba(100,100,100,0.3)", line=dict(color=CLR_BEAM, width=1.5),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    if roller:
        fig.add_trace(go.Scatter(
            x=[x0], y=[y0 - h - s * 0.25], mode="markers",
            marker=dict(size=6, color=CLR_BEAM, symbol="circle-open", line_width=1.5),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)


def _arrow_height(load, span):
    return max(0.5, min(3.5, 1.0 + 2.0 * abs(load) / 200)) * span / 20


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
