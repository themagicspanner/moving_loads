"""
Beam analysis engine for a simply supported beam under moving point loads and UDLs.

Sign convention:
  - Positive shear: upward force on the left face of a cut
  - Positive moment: sagging (tension on the bottom fibre)
  - Positive deflection: downward (same direction as gravity)
"""

import numpy as np


def solve_simply_supported(
    L: float,
    point_loads: list[tuple[float, float]],
    udl_segments: list[tuple[float, float, float]],
    n_points: int = 501,
    EI: float | None = None,
):
    """
    Analyse a simply supported beam (pin at x=0, roller at x=L).

    Parameters
    ----------
    L : float
        Beam span.
    point_loads : list of (position, magnitude)
        Each tuple is (x_pos, P) where P is positive downward.
    udl_segments : list of (start, end, intensity)
        Each tuple is (a, b, w) — UDL of intensity w (positive downward)
        applied from x = a to x = b.
    n_points : int
        Number of evaluation points along the beam.
    EI : float or None
        Flexural rigidity.  If provided, deflections are computed;
        otherwise the deflection array is all zeros.

    Returns
    -------
    x : ndarray, shape (n_points,)
    shear : ndarray
    moment : ndarray
    deflection : ndarray
    R_A : float  (reaction at x = 0, positive upward)
    R_B : float  (reaction at x = L, positive upward)
    """
    x = np.linspace(0, L, n_points)
    shear = np.zeros_like(x)
    moment = np.zeros_like(x)

    R_A = 0.0
    R_B = 0.0

    # --- Point loads ---
    for pos, P in point_loads:
        if pos < 0 or pos > L:
            continue
        # Reactions
        rB = P * pos / L
        rA = P - rB
        R_A += rA
        R_B += rB

        # Shear and moment contributions using Macaulay brackets
        for i, xi in enumerate(x):
            if xi <= pos:
                shear[i] += rA
                moment[i] += rA * xi
            else:
                shear[i] += rA - P
                moment[i] += rA * xi - P * (xi - pos)

    # --- UDL segments ---
    for a, b, w in udl_segments:
        a = max(a, 0.0)
        b = min(b, L)
        if a >= b or w == 0:
            continue
        total_udl = w * (b - a)
        centroid = (a + b) / 2.0
        rB = total_udl * centroid / L
        rA = total_udl - rB
        R_A += rA
        R_B += rB

        for i, xi in enumerate(x):
            if xi <= a:
                shear[i] += rA
                moment[i] += rA * xi
            elif xi <= b:
                shear[i] += rA - w * (xi - a)
                moment[i] += rA * xi - w * (xi - a) ** 2 / 2.0
            else:
                shear[i] += rA - total_udl
                moment[i] += rA * xi - total_udl * (xi - centroid)

    # --- Deflection via double integration (moment-area / numerical) ---
    deflection = np.zeros_like(x)
    if EI is not None and EI > 0:
        dx = x[1] - x[0]
        # Integrate M/EI twice using the trapezoidal rule
        theta = np.zeros_like(x)  # slope
        v = np.zeros_like(x)  # deflection

        M_over_EI = moment / EI

        # First integration: slope (plus constant C1)
        for i in range(1, len(x)):
            theta[i] = theta[i - 1] + 0.5 * (M_over_EI[i - 1] + M_over_EI[i]) * dx

        # Second integration: deflection (plus constant C2)
        for i in range(1, len(x)):
            v[i] = v[i - 1] + 0.5 * (theta[i - 1] + theta[i]) * dx

        # Boundary conditions: v(0) = 0, v(L) = 0
        # v already has v(0)=0 from initial condition.
        # Adjust for v(L) = 0 by subtracting a linear correction.
        C1_correction = v[-1] / L
        for i in range(len(x)):
            v[i] -= C1_correction * x[i]

        deflection = v

    return x, shear, moment, deflection, R_A, R_B


def vehicle_positions_at_offset(
    axle_spacings: list[float],
    axle_loads: list[float],
    front_axle_x: float,
) -> list[tuple[float, float]]:
    """
    Return the (position, load) pairs for every axle of a vehicle,
    given the position of the front axle.

    Parameters
    ----------
    axle_spacings : list of float
        Distances *between* successive axles (length = n_axles - 1).
    axle_loads : list of float
        Load on each axle (length = n_axles).
    front_axle_x : float
        Position of the leading (front) axle.

    Returns
    -------
    list of (position, load)
    """
    positions = []
    xi = front_axle_x
    for idx, load in enumerate(axle_loads):
        positions.append((xi, load))
        if idx < len(axle_spacings):
            xi -= axle_spacings[idx]  # axles trail behind the front axle
    return positions
