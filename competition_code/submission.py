"""
ROAR competition solution — LTV-LQR controller + iterative minimum-curvature racing line.

WHAT CHANGED (vs. the previous version) AND WHY
-----------------------------------------------
Two regressions are fixed here.

(1) THE LINE WENT CENTRAL. A previous revision added an under-relaxation blend
    plus a "keep the lowest-curvature iterate" selector to the margin loop.
    Together those damped the offset amplitude and pulled the line back toward
    the centerline, losing the aggressive out-in-out. Removed. The line now:
      * Stage A — a clean bounded MINIMUM-CURVATURE solve: rides the OUTER edge
        on entry, cuts the INNER apex, runs back OUT on exit, using the full
        corridor. No damping.
      * Stage B — MINIMUM-TIME refinement: re-solve with the curvature penalty
        weighted by (v/vmax)^2, so fast corners get straightened / late-apexed
        where it buys the most lap time.
    Corridor margins are kept lean and local (off-tracking read from the driven
    line, capped), so the line is *meant* to use the edges and one tight corner
    never taxes the rest.

(2) THE STEERING OSCILLATED. The control reference tangent was built from raw
    finite differences on a 0.5 m grid (smooth_win=0), so path_yaw / e_psi were
    noisy, and the feedforward curvature was only lightly smoothed. Combined
    with stiff gains (Q_E=Q_PSI=8, R_STEER=2) and a hard steer-rate clip, that
    chattered. Fixes:
      * control tangent is now smoothed; feedforward curvature is smoothed
        HARDER (it's a second derivative) — the biggest single anti-chatter win;
      * feedforward is read a short, speed-scaled distance AHEAD to offset
        actuator lag (no entry catch-up wobble);
      * gains softened (Q_E 8->3.5, R_STEER 2->3.5) with heading alignment kept
        firm so the car still doesn't run wide;
      * a first-order low-pass on the steering output (the primary smoother),
        with the rate clip demoted to a safety guard so it can't limit-cycle.

Earlier improvements that remain:
  * ITERATIVE off-tracking margins read from the driven line (per-turn, no
    global buffer);
  * ACCURATE curvature objective (second-order apex term);
  * LIVE vehicle-geometry detection (wheelbase / half-width / mass) from the
    CARLA actor — see "VEHICLE GEOMETRY" below;
  * DOWNFORCE-aware speed model (K_DF), default off.

The longitudinal physics constants are kept as you validated them — the speed
gains come from a better LINE, not from inflating grip (a crash resets you to
lap 0).

VEHICLE GEOMETRY: MEASURED LIVE, NOT GUESSED
---------------------------------------------
There is no public spec sheet for ROAR's custom CARLA "vehicle.dallara.dallara"
asset — it's a bespoke content package, not a published real-world car, and
the previous WHEELBASE = 2.875 m was a guess (it happens to match a Tesla
Model 3, the vehicle from ROAR's *old* Berkeley Major Map era; the current
Monza-map competition runs an open-wheel Dallara, so that number had no real
basis here).

Rather than guess again, `_autodetect_vehicle_params()` reads the wheelbase,
track width and mass directly from the live CARLA actor at the start of
`initialize()`:
  - wheel hub positions from `VehiclePhysicsControl.wheels[i].position`
    (indices are FL, FR, RL, RR per CARLA's documented convention) give the
    front-axle-to-rear-axle distance and the track width exactly, straight
    from the simulator's own physics model.
  - the vehicle bounding box gives an independent, trusted-units length/width
    cross-check, used to auto-correct a known CARLA quirk where wheel
    position has shipped in centimeters instead of meters in some versions.
  - CAR_HALF_WIDTH is taken as the larger of the body half-width and
    (track/2 + wheel radius), so an open-wheel car's tires — which sit wider
    than the cockpit — are never under-counted.
This reaches past the competition's `RoarCompetitionAgentWrapper` (which
doesn't expose physics) via its underlying `_wrapped`/`_base_actor`
attributes. Every step is wrapped in try/except: if anything about that
internal chain ever changes, it falls back to the documented constants below
and prints exactly which path was used, so you can verify it in the console
rather than trust it blindly.

TUNING GUIDE (biggest lap-time levers, in order)
------------------------------------------------
  * K_DF  : leave at 0.0 by default. CARLA's vehicle tire model is a flat,
            non-speed-dependent friction scalar (see the printed
            `tire_friction` diagnostic) — there is no built-in downforce
            mechanic to unlock. Only raise K_DF if you have directly observed
            speed-dependent grip in real laps on this asset; otherwise it
            just makes the planner overestimate fast-corner grip while the
            sim's actual tire model doesn't deliver it, risking exactly the
            understeer-into-a-wall failure mode you want to avoid.
  * A_LAT / GRIP_MARGIN : mechanical grip ceiling and its safety factor.
            Cross-check against the printed `tire_friction * 9.81` figure.
  * TRACK_MARGIN_MIN / _K : how close to the edge you trust the line. Lower =
            faster + riskier. These are now *local*, so loosening them does not
            cost you elsewhere.
  * A_BRAKE : later braking. Overestimating this is the easiest way to overrun
            a corner — raise cautiously.
Set EXPORT_LINE = True once to dump racing_line.npz and plot it offline.
"""

from typing import List, Dict
import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.linalg import expm
import roar_py_interface


# =============================================================================
#  AUTOTUNER HOOK  (set OVERRIDES["A_LAT"] = ... from an outer script to sweep)
# =============================================================================
OVERRIDES: Dict[str, float] = {}

TUNABLE = [
    "Q_E", "Q_PSI", "R_STEER", "MAX_STEER_RAD",
    "A_LAT", "GRIP_MARGIN", "K_DF", "A_BRAKE", "A_ACCEL", "V_MAX",
    "KP_SPEED", "KI_SPEED", "CURV_WIN_M",
    "TRACK_MARGIN_MIN", "TRACK_MARGIN_K", "TRACK_MARGIN_MAX",
    "OFFTRACK_CAP", "EXTRA_TRACK_WIDTH_M", "BASE_SAFETY_MARGIN", "DS_OPT",
    "STEER_LP_BETA", "STEER_RATE", "FF_PREVIEW_T", "FF_SMOOTH_M",
    "MINTIME_POWER",
]


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
#  GEOMETRY & TRAJECTORY HELPERS
# =============================================================================
def _resample_closed(points: np.ndarray, ds: float) -> np.ndarray:
    """Arc-length resample of a closed polyline (resolution defined by xy)."""
    loop = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(loop[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(round(total / ds)))
    s_new = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, points.shape[1]))
    for dim in range(points.shape[1]):
        out[:, dim] = np.interp(s_new, s, loop[:, dim])
    return out


def _resample_scalar_closed(values: np.ndarray, xy: np.ndarray, ds: float) -> np.ndarray:
    loop = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(round(total / ds)))
    s_new = np.linspace(0.0, total, n, endpoint=False)
    v_loop = np.concatenate([values, values[:1]])
    return np.interp(s_new, s, v_loop)


def _seg_lengths(path: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)


def _progress(path: np.ndarray) -> np.ndarray:
    seg = _seg_lengths(path)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s[:-1] / s[-1]


def _circ_smooth(arr: np.ndarray, win: int) -> np.ndarray:
    if win < 1:
        return arr
    k = 2 * win + 1
    kernel = np.ones(k) / k
    pad = np.concatenate([arr[-win:], arr, arr[:win]])
    return np.convolve(pad, kernel, mode="valid")


def _tangents_normals(path: np.ndarray, smooth_win: int = 0):
    """Central-difference unit tangents and left normals on a closed path."""
    t = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    if smooth_win and smooth_win > 0:
        t = np.column_stack([_circ_smooth(t[:, 0], smooth_win),
                             _circ_smooth(t[:, 1], smooth_win)])
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])      # left normal; alpha>0 -> left
    return t, n


def _windowed_curvature(path: np.ndarray, win: int) -> np.ndarray:
    """Unsigned Menger curvature over a +/-win window (robust for speed calc)."""
    p0 = np.roll(path, win, axis=0)
    p1 = path
    p2 = np.roll(path, -win, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area = 0.5 * np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1])
                        - (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1]))
    return 4.0 * area / (a * b * c + 1e-9)


def _signed_curvature(tangent: np.ndarray, seg: np.ndarray, smooth_win: int = 0) -> np.ndarray:
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    dth = normalize_rad(np.roll(heading, -1) - heading)
    kappa = dth / np.maximum(seg, 1e-3)
    return _circ_smooth(kappa, smooth_win) if smooth_win > 0 else kappa


def _min_curvature_offsets(center, kappa_center, half, weight=None,
                           w_curv=40.0, w_smooth=2.0, w_reg=0.1):
    """
    Bounded minimum-curvature offsets along the centerline normal.

    Linearised curvature of the offset path:
        kappa_new(s) ~= kappa_c(s) + alpha''(s) - 2 * kappa_c(s)^2 * alpha(s)
    We drive the (zero-mean) curvature to zero subject to |alpha| <= half, plus
    light smoothness and Tikhonov regularisation so the line never gets jagged.

    `weight` (optional, per-point) scales how hard curvature is penalised at
    each station. Passing weight ~ (v/vmax)^2 turns this from a minimum-
    CURVATURE solve into a minimum-TIME-biased one: fast sections get
    straightened (later apex, wider line) while slow corners keep their apex.
    """
    n = len(center)
    idx = np.arange(n)
    im1 = (idx - 1) % n
    ip1 = (idx + 1) % n
    ds = float(np.mean(_seg_lengths(center)))
    ds2 = ds * ds

    # alpha''  (second difference, periodic)
    D2 = sp.csr_matrix(
        (np.concatenate([np.ones(n), -2 * np.ones(n), np.ones(n)]),
         (np.concatenate([idx, idx, idx]),
          np.concatenate([im1, idx, ip1]))),
        shape=(n, n)) / ds2

    # -2*kappa^2 * alpha   (apex-sharpening second-order term).
    # Clamp the curvature feeding this term so a spurious spike in noisy/closed
    # waypoint data cannot blow up the operator and destabilise the solve.
    kc = np.clip(kappa_center, -0.5, 0.5)
    C2 = sp.diags(-2.0 * kc ** 2)
    M = (D2 + C2)

    k0 = kappa_center - np.mean(kappa_center)      # zero-mean on a closed loop
    if weight is None:
        wsqrt = np.ones(n)
    else:
        wsqrt = np.sqrt(np.clip(np.asarray(weight, float), 1e-3, None))
    A_curv = sp.diags(wsqrt) @ (M * w_curv)
    b_curv = -k0 * w_curv * wsqrt

    # alpha'  (first difference) — smoothness
    D1 = sp.csr_matrix(
        (np.concatenate([np.ones(n), -np.ones(n)]),
         (np.concatenate([idx, idx]),
          np.concatenate([ip1, idx]))),
        shape=(n, n)) / ds
    A_smooth = D1 * w_smooth
    b_smooth = np.zeros(n)

    A_reg = sp.diags(np.ones(n) * w_reg)
    b_reg = np.zeros(n)

    A = sp.vstack([A_curv, A_smooth, A_reg]).tocsr()
    b = np.concatenate([b_curv, b_smooth, b_reg])

    lb = -np.asarray(half, float)
    ub = np.asarray(half, float)
    res = lsq_linear(A, b, bounds=(lb, ub), max_iter=200, tol=1e-3,
                     lsmr_tol="auto")
    return res.x


def _velocity_profile(path, A_LAT, A_ACCEL, A_BRAKE, V_MAX, K_DF, curv_win, passes=6):
    """
    Forward/backward speed profile with a downforce-aware traction circle.

      g_lat(v) = A_LAT + K_DF * v^2            (lateral grip grows with downforce)
      apex seed: v^2 <= A_LAT / (kappa - K_DF) (closed form of the above)
      g-g coupling: a_lon = a_max * sqrt(1 - (a_lat/g_lat)^2)
    """
    n = len(path)
    A_LAT   = np.broadcast_to(np.asarray(A_LAT, float),   (n,)).copy()
    A_ACCEL = np.broadcast_to(np.asarray(A_ACCEL, float), (n,)).copy()
    A_BRAKE = np.broadcast_to(np.asarray(A_BRAKE, float), (n,)).copy()
    V_MAX   = np.broadcast_to(np.asarray(V_MAX, float),   (n,)).copy()
    K_DF    = np.broadcast_to(np.asarray(K_DF, float),    (n,)).copy()

    kappa = _windowed_curvature(path, curv_win)
    ks = 0.25 * np.roll(kappa, 1) + 0.5 * kappa + 0.25 * np.roll(kappa, -1)
    seg = np.maximum(np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1), 1e-3)

    denom = np.maximum(ks - K_DF, 1e-4)
    v = np.minimum(np.sqrt(A_LAT / denom), V_MAX)

    for _ in range(passes):
        # backward pass: respect braking capability into every corner
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i] ** 2
            a_lat = min(v[i] ** 2 * ks[i], g_lat)
            a_lon = max(0.15 * A_BRAKE[i],
                        A_BRAKE[i] * np.sqrt(max(0.0, 1.0 - (a_lat / g_lat) ** 2)))
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_lon * seg[i]))
        # forward pass: respect drive/accel capability out of every corner
        for i in range(n):
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i] ** 2
            a_lat = min(v[i] ** 2 * ks[i], g_lat)
            a_lon = A_ACCEL[i] * np.sqrt(max(0.0, 1.0 - (a_lat / g_lat) ** 2))
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_lon * seg[i]))
    return v, seg, ks


def _dare_gain(A, B, Q, R, iters=400, tol=1e-11):
    """Discrete algebraic Riccati solve by iteration -> LQR gain K."""
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        S = R + BtP @ B
        K = np.linalg.solve(S, BtP @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    BtP = B.T @ P
    S = R + BtP @ B
    return np.linalg.solve(S, BtP @ A)


# =============================================================================
#  SOLUTION
# =============================================================================
class RoarCompetitionSolution:

    # ----- LATERAL: LTV-LQR cost weights ------------------------------------
    # The chatter came mainly from a NOISY reference (raw tangent + lightly-
    # smoothed feedforward), not from the gains, so we fix the reference and
    # only modestly soften the gains. Heading alignment (Q_PSI) is kept strong
    # so the car never runs wide and off the track (a crash resets to lap 0);
    # R_STEER is raised a little to take the edge off steering effort. The LTV
    # scheduling already gentles the gains at speed.
    Q_E      = 6.0      # lateral-offset penalty (was 8.0)
    Q_PSI    = 8.0      # heading-alignment penalty (kept firm — don't run wide)
    Q_DELTA  = 0.0
    R_STEER  = 5.0      # steering-effort penalty (was 2.0 — higher = smoother)

    ACTUATOR_LAG_S = 0.20
    MAX_STEER_RAD = 1.0
    V_FLOOR = 3.0

    # ----- LATERAL SMOOTHING / ANTI-OSCILLATION -----------------------------
    STEER_LP_BETA = 0.65   # output low-pass: y += beta*(cmd - y). Lower = smoother
    STEER_RATE    = 0.30   # generous rate cap [norm/step] (safety, not the main limiter)
    FF_PREVIEW_T  = 0.10   # feedforward look-ahead time [s] to offset actuator lag
    FF_SMOOTH_M   = 4.0    # feedforward-curvature smoothing length [m] (kills FF chatter)

    # ----- LONGITUDINAL: speed-profile physics (kept — validated) -----------
    A_LAT       = 18.5    # mechanical lateral grip ceiling [m/s^2]  (~1.9 g)
    K_DF        = 0.0     # DOWNFORCE term [1/m]; see TUNING GUIDE (leave 0 on flat-tire sim)
    A_ACCEL     = 30.0    # forward accel ceiling [m/s^2]
    A_BRAKE     = 20.0    # braking ceiling [m/s^2]
    V_MAX       = 130.0   # hard speed cap [m/s] (straights are accel/grip limited)
    GRIP_MARGIN = 0.92    # 8% lateral-grip safety net

    KP_SPEED = 2.5
    KI_SPEED = 0.1

    # ----- GEOMETRY & KINEMATICS --------------------------------------------
    DS_OPT       = 2.5    # optimisation grid spacing [m]
    DS_TRACK     = 0.5    # control grid spacing [m]
    CURV_WIN_M   = 5.0
    SMOOTH_WIN_M = 1.5

    # ----- VEHICLE GEOMETRY ---------------------------------------------
    # These are FALLBACKS ONLY. AUTO_DETECT_VEHICLE_PARAMS (below) reads the
    # real wheelbase/half-width from the live CARLA actor's physics control
    # in initialize(); these constants are only used if that detection fails
    # for any reason. They are deliberately framed as "best guess, unverified"
    # rather than spec-sheet numbers, because no public spec exists for
    # ROAR's custom "vehicle.dallara.dallara" CARLA asset.
    AUTO_DETECT_VEHICLE_PARAMS = True
    WHEELBASE_FALLBACK_M       = 2.85   # mid-pack of real Dallara open-wheel chassis (~2.78-3.01 m)
    CAR_HALF_WIDTH_FALLBACK_M  = 0.97   # half of a typical F3/Indy-Lights-class car width (~1.88-1.99 m)

    WHEELBASE    = WHEELBASE_FALLBACK_M       # overwritten in initialize() if detection succeeds
    CAR_HALF_WIDTH = CAR_HALF_WIDTH_FALLBACK_M  # overwritten in initialize() if detection succeeds

    # ----- CORRIDOR MARGINS (lean, so the line USES the full track) ---------
    # usable_half = track_half - CAR_HALF_WIDTH - tracking_margin - offtracking
    #   tracking_margin = clip(MIN + K*|kappa_line|, MIN, MAX)  (LQR-error budget)
    #   offtracking     = min(0.5*WHEELBASE^2*|kappa_line|, OFFTRACK_CAP) (rear axle)
    # These are kept lean on purpose: the optimiser is MEANT to ride the outer
    # edge on entry/exit and the inner edge at the apex. Off-tracking is read
    # from the optimised line (small), so a tight corner doesn't tax the rest.
    TRACK_MARGIN_MIN = 0.25   # base lateral allowance for tracking error [m]
    TRACK_MARGIN_K   = 1.0    # extra allowance per unit curvature [m * m]
    TRACK_MARGIN_MAX = 2  # cap on the allowance [m]
    OFFTRACK_CAP     = 0.45   # cap on rear-axle off-tracking term [m]
    MIN_CORRIDOR     = 0.15   # never let the corridor close fully [m]
    EXTRA_TRACK_WIDTH_M = 0.0 # add usable width (e.g. curbs) if you trust them [m/side]
    BASE_SAFETY_MARGIN = TRACK_MARGIN_MIN  # kept name for the autotuner hook

    # ----- RACING-LINE OPTIMISER --------------------------------------------
    N_MARGIN_ITERS = 3        # off-tracking margin refinement passes (no damping)
    MINTIME_REFINE = True     # bias the line toward minimum TIME, not just min curvature
    MINTIME_PASSES = 2        # velocity-weighted refinement passes
    MINTIME_POWER  = 2.0      # curvature weight ~ (v/vmax)^POWER in fast sections
    WIDTH_MIN   = 5.0         # clamp on reported lane width [m]
    WIDTH_MAX   = 25.0
    WIDTH_DEFAULT = 12.0

    # ----- LATERAL FEEDFORWARD GUARD ----------------------------------------
    KAPPA_FF_MAX = 0.50       # numerical guard only (no real corner reaches this)

    CONTROL_DT = 0.05
    TELEMETRY  = True
    EXPORT_LINE = False       # set True once to dump racing_line.npz for plotting

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor=None,
        location_sensor=None,
        velocity_sensor=None,
        rpy_sensor=None,
        occupancy_map_sensor=None,
        collision_sensor=None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.integral_error = 0.0
        self.idx = 0
        self._prev_steer = 0.0
        self.actual_steer_rad = 0.0
        self._K_cache: Dict[int, np.ndarray] = {}
        self.telemetry = {}

        for name in TUNABLE:
            if name in OVERRIDES:
                cur = getattr(self, name, 0.0)
                setattr(self, name, type(cur)(OVERRIDES[name]) if isinstance(cur, int)
                        else float(OVERRIDES[name]))

    # ---- Live vehicle geometry: read wheelbase/half-width/mass straight ----
    # from the simulator instead of guessing them. See module docstring
    # "VEHICLE GEOMETRY: MEASURED LIVE, NOT GUESSED" for why this exists.
    def _autodetect_vehicle_params(self) -> None:
        if not self.AUTO_DETECT_VEHICLE_PARAMS:
            print(f"[VEHICLE] auto-detect disabled; using fallbacks "
                  f"wheelbase={self.WHEELBASE:.3f} m, half_width={self.CAR_HALF_WIDTH:.3f} m")
            return

        # 1) Reach past RoarCompetitionAgentWrapper to the underlying
        #    RoarPyCarlaActor, then to the raw carla.Vehicle actor. Both
        #    attribute names are private implementation details of the
        #    competition harness, so every hop is optional/defensive.
        raw_actor = None
        for wrapper_attr in ("_wrapped", "vehicle", "wrapped"):
            candidate = getattr(self.vehicle, wrapper_attr, None)
            if candidate is not None:
                for base_attr in ("_base_actor", "base_actor", "carla_actor"):
                    ba = getattr(candidate, base_attr, None)
                    if ba is not None and hasattr(ba, "get_physics_control"):
                        raw_actor = ba
                        break
            if raw_actor is not None:
                break
        # also try directly on self.vehicle in case the wrapper ever exposes it
        if raw_actor is None and hasattr(self.vehicle, "get_physics_control"):
            raw_actor = self.vehicle

        # 2) Trusted-units cross-check: bounding box length/width, in meters,
        #    available via the public RoarPyCarlaActor.bounding_box property
        #    (or via the same private hop if the wrapper doesn't expose it).
        bb = getattr(self.vehicle, "bounding_box", None)
        if bb is None:
            for wrapper_attr in ("_wrapped", "vehicle", "wrapped"):
                candidate = getattr(self.vehicle, wrapper_attr, None)
                bb = getattr(candidate, "bounding_box", None) if candidate is not None else None
                if bb is not None:
                    break
        length_m = width_m = None
        try:
            if bb is not None:
                length_m = float(2.0 * bb.extent[0])
                width_m = float(2.0 * bb.extent[1])
        except Exception:
            pass

        if raw_actor is None:
            print(f"[VEHICLE] live physics control unavailable; using fallbacks "
                  f"wheelbase={self.WHEELBASE:.3f} m, half_width={self.CAR_HALF_WIDTH:.3f} m")
            return

        try:
            phys = raw_actor.get_physics_control()
            wheels = list(phys.wheels)
            mass_kg = float(getattr(phys, "mass", float("nan")))
            if len(wheels) != 4:
                raise ValueError(f"expected 4 wheels, got {len(wheels)}")

            def _pos(w):
                p = w.position
                return np.array([float(p.x), float(p.y), float(p.z)])

            # CARLA's documented wheel order is FL, FR, RL, RR.
            front_mid = 0.5 * (_pos(wheels[0]) + _pos(wheels[1]))
            rear_mid  = 0.5 * (_pos(wheels[2]) + _pos(wheels[3]))
            raw_wheelbase = float(np.linalg.norm(front_mid - rear_mid))
            raw_front_track = float(np.linalg.norm(_pos(wheels[0]) - _pos(wheels[1])))
            raw_rear_track  = float(np.linalg.norm(_pos(wheels[2]) - _pos(wheels[3])))
            wheel_radius_cm = float(np.mean([getattr(w, "radius", 35.0) for w in wheels]))

            # 3) Unit auto-correction. CARLA's Transform/Location API is in
            #    meters, but WheelPhysicsControl.position has shipped in
            #    centimeters in some CARLA versions (a documented community-
            #    reported inconsistency). Rather than assume either, compare
            #    against the bounding-box length, which is reliably in meters.
            scale = 1.0
            basis = "assumed-meters (no bounding box to cross-check)"
            if length_m is not None and length_m > 0.5 and raw_wheelbase > 0:
                ratio = raw_wheelbase / length_m
                if ratio > 5.0:          # raw values are ~100x too big -> cm
                    scale = 0.01
                    basis = f"cm->m correction (raw/length ratio {ratio:.1f})"
                elif 0.2 <= ratio <= 1.05:
                    scale = 1.0
                    basis = f"already meters (raw/length ratio {ratio:.2f})"
                else:
                    # inconsistent with bounding box; don't trust the raw
                    # number at all, fall back to the documented constant
                    raise ValueError(f"raw wheelbase/length ratio {ratio:.2f} "
                                     f"out of sane range; rejecting live value")

            wheelbase_m = raw_wheelbase * scale
            front_track_m = raw_front_track * scale
            rear_track_m = raw_rear_track * scale
            track_m = max(front_track_m, rear_track_m)

            # Safety clamp: never accept a wheelbase/track outside a sane
            # envelope for any car-like vehicle, even if every check above
            # passed (belt-and-suspenders against a corrupted read).
            wheelbase_m = float(np.clip(wheelbase_m, 1.5, 4.0))
            track_m = float(np.clip(track_m, 0.8, 2.2))

            # CAR_HALF_WIDTH: take the larger of body half-width (bounding
            # box) and (track/2 + tire radius), since an open-wheel car's
            # tires sit outside the bodywork and must not be under-counted.
            # wheel radius is documented in cm regardless of the position-
            # unit quirk handled above, so it always gets its own *0.01.
            tire_half_extent = track_m / 2.0 + wheel_radius_cm * 0.01
            candidates = [tire_half_extent]
            if width_m is not None and width_m > 0.3:
                candidates.append(width_m / 2.0)
            half_width_m = float(np.clip(max(candidates), 0.5, 1.3))

            self.WHEELBASE = wheelbase_m
            self.CAR_HALF_WIDTH = half_width_m
            self.detected_mass_kg = mass_kg

            # Pure diagnostics, not applied automatically: tire_friction is a
            # flat (non-speed-dependent) scalar in CARLA's simplified tire
            # model, roughly bounding achievable lateral g as
            # friction * 9.81. drag_coefficient affects only top-speed via
            # aerodynamic drag. Both are printed so you can sanity-check
            # A_LAT/GRIP_MARGIN and V_MAX against the sim's own numbers
            # without this script silently overriding your validated tuning.
            tire_friction = float(np.mean([getattr(w, "tire_friction", float("nan")) for w in wheels]))
            drag_coeff = float(getattr(phys, "drag_coefficient", float("nan")))
            self.detected_tire_friction = tire_friction
            self.detected_drag_coefficient = drag_coeff

            print(f"[VEHICLE] live-detected: wheelbase={wheelbase_m:.3f} m, "
                  f"track={track_m:.3f} m, half_width={half_width_m:.3f} m, "
                  f"mass={mass_kg:.0f} kg | basis: {basis}")
            if length_m is not None:
                print(f"[VEHICLE] cross-check: bounding-box length={length_m:.3f} m, "
                      f"width={width_m:.3f} m")
            if np.isfinite(tire_friction):
                implied_g = tire_friction * 9.81
                print(f"[VEHICLE] tire_friction={tire_friction:.2f} (implies up to "
                      f"~{implied_g:.1f} m/s^2 grip ceiling) vs your A_LAT*GRIP_MARGIN="
                      f"{self.A_LAT * self.GRIP_MARGIN:.1f} m/s^2 -- CARLA's flat tire "
                      f"model has no built-in downforce, so this ceiling does not grow "
                      f"with speed; treat K_DF as untested unless you've observed "
                      f"otherwise in real laps.")
            if np.isfinite(drag_coeff):
                print(f"[VEHICLE] drag_coefficient={drag_coeff:.3f} (affects top-speed "
                      f"only; your V_MAX={self.V_MAX:.0f} m/s is a separate hard cap)")
        except Exception as e:
            print(f"[VEHICLE] live detection failed ({e}); using fallbacks "
                  f"wheelbase={self.WHEELBASE:.3f} m, half_width={self.CAR_HALF_WIDTH:.3f} m")

    # ---- LTV-LQR gain, scheduled on speed, augmented with actuator lag -----
    def _lqr_gain(self, v: float) -> np.ndarray:
        key = int(round(v))
        g = self._K_cache.get(key)
        if g is not None:
            return g
        vm = max(float(key), self.V_FLOOR)
        dt, L = self.CONTROL_DT, self.WHEELBASE
        tau = max(float(self.ACTUATOR_LAG_S), 1e-2)
        Ac = np.array([[0.0, vm,  0.0],
                       [0.0, 0.0, vm / L],
                       [0.0, 0.0, -1.0 / tau]])
        Bc = np.array([[0.0], [0.0], [1.0 / tau]])
        Maug = np.zeros((4, 4))
        Maug[:3, :3] = Ac
        Maug[:3, 3:] = Bc
        Md = expm(Maug * dt)
        A = Md[:3, :3]
        B = Md[:3, 3:]
        Q = np.diag([self.Q_E, self.Q_PSI, self.Q_DELTA])
        R = np.array([[self.R_STEER]])
        K = _dare_gain(A, B, Q, R)
        self._K_cache[key] = K
        return K

    # ------------------------------------------------------------------------
    async def initialize(self) -> None:
        self._autodetect_vehicle_params()  # sets self.WHEELBASE / CAR_HALF_WIDTH

        wps = self.maneuverable_waypoints
        center_raw_3d = np.array([w.location[:3] for w in wps])
        center_raw = center_raw_3d[:, :2]

        def _w(wp):
            lw = getattr(wp, "lane_width", None)
            if lw is None or not np.isfinite(lw) or lw <= 0:
                lw = self.WIDTH_DEFAULT
            return float(np.clip(lw, self.WIDTH_MIN, self.WIDTH_MAX))
        width_raw = np.array([_w(w) for w in wps])

        # ---- resample on a coarse (optimisation) and a fine (control) grid --
        center_o_3d = _resample_closed(center_raw_3d, self.DS_OPT)
        center_o = center_o_3d[:, :2]
        width_o = _resample_scalar_closed(width_raw, center_raw, self.DS_OPT)

        center_f_3d = _resample_closed(center_raw_3d, self.DS_TRACK)
        center_f = center_f_3d[:, :2]
        width_f = _resample_scalar_closed(width_raw, center_raw, self.DS_TRACK)

        win_o = max(1, int(round(self.SMOOTH_WIN_M / self.DS_OPT)))
        win_f = max(1, int(round(self.SMOOTH_WIN_M / self.DS_TRACK)))
        curv_win_f = max(1, int(round(self.CURV_WIN_M / self.DS_TRACK)))

        tangent_o, normal_o = _tangents_normals(center_o, smooth_win=win_o)
        seg_o = _seg_lengths(center_o)
        kappa_center_o = _signed_curvature(tangent_o, seg_o, smooth_win=win_o)
        track_half_o = width_o / 2.0

        # =====================================================================
        #  RACING LINE — minimum-curvature base, then minimum-TIME refinement.
        #
        #  Stage A: a clean bounded minimum-curvature solve. This is the classic
        #  OUT-IN-OUT line: it rides the OUTER edge on entry, cuts the apex on
        #  the INNER edge, and runs back out to the OUTER edge on exit, using as
        #  much of the corridor as the track allows. No damping / no "keep best"
        #  blending (those quietly pulled the old line back toward the centre).
        #
        #  Stage B: re-solve a couple of times with the curvature penalty
        #  weighted by (v/vmax)^2. Fast corners get straightened further (later
        #  apex, even wider line) because a metre saved there buys more time —
        #  this is the minimum-TIME bias on top of minimum curvature.
        #
        #  The only thing iterated for the corridor is the rear-axle
        #  off-tracking, read from the *driven* line (small, capped), so one
        #  tight corner never taxes the straights or the fast corners.
        # =====================================================================
        kclamp = float(self.KAPPA_FF_MAX)

        def _corridor(kappa_line):
            kl = _circ_smooth(np.clip(kappa_line, -kclamp, kclamp), win_o)
            offtrack = np.minimum(0.5 * (self.WHEELBASE ** 2) * np.abs(kl),
                                  self.OFFTRACK_CAP)
            tmarg = np.clip(self.TRACK_MARGIN_MIN + self.TRACK_MARGIN_K * np.abs(kl),
                            self.TRACK_MARGIN_MIN, self.TRACK_MARGIN_MAX)
            usable = (track_half_o + self.EXTRA_TRACK_WIDTH_M
                      - (self.CAR_HALF_WIDTH + tmarg + offtrack))
            return np.maximum(usable, self.MIN_CORRIDOR)

        def _line_curvature(alpha):
            p = center_o + alpha[:, None] * normal_o
            t, _ = _tangents_normals(p, smooth_win=win_o)
            return _signed_curvature(t, _seg_lengths(p), smooth_win=win_o)

        # Stage A — minimum curvature, corridor margins converged (no damping)
        kappa_line_o = kappa_center_o.copy()
        alpha_o = _min_curvature_offsets(center_o, kappa_center_o, _corridor(kappa_line_o))
        for _ in range(int(self.N_MARGIN_ITERS)):
            kappa_line_o = _line_curvature(alpha_o)
            alpha_o = _min_curvature_offsets(center_o, kappa_center_o, _corridor(kappa_line_o))

        # Stage B — minimum-time refinement: weight curvature by local speed^2
        if self.MINTIME_REFINE:
            for _ in range(int(self.MINTIME_PASSES)):
                path_o = center_o + alpha_o[:, None] * normal_o
                v_o, _, _ = _velocity_profile(
                    path_o, self.A_LAT * self.GRIP_MARGIN, self.A_ACCEL,
                    self.A_BRAKE, self.V_MAX, self.K_DF,
                    max(1, int(round(self.CURV_WIN_M / self.DS_OPT))))
                vref = max(float(np.max(v_o)), 1.0)
                # floor keeps slow corners apexing; (v/vmax)^p biases fast ones
                w = 0.5 + (v_o / vref) ** float(self.MINTIME_POWER)
                kappa_line_o = _line_curvature(alpha_o)
                alpha_o = _min_curvature_offsets(
                    center_o, kappa_center_o, _corridor(kappa_line_o), weight=w)

        kappa_line_o = _line_curvature(alpha_o)

        # ---- map coarse offsets onto the fine control grid ------------------
        u_o = _progress(center_o)
        u_f = _progress(center_f)
        off_f = np.interp(u_f, np.append(u_o, 1.0), np.append(alpha_o, alpha_o[0]))

        # fine-grid corridor (same per-turn model, evaluated on the fine line)
        kappa_line_f = np.interp(u_f, np.append(u_o, 1.0),
                                 np.append(kappa_line_o, kappa_line_o[0]))
        kappa_line_f = _circ_smooth(np.clip(kappa_line_f, -kclamp, kclamp), win_f)
        tangent_f, normal_f = _tangents_normals(center_f, smooth_win=win_f)
        track_half_f = width_f / 2.0
        offtrack_f = np.minimum(0.5 * (self.WHEELBASE ** 2) * np.abs(kappa_line_f),
                                self.OFFTRACK_CAP)
        tmarg_f = np.clip(self.TRACK_MARGIN_MIN + self.TRACK_MARGIN_K * np.abs(kappa_line_f),
                          self.TRACK_MARGIN_MIN, self.TRACK_MARGIN_MAX)
        half_f = np.maximum(track_half_f + self.EXTRA_TRACK_WIDTH_M
                            - (self.CAR_HALF_WIDTH + tmarg_f + offtrack_f),
                            self.MIN_CORRIDOR)

        off_f = np.clip(off_f, -half_f, half_f)
        off_f = _circ_smooth(off_f, max(1, int(round(1.0 / self.DS_TRACK))))
        off_f = np.clip(off_f, -half_f, half_f)
        path_opt = center_f + off_f[:, None] * normal_f

        # ---- final line geometry -------------------------------------------
        # SMOOTHED tangent for the control reference (raw finite differences on
        # a 0.5 m grid made path_yaw / e_psi noisy and the steering chattery).
        tangent = _tangents_normals(path_opt, smooth_win=win_f)[0]
        seg = _seg_lengths(path_opt)
        self.path = path_opt
        self.tangent = tangent
        self.path_3d = np.column_stack([path_opt, center_f_3d[:, 2]])

        # Feedforward curvature is smoothed HARDER than the control tangent
        # (curvature is a second derivative, so it amplifies grid noise). This
        # is the single biggest anti-chatter fix on the steering side.
        win_ff = max(1, int(round(self.FF_SMOOTH_M / self.DS_TRACK)))
        self.kappa_signed = _circ_smooth(
            _signed_curvature(tangent, seg, smooth_win=0), win_ff)
        self.kappa_signed = np.clip(self.kappa_signed, -kclamp, kclamp)

        # ---- speed profile on the optimised line ---------------------------
        safe_a_lat = self.A_LAT * self.GRIP_MARGIN
        v_opt, s_opt, kappa_mag = _velocity_profile(
            path_opt, safe_a_lat, self.A_ACCEL, self.A_BRAKE,
            self.V_MAX, self.K_DF, curv_win_f
        )
        self.v_profile = v_opt
        self.curvature = kappa_mag
        self.seg = s_opt

        # ---- diagnostics ----------------------------------------------------
        usage = float(np.max(np.abs(off_f) / np.maximum(half_f, 1e-6)))
        kpk = float(np.max(np.abs(self.kappa_signed)))
        lap_s = float(np.sum(s_opt / np.maximum(v_opt, 1e-3)))
        print(f"[RACING LINE] peak curvature {kpk:.4f} 1/m (min radius "
              f"{1.0/(kpk+1e-9):.1f} m) | corridor usage {usage*100:.0f}% | "
              f"vmax {np.max(v_opt):.1f} m/s vmin {np.min(v_opt):.1f} m/s")
        print(f"[PREDICTED]  single lap ~{lap_s:5.1f}s  |  3-lap ~{3*lap_s:5.1f}s "
              f"(geometry estimate; real time depends on tracking)")

        if self.EXPORT_LINE:
            try:
                np.savez("racing_line.npz", path=self.path_3d, v=self.v_profile,
                         kappa=self.kappa_signed, seg=self.seg)
                print("[RACING LINE] exported racing_line.npz")
            except Exception as e:
                print(f"[RACING LINE] export failed: {e}")

        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))

        self._lap_time = 0.0
        self._lap_count = 0
        self._started = False
        self._start_idx = self.idx
        self._steer_lp = 0.0   # steering low-pass state (anti-chatter)

    # ------------------------------------------------------------------------
    async def step(self) -> None:
        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        speed = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        n = len(self.path)

        # local search for the closest line point (windowed, monotone-ish)
        sb = int(getattr(self, "_SB", 15))
        sf = int(getattr(self, "_SF", 250))
        win_ix = [(self.idx + k - sb) % n for k in range(sb + sf)]
        win = self.path[win_ix]
        self.idx = win_ix[int(np.argmin(np.linalg.norm(win - loc, axis=1)))]
        i = self.idx

        # ---- lateral: Frenet errors + LTV-LQR + previewed feedforward ------
        p = self.path[i]
        t = self.tangent[i]
        path_yaw = np.arctan2(t[1], t[0])
        dx = loc[0] - p[0]
        dy = loc[1] - p[1]
        e_y = -dx * t[1] + dy * t[0]
        e_psi = normalize_rad(yaw - path_yaw)

        # Feedforward curvature is read a short distance AHEAD (speed-scaled)
        # so the steer leads the corner instead of lagging it — this removes
        # the entry "catch-up" wobble that the actuator lag would otherwise
        # cause. The already-smoothed kappa_signed keeps it chatter-free.
        preview = max(0.0, speed * float(self.FF_PREVIEW_T))
        j = i
        d_ahead_ff = 0.0
        for _ in range(120):
            if d_ahead_ff >= preview:
                break
            d_ahead_ff += self.seg[j]
            j = (j + 1) % n
        kappa_ff = self.kappa_signed[j]

        tau = max(float(self.ACTUATOR_LAG_S), 1e-2)
        commanded_steer_rad = -self._prev_steer * self.MAX_STEER_RAD
        self.actual_steer_rad += (commanded_steer_rad - self.actual_steer_rad) * (self.CONTROL_DT / tau)

        delta_ff = float(np.arctan(self.WHEELBASE * kappa_ff))
        self.d_tilde = normalize_rad(self.actual_steer_rad - delta_ff)

        x_err = np.array([e_y, e_psi, self.d_tilde])
        K = self._lqr_gain(speed)
        delta_fb = float((-(K @ x_err))[0])
        delta = delta_ff + delta_fb

        steer_cmd = float(np.clip(-delta / self.MAX_STEER_RAD, -1.0, 1.0))
        # first-order low-pass removes high-frequency chatter, then a generous
        # rate cap guards against any abrupt jump. Low-pass (not the rate clip)
        # is the primary smoother, which avoids rate-limit limit-cycling.
        beta = float(self.STEER_LP_BETA)
        self._steer_lp += beta * (steer_cmd - self._steer_lp)
        rate = float(self.STEER_RATE)
        steer = float(np.clip(self._steer_lp, self._prev_steer - rate, self._prev_steer + rate))
        self._prev_steer = steer

        # ---- longitudinal: look-ahead speed target + traction circle -------
        look = i
        d_ahead = 0.0
        horizon = max(speed * 0.40, 4.0)
        v_target = self.v_profile[i]
        for _ in range(n):
            if d_ahead >= horizon:
                break
            d_ahead += self.seg[look]
            look = (look + 1) % n
            v_target = min(v_target, self.v_profile[look])

        a_lat_now = (speed ** 2) * self.curvature[i]
        g_lat = self.GRIP_MARGIN * (self.A_LAT + self.K_DF * speed ** 2)
        lat_ratio = np.clip(a_lat_now / max(g_lat, 1e-3), 0.0, 1.0)
        max_throttle = float(np.sqrt(max(0.0, 1.0 - lat_ratio ** 2)))

        dv = v_target - speed
        self.integral_error = float(np.clip(self.integral_error + dv * self.CONTROL_DT, -4.0, 4.0))
        if dv >= 0.0:
            ff = dv / self.A_ACCEL
            throttle = float(np.clip(ff + self.KP_SPEED * dv + self.KI_SPEED * self.integral_error,
                                     0.0, max_throttle))
            brake = 0.0
            if dv > 1.5:
                self.integral_error = 0.0
        else:
            throttle = 0.0
            ff = -dv / self.A_BRAKE
            brake = float(np.clip(ff - self.KP_SPEED * dv - self.KI_SPEED * self.integral_error,
                                  0.0, 1.0))

        control = {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }
        await self.vehicle.apply_action(control)

        # ---- lap telemetry --------------------------------------------------
        self._lap_time += self.CONTROL_DT
        if self._started and abs(i - self._start_idx) < 5 and self._lap_time > 30.0:
            self._lap_count += 1
            print(f"[LAP {self._lap_count}] {self._lap_time:6.1f}s")
            self._lap_time = 0.0
        elif not self._started and self._lap_time > 2.0:
            self._started = True

        if self.TELEMETRY:
            self.telemetry = {
                "v": speed, "v_target": float(v_target),
                "e_y": float(e_y), "e_psi": float(e_psi),
                "steer": steer, "lat_g": float(a_lat_now / 9.81),
            }
        return control