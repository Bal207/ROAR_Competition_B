import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from typing import List, Tuple, Dict
import roar_py_interface

def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

# =============================================================================
#  GEOMETRY & TRAJECTORY OPTIMIZATION HELPERS
# =============================================================================
def _resample_closed(points: np.ndarray, ds: float) -> np.ndarray:
    loop = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(loop[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(round(total / ds)))
    s_new = np.linspace(0.0, total, n, endpoint=False)

    resampled = np.zeros((n, points.shape[1]))
    for dim in range(points.shape[1]):
        resampled[:, dim] = np.interp(s_new, s, loop[:, dim])
    return resampled

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
    """Normalized arc-length progress in [0, 1) for a closed path."""
    seg = _seg_lengths(path)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s[:-1] / s[-1]

def _circ_smooth(arr: np.ndarray, win: int) -> np.ndarray:
    """Circular moving-average smoothing of a 1-D periodic signal."""
    if win < 1:
        return arr
    k = 2 * win + 1
    kernel = np.ones(k) / k
    pad = np.concatenate([arr[-win:], arr, arr[:win]])
    return np.convolve(pad, kernel, mode="valid")

def _tangents_normals(path: np.ndarray, smooth_win: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central-difference tangents/normals. With smooth_win > 0 the tangent field
    is circularly averaged BEFORE normalization, so discretization noise in the
    raw map can't make the normals oscillate and the offset corridor can't
    crisscross at sharp apexes.
    """
    t = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    if smooth_win and smooth_win > 0:
        t = np.column_stack([_circ_smooth(t[:, 0], smooth_win),
                             _circ_smooth(t[:, 1], smooth_win)])
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n

def _windowed_curvature(path: np.ndarray, win: int) -> np.ndarray:
    p0 = np.roll(path, win, axis=0)
    p1 = path
    p2 = np.roll(path, -win, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area = 0.5 * np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1]) - (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1]))
    return 4.0 * area / (a * b * c + 1e-9)

def _min_curvature_offsets(center: np.ndarray,
                           normal0: np.ndarray,
                           half_width: np.ndarray,
                           n_iter: int = 4,
                           reg_smooth: float = 0.05,
                           min_h_frac: float = 0.1) -> np.ndarray:
    """
    Iterated arc-length-weighted minimum-curvature offset solver.

    Path is P = center + alpha * normal0; offsets are measured along the FIXED
    centerline normals so the box bounds stay exact w.r.t. the corridor.

    Curvature is the arc-length-weighted second derivative, whose weights depend
    on the current point spacing h. When the solver clips an apex the inside
    points crowd and h shrinks; we floor h at a fraction of nominal spacing so
    the weights 2/(h_{i-1}+h_i) can't blow up and destabilize lsq_linear.
    """
    n = len(center)
    idx = np.arange(n)
    im1 = (idx - 1) % n
    ip1 = (idx + 1) % n

    Nx = sp.diags(normal0[:, 0])
    Ny = sp.diags(normal0[:, 1])

    Dsm = sp.csr_matrix(
        (np.concatenate([np.ones(n), -np.ones(n)]),
         (np.concatenate([idx, idx]), np.concatenate([idx, ip1]))),
        shape=(n, n)) * reg_smooth

    ds_nom = float(np.mean(_seg_lengths(center)))
    min_h = max(0.1, min_h_frac * ds_nom)

    alpha = np.zeros(n)
    path = center.copy()
    for _ in range(n_iter):
        h = np.maximum(_seg_lengths(path), min_h)   # |P_i -> P_{i+1}|
        hm = h[im1]                                 # |P_{i-1} -> P_i|
        w = 2.0 / (hm + h)
        c_im1 = w / hm
        c_i = -w * (1.0 / h + 1.0 / hm)
        c_ip1 = w / h
        M = sp.csr_matrix(
            (np.concatenate([c_im1, c_i, c_ip1]),
             (np.concatenate([idx, idx, idx]),
              np.concatenate([im1, idx, ip1]))),
            shape=(n, n))

        Ax = M @ Nx
        Ay = M @ Ny
        bx = -(M @ center[:, 0])
        by = -(M @ center[:, 1])

        A = sp.vstack([Ax, Ay, Dsm]).tocsr()
        b = np.concatenate([bx, by, np.zeros(n)])

        res = lsq_linear(A, b, bounds=(-half_width, half_width),
                         max_iter=300, tol=1e-4)
        alpha = res.x
        path = center + alpha[:, None] * normal0
    return alpha

def _velocity_profile(path, A_LAT, A_ACCEL, A_BRAKE, V_MAX, K_DF, curv_win, passes=5):
    kappa = _windowed_curvature(path, curv_win)
    ks = 0.25*np.roll(kappa, 1) + 0.5*kappa + 0.25*np.roll(kappa, -1)
    seg = np.maximum(np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1), 1e-3)
    n = len(path)

    denom = np.maximum(ks - K_DF, 1e-4)
    v = np.minimum(np.sqrt(A_LAT / denom), V_MAX)

    for _ in range(passes):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            a_lat = min(v[i]**2 * ks[i], A_LAT)
            a_lon = A_BRAKE * np.sqrt(max(0.0, 1.0 - (a_lat/A_LAT)**2))
            v[i] = min(v[i], np.sqrt(v[j]**2 + 2*a_lon*seg[i]))
        for i in range(n):
            j = (i + 1) % n
            a_lat = min(v[i]**2 * ks[i], A_LAT)
            a_lon = A_ACCEL * np.sqrt(max(0.0, 1.0 - (a_lat/A_LAT)**2))
            v[j] = min(v[j], np.sqrt(v[i]**2 + 2*a_lon*seg[i]))
    return v, seg, ks

# =============================================================================
#  ROBUST LOCAL-FRAME TRACKING SOLUTION FOR TESLA MODEL 3
# =============================================================================
class RoarCompetitionSolution:

    # ----- CARLA TESLA MODEL 3 PHYSICALLY DERIVED CONSTANTS ------------------
    A_LAT       = 10.1     # Max lateral tire grip (~1.03G)
    GRIP_MARGIN = 0.95
    A_ACCEL     = 7.5      # Smooth electric motor torque ramp
    A_BRAKE     = 11.8     # ABS brake limit allocation
    V_MAX       = 72.2     # 260 km/h cap
    K_DF        = 0.0

    DS_OPT      = 2.0
    DS_TRACK    = 0.5
    CURV_WIN_M  = 5.0
    WHEELBASE   = 2.875    # Exact Tesla Model 3 Wheelbase

    # ----- SYSTEM GAINS -----------------------------------------------------
    STANLEY_K       = 1.2
    STANLEY_K_SOFT  = 3.0
    PID_KP          = 2.0
    PID_KI          = 0.1

    # ----- CORRIDOR MARGIN --------------------------------------------------
    # Constant physical clearance only (does NOT grow with curvature).
    CAR_HALF_WIDTH = 0.95
    SAFETY_BUFFER  = 0.65
    LANE_MARGIN    = CAR_HALF_WIDTH + SAFETY_BUFFER

    # Geometric guard: never offset inward past a fraction of the local radius
    # of curvature, or the inside of the corridor self-intersects. Generous on
    # purpose - only binds below ~R=4-5 m, so normal corners are untouched.
    MAX_OFFSET_RADIUS_FRAC = 0.7

    # Tangent/normal smoothing length (m) for CONSTRUCTION NORMALS ONLY, to
    # suppress raw-map discretization noise without distorting offset direction.
    # NOTE: this is deliberately NOT applied to the controller heading reference
    # (self.tangent) - smoothing that lags turn-in and makes Stanley understeer.
    SMOOTH_WIN_M = 1.5

    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.integral_error = 0.0
        self.idx = 0

    async def initialize(self) -> None:
        wps = self.maneuverable_waypoints
        center_raw_3d = np.array([w.location[:3] for w in wps])
        center_raw = center_raw_3d[:, :2]
        width_raw = np.array([float(getattr(w, "lane_width", 6.0)) for w in wps])

        # ----- Coarse grid (solve) and fine grid (track) ---------------------
        center_o_3d = _resample_closed(center_raw_3d, self.DS_OPT)
        center_o = center_o_3d[:, :2]
        width_o = _resample_scalar_closed(width_raw, center_raw, self.DS_OPT)

        center_f_3d = _resample_closed(center_raw_3d, self.DS_TRACK)
        center_f = center_f_3d[:, :2]
        width_f = _resample_scalar_closed(width_raw, center_raw, self.DS_TRACK)

        win_o = max(1, int(round(self.SMOOTH_WIN_M / self.DS_OPT)))
        win_f = max(1, int(round(self.SMOOTH_WIN_M / self.DS_TRACK)))
        _, normal_o = _tangents_normals(center_o, smooth_win=win_o)
        _, normal_f = _tangents_normals(center_f, smooth_win=win_f)

        curv_win_o = max(1, int(round(self.CURV_WIN_M / self.DS_OPT)))
        curv_win_f = max(1, int(round(self.CURV_WIN_M / self.DS_TRACK)))
        kappa_o = _windowed_curvature(center_o, curv_win_o)
        kappa_f = _windowed_curvature(center_f, curv_win_f)

        # ----- Feasible lateral band -----------------------------------------
        # Constant margin, plus the radius guard so the inside never collapses.
        half_o = np.maximum(width_o / 2.0 - self.LANE_MARGIN, 0.0)
        half_f = np.maximum(width_f / 2.0 - self.LANE_MARGIN, 0.0)
        half_o = np.minimum(half_o, self.MAX_OFFSET_RADIUS_FRAC / (kappa_o + 1e-6))
        half_f = np.minimum(half_f, self.MAX_OFFSET_RADIUS_FRAC / (kappa_f + 1e-6))

        safe_a_lat = self.A_LAT * self.GRIP_MARGIN

        # ----- Iterated minimum-curvature solve on the coarse grid -----------
        alpha_o = _min_curvature_offsets(center_o, normal_o, half_o)

        # ----- Transfer offsets coarse -> fine in NORMALIZED progress space --
        # Both grids run 0..1 with an explicit periodic knot, so there's no
        # perimeter-scale drift and no premature modulo wrap/seam.
        u_o = _progress(center_o)
        u_f = _progress(center_f)
        off_f = np.interp(u_f, np.append(u_o, 1.0), np.append(alpha_o, alpha_o[0]))
        off_f = np.clip(off_f, -half_f, half_f)   # re-clip on the fine band
        path_opt = center_f + off_f[:, None] * normal_f

        v_opt, s_opt, kappa_opt = _velocity_profile(
            path_opt, safe_a_lat, self.A_ACCEL, self.A_BRAKE,
            self.V_MAX, self.K_DF, curv_win_f)

        self.path = path_opt
        # Controller heading reference MUST be the true local path direction.
        # Do not smooth this - a smoothed tangent lags turn-in and makes the
        # Stanley feedforward understeer into the outer wall on every corner.
        self.tangent = _tangents_normals(path_opt, smooth_win=0)[0]
        self.path_3d = np.column_stack([path_opt, center_f_3d[:, 2]])
        self.v_profile = v_opt
        self.curvature = kappa_opt
        self.seg = s_opt

        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))

    @staticmethod
    def _arclen(path):
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    async def step(self) -> None:
        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        speed = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        n = len(self.path)

        # 1. Front Axle Kinematic Position
        fx = loc[0] + self.WHEELBASE * np.cos(yaw)
        fy = loc[1] + self.WHEELBASE * np.sin(yaw)
        front_axle = np.array([fx, fy])

        win_ix = [(self.idx + k - 50) % n for k in range(450)]
        win = self.path[win_ix]
        wd = np.linalg.norm(win - front_axle, axis=1)
        self.idx = win_ix[int(np.argmin(wd))]

        # 2. Local Vehicle Frame Transformation
        closest_pt = self.path[self.idx]
        dx = closest_pt[0] - front_axle[0]
        dy = closest_pt[1] - front_axle[1]

        local_x = dx * np.cos(yaw) + dy * np.sin(yaw)
        local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

        tangent_vec = self.tangent[self.idx]
        path_yaw = np.arctan2(tangent_vec[1], tangent_vec[0])
        heading_error = normalize_rad(path_yaw - yaw)
        crosstrack_error = local_y

        # 3. Combined Stanley Law Matrix
        dyn_k = self.STANLEY_K * (1.0 - np.clip(speed / self.V_MAX, 0.0, 0.5))

        raw_steer_angle = heading_error + np.arctan2(dyn_k * crosstrack_error, speed + self.STANLEY_K_SOFT)

        # FIX APPLIED HERE: Invert the steering mapping.
        # Standard math outputs a positive angle for Left, but the CARLA actuator expects negative for Left.
        steer_angle = -raw_steer_angle

        steer = float(np.clip(steer_angle, -1.0, 1.0))

        # 4. Long-Range Velocity Horizon Lookahead
        look = self.idx
        d_ahead = 0.0
        horizon = max(speed * 0.85, 10.0)
        v_target = self.v_profile[self.idx]

        for _ in range(n):
            if d_ahead >= horizon:
                break
            d_ahead += self.seg[look]
            look = (look + 1) % n
            v_target = min(v_target, self.v_profile[look])

        # 5. Active Traction Control Allocation
        local_curvature = self.curvature[self.idx]
        current_a_lat = (speed ** 2) * local_curvature

        grip_margin = self.A_LAT - current_a_lat
        if grip_margin < 2.0:
            max_throttle_allowed = max(0.0, grip_margin / 2.0)
        else:
            max_throttle_allowed = 1.0

        # 6. Feedforward + PI Execution
        dv = v_target - speed
        self.integral_error += dv * 0.05
        self.integral_error = np.clip(self.integral_error, -4.0, 4.0)

        if dv >= 0.0:
            ff_throttle = dv / self.A_ACCEL
            raw_throttle = ff_throttle + self.PID_KP * dv + self.PID_KI * self.integral_error
            throttle = float(np.clip(raw_throttle, 0.0, max_throttle_allowed))
            brake = 0.0
            if dv > 1.5:
                self.integral_error = 0.0
        else:
            throttle = 0.0
            ff_brake = -dv / self.A_BRAKE
            raw_brake = ff_brake - self.PID_KP * dv - self.PID_KI * self.integral_error
            brake = float(np.clip(raw_brake, 0.0, 1.0))

        control = {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }
        await self.vehicle.apply_action(control)
        return control