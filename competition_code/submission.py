import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from typing import List, Tuple, Dict
import roar_py_interface

# =============================================================================
#  AUTOTUNER HOOK
#  The optimizer sets `submission.OVERRIDES = {...}` before constructing the
#  solution. Any name in TUNABLE is read from here, falling back to the class
#  default. With OVERRIDES empty this runs the in-file baseline / SECTION_PARAMS.
# =============================================================================
OVERRIDES: Dict[str, float] = {}

TUNABLE = [
    "GRIP_MARGIN", "K_DF", "A_BRAKE", "A_ACCEL", "V_MAX",
    "STANLEY_K", "STANLEY_K_SOFT", "PID_KP", "PID_KI", "ACTUATOR_LAG_S",
    "SAFETY_BUFFER", "CURV_WIN_M",
]

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

def _tangents_normals(path: np.ndarray, smooth_win: int = 0) -> Tuple[np.ndarray, np.ndarray]:
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

def _min_curvature_offsets(center, normal0, half_width, n_iter=4, reg_smooth=0.05, min_h_frac=0.1):
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
        h = np.maximum(_seg_lengths(path), min_h)
        hm = h[im1]
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
        res = lsq_linear(A, b, bounds=(-half_width, half_width), max_iter=300, tol=1e-4)
        alpha = res.x
        path = center + alpha[:, None] * normal0
    return alpha

def _velocity_profile(path, A_LAT, A_ACCEL, A_BRAKE, V_MAX, K_DF, curv_win, passes=5):
    """
    Forward/backward speed-limit pass. Each grip/accel argument may be a SCALAR
    or a PER-POINT ARRAY (length == len(path)); arrays let the limits differ by
    lap section. Scalars are broadcast, so this works either way.
    """
    n = len(path)
    A_LAT   = np.broadcast_to(np.asarray(A_LAT, float),   (n,)).copy()
    A_ACCEL = np.broadcast_to(np.asarray(A_ACCEL, float), (n,)).copy()
    A_BRAKE = np.broadcast_to(np.asarray(A_BRAKE, float), (n,)).copy()
    V_MAX   = np.broadcast_to(np.asarray(V_MAX, float),   (n,)).copy()
    K_DF    = np.broadcast_to(np.asarray(K_DF, float),    (n,)).copy()

    kappa = _windowed_curvature(path, curv_win)
    ks = 0.25*np.roll(kappa, 1) + 0.5*kappa + 0.25*np.roll(kappa, -1)
    seg = np.maximum(np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1), 1e-3)

    # Apex cap with downforce: v^2*ks <= A_LAT + K_DF*v^2 -> v = sqrt(A_LAT/(ks-K_DF)).
    denom = np.maximum(ks - K_DF, 1e-4)
    v = np.minimum(np.sqrt(A_LAT / denom), V_MAX)

    for _ in range(passes):
        for i in range(n - 1, -1, -1):                # backward (braking-limited)
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i]**2
            a_lat = min(v[i]**2 * ks[i], g_lat)
            a_lon = A_BRAKE[i] * np.sqrt(max(0.0, 1.0 - (a_lat/g_lat)**2))
            v[i] = min(v[i], np.sqrt(v[j]**2 + 2*a_lon*seg[i]))
        for i in range(n):                            # forward (traction-limited)
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i]**2
            a_lat = min(v[i]**2 * ks[i], g_lat)
            a_lon = A_ACCEL[i] * np.sqrt(max(0.0, 1.0 - (a_lat/g_lat)**2))
            v[j] = min(v[j], np.sqrt(v[i]**2 + 2*a_lon*seg[i]))
    return v, seg, ks

# =============================================================================
#  SOLUTION  (Dallara) — per-section gains, sector timing, autotuner hook
# =============================================================================
class RoarCompetitionSolution:

    # ----- PHYSICS / SPEED (scalar defaults; sections override below) --------
    A_LAT       = 13.0
    K_DF        = 0.0
    A_ACCEL     = 200.0
    A_BRAKE     = 30.3
    V_MAX       = 300.0
    GRIP_MARGIN = 2.05

    DS_OPT      = 2.0
    DS_TRACK    = 0.5
    CURV_WIN_M  = 5.0
    WHEELBASE   = 2.875
    TELEMETRY   = True

    # ----- CONTROLLER GAINS -------------------------------------------------
    STANLEY_K       = 5.0
    STANLEY_K_SOFT  = 2.0
    PID_KP          = 256.0
    PID_KI          = 0.1
    ACTUATOR_LAG_S  = 0.4

    # ----- CORRIDOR ---------------------------------------------------------
    CAR_HALF_WIDTH = 0.95
    SAFETY_BUFFER  = 0.65
    LANE_MARGIN    = CAR_HALF_WIDTH + SAFETY_BUFFER
    MAX_OFFSET_RADIUS_FRAC = 0.7
    SMOOTH_WIN_M = 1.5

    # =========================================================================
    #  PER-SECTION TUNING
    #  Lap split into N_SECTIONS equal-distance sections (by arc length on the
    #  racing line). For any knob, give a list of N_SECTIONS values; omit it to
    #  use the scalar default. Supported:
    #    plan-time : GRIP_MARGIN, A_LAT, K_DF, A_BRAKE, A_ACCEL, V_MAX
    #    live ctrl : STANLEY_K, STANLEY_K_SOFT, PID_KP, PID_KI, ACTUATOR_LAG_S
    #  Startup prints each section's metre range; the lap printout gives splits.
    # =========================================================================
    N_SECTIONS = 5
    SECTION_PARAMS: Dict[str, list] = {
        #                  S1     S2     S3     S4     S5
        "A_LAT":          [18.0,  16.0,  18.0,  18.0,  18.5],
        "K_DF":           [0.0,   0.0,   0.0,   0.0,   0.0],
        "A_ACCEL":        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        "V_MAX":          [300.0, 300.0, 200.0, 300.0, 300.0],
        "GRIP_MARGIN":    [1.8,  2.2,  1.8,  2.3,  2.4],
        "STANLEY_K":      [5.0,   3.0,   4,   3.0,   3.0],
        "STANLEY_K_SOFT": [4.0,   2.0,   3.0,   4.0,   1.5],
        "PID_KP":         [256.0, 256.0, 128.0, 256.0, 200.0],
        "PID_KI":         [0.5,   0.5,   0.5,   0.5,   0.5],
        "ACTUATOR_LAG_S": [0.38,  0.38,  0.38,  0.38,  0.38],
    }

    CONTROL_DT = 0.05   # sim seconds per control step (matches the runner)

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
        self.telemetry = {}
        # Apply autotuner overrides (instance attrs shadow class defaults).
        for name in TUNABLE:
            if name in OVERRIDES:
                setattr(self, name, float(OVERRIDES[name]))
        self.LANE_MARGIN = self.CAR_HALF_WIDTH + self.SAFETY_BUFFER

    # Per-section scalar lookup.
    def _sec(self, name, sec):
        vals = self.SECTION_PARAMS.get(name)
        if vals is None:
            return float(getattr(self, name))
        return float(vals[int(sec)])

    async def initialize(self) -> None:
        wps = self.maneuverable_waypoints
        center_raw_3d = np.array([w.location[:3] for w in wps])
        center_raw = center_raw_3d[:, :2]
        width_raw = np.array([float(getattr(w, "lane_width", 6.0)) for w in wps])

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

        half_o = np.maximum(width_o / 2.0 - self.LANE_MARGIN, 0.0)
        half_f = np.maximum(width_f / 2.0 - self.LANE_MARGIN, 0.0)
        half_o = np.minimum(half_o, self.MAX_OFFSET_RADIUS_FRAC / (kappa_o + 1e-6))
        half_f = np.minimum(half_f, self.MAX_OFFSET_RADIUS_FRAC / (kappa_f + 1e-6))

        # ----- Racing line (NOT affected by per-section speeds) --------------
        alpha_o = _min_curvature_offsets(center_o, normal_o, half_o)
        u_o = _progress(center_o)
        u_f = _progress(center_f)
        off_f = np.interp(u_f, np.append(u_o, 1.0), np.append(alpha_o, alpha_o[0]))
        off_f = np.clip(off_f, -half_f, half_f)
        path_opt = center_f + off_f[:, None] * normal_f

        n_fine = len(path_opt)

        # ----- Section assignment (equal arc-length on the racing line) ------
        prog = _progress(path_opt)
        self.section_of = np.minimum(self.N_SECTIONS - 1,
                                     (prog * self.N_SECTIONS).astype(int))

        def sec_arr(name):
            base = float(getattr(self, name))
            vals = self.SECTION_PARAMS.get(name)
            if vals is None:
                return np.full(n_fine, base)
            return np.asarray(vals, float)[self.section_of]

        # ----- Per-point plan limits from section values --------------------
        A_LAT_arr  = sec_arr("A_LAT")
        GM_arr     = sec_arr("GRIP_MARGIN")
        KDF_arr    = sec_arr("K_DF")
        ABRAKE_arr = sec_arr("A_BRAKE")
        AACCEL_arr = sec_arr("A_ACCEL")
        VMAX_arr   = sec_arr("V_MAX")
        safe_a_lat = A_LAT_arr * GM_arr

        v_opt, s_opt, kappa_opt = _velocity_profile(
            path_opt, safe_a_lat, AACCEL_arr, ABRAKE_arr, VMAX_arr, KDF_arr, curv_win_f)

        self.path = path_opt
        self.tangent = _tangents_normals(path_opt, smooth_win=0)[0]
        self.path_3d = np.column_stack([path_opt, center_f_3d[:, 2]])
        self.v_profile = v_opt
        self.curvature = kappa_opt
        self.seg = s_opt

        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))

        # ----- Sector-timing state -------------------------------------------
        self._lap_sec_time = np.zeros(self.N_SECTIONS)
        self._total_sec_time = np.zeros(self.N_SECTIONS)
        self._last_sec = int(self.section_of[self.idx])
        self._lap_count = 0

        total_len = float(np.sum(_seg_lengths(path_opt)))
        bounds = [k * total_len / self.N_SECTIONS for k in range(self.N_SECTIONS + 1)]
        ranges = "  ".join(f"S{k+1}:{bounds[k]:4.0f}-{bounds[k+1]:4.0f}m"
                           for k in range(self.N_SECTIONS))
        print(f"[SECTIONS] lap length {total_len:.0f} m | {ranges}")

    @staticmethod
    def _arclen(path):
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    async def step(self) -> None:
        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        speed = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        n = len(self.path)

        # 1. Front axle position + nearest path index
        fx = loc[0] + self.WHEELBASE * np.cos(yaw)
        fy = loc[1] + self.WHEELBASE * np.sin(yaw)
        front_axle = np.array([fx, fy])

        win_ix = [(self.idx + k - 50) % n for k in range(450)]
        win = self.path[win_ix]
        wd = np.linalg.norm(win - front_axle, axis=1)
        self.idx = win_ix[int(np.argmin(wd))]

        # 2. This section's gains
        sec = int(self.section_of[self.idx])
        stanley_k      = self._sec("STANLEY_K", sec)
        stanley_k_soft = self._sec("STANLEY_K_SOFT", sec)
        pid_kp         = self._sec("PID_KP", sec)
        pid_ki         = self._sec("PID_KI", sec)
        actuator_lag   = self._sec("ACTUATOR_LAG_S", sec)
        grip_margin    = self._sec("GRIP_MARGIN", sec)
        a_lat_c        = self._sec("A_LAT", sec)
        k_df_c         = self._sec("K_DF", sec)
        a_accel_c      = self._sec("A_ACCEL", sec)
        a_brake_c      = self._sec("A_BRAKE", sec)

        # 3. Local frame + Stanley steering
        closest_pt = self.path[self.idx]
        dx = closest_pt[0] - front_axle[0]
        dy = closest_pt[1] - front_axle[1]
        local_y = -dx * np.sin(yaw) + dy * np.cos(yaw)

        tangent_vec = self.tangent[self.idx]
        path_yaw = np.arctan2(tangent_vec[1], tangent_vec[0])
        heading_error = normalize_rad(path_yaw - yaw)
        crosstrack_error = local_y

        dyn_k = stanley_k * (1.0 - np.clip(speed / self.V_MAX, 0.0, 0.5))
        raw_steer_angle = heading_error + np.arctan2(dyn_k * crosstrack_error, speed + stanley_k_soft)
        steer = float(np.clip(-raw_steer_angle, -1.0, 1.0))

        # 4. Actuator-lag lookahead for target speed
        look = self.idx
        d_ahead = 0.0
        horizon = max(speed * actuator_lag, 2.0)
        v_target = self.v_profile[self.idx]
        for _ in range(n):
            if d_ahead >= horizon:
                break
            d_ahead += self.seg[look]
            look = (look + 1) % n
            v_target = min(v_target, self.v_profile[look])

        # 5. Friction-circle throttle cap
        local_curvature = self.curvature[self.idx]
        current_a_lat = (speed ** 2) * local_curvature
        g_lat = grip_margin * (a_lat_c + k_df_c * speed ** 2)
        lat_ratio = np.clip(current_a_lat / max(g_lat, 1e-3), 0.0, 1.0)
        max_throttle_allowed = float(np.sqrt(max(0.0, 1.0 - lat_ratio ** 2)))

        # 6. Feedforward + PI
        dv = v_target - speed
        self.integral_error += dv * 0.05
        self.integral_error = np.clip(self.integral_error, -4.0, 4.0)

        if dv >= 0.0:
            ff_throttle = dv / a_accel_c
            raw_throttle = ff_throttle + pid_kp * dv + pid_ki * self.integral_error
            throttle = float(np.clip(raw_throttle, 0.0, max_throttle_allowed))
            brake = 0.0
            if dv > 1.5:
                self.integral_error = 0.0
        else:
            throttle = 0.0
            ff_brake = -dv / a_brake_c
            raw_brake = ff_brake - pid_kp * dv - pid_ki * self.integral_error
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

        # 7. Sector timing -> per-lap breakdown
        self._lap_sec_time[sec] += self.CONTROL_DT
        self._total_sec_time[sec] += self.CONTROL_DT
        if self._last_sec == self.N_SECTIONS - 1 and sec == 0 \
                and self._lap_sec_time.sum() > 30.0:
            self._lap_count += 1
            lap_total = float(self._lap_sec_time.sum())
            parts = "  ".join(f"S{k+1} {self._lap_sec_time[k]:5.1f}s"
                              for k in range(self.N_SECTIONS))
            print(f"[LAP {self._lap_count}] total {lap_total:6.1f}s | {parts}")
            self._lap_sec_time = np.zeros(self.N_SECTIONS)
        self._last_sec = sec

        self.telemetry = {
            "section": sec + 1,
            "target_speed": float(v_target),
            "lat_g": float(current_a_lat),
            "sector_times": [float(x) for x in self._lap_sec_time],
        }

        if self.TELEMETRY:
            a_lat_now = (speed ** 2) * abs(local_curvature)
            self._t_n = getattr(self, "_t_n", 0) + 1
            self._t_vmax = max(getattr(self, "_t_vmax", 0.0), speed)
            self._t_amax = max(getattr(self, "_t_amax", 0.0), a_lat_now)
            if self._t_n % 200 == 0:
                pass
                #print(f"[TELEMETRY] top {self._t_vmax:5.1f} m/s | "
                      #f"peak lat {self._t_amax:4.1f} ({self._t_amax/9.81:4.2f}g)")
        return control