"""
ROAR competition solution — LTV-LQR controller + Iterative Physics-Aware FISTA Racing Line.
With 7-Segment Gain Scheduling for Monza v1.
"""

from typing import List, Dict
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
import roar_py_interface

def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

# =============================================================================
#  GEOMETRY & TRAJECTORY HELPERS
# =============================================================================
def _resample_closed(points: np.ndarray, ds: float) -> np.ndarray:
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
    area = 0.5 * np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1])
                        - (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1]))
    return 4.0 * area / (a * b * c + 1e-9)

def _signed_curvature(tangent: np.ndarray, seg: np.ndarray, smooth_win: int = 0) -> np.ndarray:
    heading = np.arctan2(tangent[:, 1], tangent[:, 0])
    dth = normalize_rad(np.roll(heading, -1) - heading)
    kappa = dth / np.maximum(seg, 1e-3)
    return _circ_smooth(kappa, smooth_win) if smooth_win > 0 else kappa

def _fista_raceline(kappa_center, usable, ds, weight=None, iters=2500, w_smooth=0.0, seed=0):
    n = len(kappa_center)
    idx = np.arange(n)
    im1 = (idx - 1) % n
    ip1 = (idx + 1) % n
    D2 = sp.csr_matrix(
        (np.concatenate([np.ones(n), -2 * np.ones(n), np.ones(n)]),
         (np.concatenate([idx, idx, idx]),
          np.concatenate([im1, idx, ip1]))),
        shape=(n, n)) / (ds * ds)
    kc = np.clip(kappa_center, -0.5, 0.5)
    M = (D2 + sp.diags(kc ** 2)).tocsr()
    k0 = kappa_center - np.mean(kappa_center)

    w = np.ones(n) if weight is None else np.clip(np.asarray(weight, float), 1e-3, None)
    W = sp.diags(w)
    A = (M.T @ (W @ M)).tocsr()
    b = -(M.T @ (W @ k0))
    if w_smooth > 0:
        D1 = sp.csr_matrix(
            (np.concatenate([np.ones(n), -np.ones(n)]),
             (np.concatenate([idx, idx]), np.concatenate([ip1, idx]))),
            shape=(n, n)) / ds
        A = (A + (w_smooth ** 2) * (D1.T @ D1)).tocsr()

    x = np.random.default_rng(seed).standard_normal(n)
    for _ in range(40):
        x = A @ x
        x /= (np.linalg.norm(x) + 1e-12)
    Lip = float(x @ (A @ x)) / float(x @ x) * 1.02 + 1e-9

    ub = np.asarray(usable, float)
    lb = -ub
    a = np.zeros(n)
    y = a.copy()
    t = 1.0
    for _ in range(int(iters)):
        grad = A @ y - b
        a_new = np.clip(y - grad / Lip, lb, ub)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a = a_new
        t = t_new
    return a

def _velocity_profile(path, A_LAT, A_ACCEL, A_BRAKE, V_MAX, K_DF, curv_win, passes=6):
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
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i] ** 2
            a_lat = min(v[i] ** 2 * ks[i], g_lat)
            a_lon = max(0.15 * A_BRAKE[i],
                        A_BRAKE[i] * np.sqrt(max(0.0, 1.0 - (a_lat / g_lat) ** 2)))
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_lon * seg[i]))
        for i in range(n):
            j = (i + 1) % n
            g_lat = A_LAT[i] + K_DF[i] * v[i] ** 2
            a_lat = min(v[i] ** 2 * ks[i], g_lat)
            a_lon = A_ACCEL[i] * np.sqrt(max(0.0, 1.0 - (a_lat / g_lat) ** 2))
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_lon * seg[i]))
    return v, seg, ks

def _dare_gain(A, B, Q, R, iters=400, tol=1e-11):
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

    # =========================================================================
    #  7-SEGMENT GAIN SCHEDULING (MONZA)
    # =========================================================================
    # Track is divided based on normalized arc-length (0.0 to 1.0).
    # Tune A_LAT, A_BRAKE, Margins, and LQR tracking specific to each corner.
    TRACK_SEGMENTS = [
        {"name": "Start/T1 Chicane", "start_pct": 0.00, "A_LAT": 350.5, "A_ACCEL": 200.0, "A_BRAKE": 25.0, "V_MAX": 1000.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0, "Q_E": 10.0, "Q_PSI": 10.0}, #fast
        {"name": "Curva Grande",     "start_pct": 0.185, "A_LAT": 30.5, "A_ACCEL": 200.0, "A_BRAKE": 20.0, "V_MAX": 300.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0.062, "Q_E": 15.0, "Q_PSI": 15.0},
        {"name": "Roggia Chicane",   "start_pct": 0.35, "A_LAT": 10000.5, "A_ACCEL": 300.0, "A_BRAKE": 25.0, "V_MAX": 1000.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0, "Q_E": 5.0, "Q_PSI": 5.0}, #fast
        {"name": "Lesmo 1 & 2",      "start_pct": 0.45, "A_LAT": 35.5, "A_ACCEL": 50.0, "A_BRAKE": 18.0, "V_MAX": 135.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0.01, "Q_E": 13.0, "Q_PSI": 13.0},
        {"name": "Middle straight 2", "start_pct": 0.55, "A_LAT": 10000.5, "A_ACCEL": 300.0, "A_BRAKE": 25.0, "V_MAX": 1000.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0, "Q_E": 5.0, "Q_PSI": 5.0},
        {"name": "Serraglio",        "start_pct": 0.65, "A_LAT": 35.5, "A_ACCEL": 50.0, "A_BRAKE": 18.0, "V_MAX": 135.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0.01, "Q_E": 13.0, "Q_PSI": 13.0},
        {"name": "Ascari Chicane",   "start_pct": 0.77, "A_LAT": 1000.5, "A_ACCEL": 300.0, "A_BRAKE": 25.0, "V_MAX": 1000.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0, "Q_E": 3.0, "Q_PSI": 3.0},
        {"name": "Parabolica",       "start_pct": 0.935, "A_LAT": 35.5, "A_ACCEL": 50.0, "A_BRAKE": 18.0, "V_MAX": 135.0, "TRACK_MARGIN_MIN": 0, "LQR_DRIFT_COEFF": 0.01, "Q_E": 13.0, "Q_PSI": 13.0},
    ]

    # Global Parameters
    Q_DELTA  = 0.0
    R_STEER  = 2.5      
    ACTUATOR_LAG_S = 0.20
    MAX_STEER_RAD = 1.0
    V_FLOOR = 3.0
    STEER_LP_BETA = 0.65   
    STEER_RATE    = 0.35   
    FF_PREVIEW_T  = 0.12   
    FF_SMOOTH_M   = 4.0    
    K_DF        = 0.0     
    GRIP_MARGIN = 0.94    
    KP_SPEED = 256
    KI_SPEED = 0.5

    DS_OPT       = 2.0    
    DS_TRACK     = 0.5    
    CURV_WIN_M   = 5.0
    SMOOTH_WIN_M = 1.5

    AUTO_DETECT_VEHICLE_PARAMS = True
    WHEELBASE_FALLBACK_M       = 2.85   
    CAR_HALF_WIDTH_FALLBACK_M  = 0.97   
    WHEELBASE    = WHEELBASE_FALLBACK_M       
    CAR_HALF_WIDTH = CAR_HALF_WIDTH_FALLBACK_M  

    TRACK_MARGIN_MAX = 5.5   
    OFFTRACK_CAP     = 0.45  
    MIN_CORRIDOR     = 0.15  
    EXTRA_TRACK_WIDTH_M = 0.0 

    FISTA_ITERS    = 2500     
    N_MARGIN_ITERS = 3        
    MINTIME_REFINE = True     
    MINTIME_PASSES = 3        
    MINTIME_POWER  = 3.5      
    MINTIME_FLOOR  = 0.2      
    LINE_SMOOTH    = 0.0      
    WIDTH_MIN   = 5.0         
    WIDTH_MAX   = 25.0
    WIDTH_DEFAULT = 12.0
    KAPPA_FF_MAX = 0.50       
    CONTROL_DT = 0.05
    TELEMETRY  = True

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

    def _autodetect_vehicle_params(self) -> None:
        if not self.AUTO_DETECT_VEHICLE_PARAMS:
            return
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
        if raw_actor is None and hasattr(self.vehicle, "get_physics_control"):
            raw_actor = self.vehicle

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

            front_mid = 0.5 * (_pos(wheels[0]) + _pos(wheels[1]))
            rear_mid  = 0.5 * (_pos(wheels[2]) + _pos(wheels[3]))
            raw_wheelbase = float(np.linalg.norm(front_mid - rear_mid))
            raw_front_track = float(np.linalg.norm(_pos(wheels[0]) - _pos(wheels[1])))
            raw_rear_track  = float(np.linalg.norm(_pos(wheels[2]) - _pos(wheels[3])))
            wheel_radius_cm = float(np.mean([getattr(w, "radius", 35.0) for w in wheels]))

            scale = 1.0
            if length_m is not None and length_m > 0.5 and raw_wheelbase > 0:
                ratio = raw_wheelbase / length_m
                if ratio > 5.0:
                    scale = 0.01
                elif 0.2 <= ratio <= 1.05:
                    scale = 1.0
                else:
                    raise ValueError("out of sane range")

            wheelbase_m = raw_wheelbase * scale
            front_track_m = raw_front_track * scale
            rear_track_m = raw_rear_track * scale
            track_m = max(front_track_m, rear_track_m)

            wheelbase_m = float(np.clip(wheelbase_m, 1.5, 4.0))
            track_m = float(np.clip(track_m, 0.8, 2.2))
            tire_half_extent = track_m / 2.0 + wheel_radius_cm * 0.01
            candidates = [tire_half_extent]
            if width_m is not None and width_m > 0.3:
                candidates.append(width_m / 2.0)
            half_width_m = float(np.clip(max(candidates), 0.5, 1.3))

            self.WHEELBASE = wheelbase_m
            self.CAR_HALF_WIDTH = half_width_m
        except Exception as e:
            pass

    def _lqr_gain(self, v: float, q_e: float, q_psi: float) -> np.ndarray:
        key = (int(round(v)), q_e, q_psi)
        g = self._K_cache.get(key)
        if g is not None:
            return g
        vm = max(float(v), self.V_FLOOR)
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
        Q = np.diag([q_e, q_psi, self.Q_DELTA])
        R = np.array([[self.R_STEER]])
        K = _dare_gain(A, B, Q, R)
        self._K_cache[key] = K
        return K

    def _map_segments(self, u_array):
        """Maps normalized arc length to segment indices (0-6)"""
        sec_idx = np.zeros_like(u_array, dtype=int)
        for i, u in enumerate(u_array):
            assigned = 0
            for sec_i, sec_config in enumerate(self.TRACK_SEGMENTS):
                if u >= sec_config["start_pct"]:
                    assigned = sec_i
            sec_idx[i] = assigned
        return sec_idx

    async def initialize(self) -> None:
        self._autodetect_vehicle_params() 

        wps = self.maneuverable_waypoints
        center_raw_3d = np.array([w.location[:3] for w in wps])
        center_raw = center_raw_3d[:, :2]

        def _w(wp):
            lw = getattr(wp, "lane_width", None)
            if lw is None or not np.isfinite(lw) or lw <= 0:
                lw = self.WIDTH_DEFAULT
            return float(np.clip(lw, self.WIDTH_MIN, self.WIDTH_MAX))
        width_raw = np.array([_w(w) for w in wps])

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

        kclamp = float(self.KAPPA_FF_MAX)
        ds_o = float(np.mean(seg_o))
        ones_o = np.ones(len(center_o))

        def _line_curvature(alpha):
            p = center_o + _circ_smooth(alpha, win_o)[:, None] * normal_o
            t, _ = _tangents_normals(p, smooth_win=win_o)
            return _signed_curvature(t, _seg_lengths(p), smooth_win=win_o)

        # ---------------------------------------------------------------------
        # ALLOCATE SEGMENT PARAMETERS TO OPTIMIZATION GRID
        # ---------------------------------------------------------------------
        u_o = _progress(center_o)
        section_of_o = self._map_segments(u_o)
        
        A_LAT_o = np.array([self.TRACK_SEGMENTS[s]["A_LAT"] for s in section_of_o])
        A_ACCEL_o = np.array([self.TRACK_SEGMENTS[s]["A_ACCEL"] for s in section_of_o])
        A_BRAKE_o = np.array([self.TRACK_SEGMENTS[s]["A_BRAKE"] for s in section_of_o])
        V_MAX_o = np.array([self.TRACK_SEGMENTS[s]["V_MAX"] for s in section_of_o])
        TRACK_MARGIN_MIN_o = np.array([self.TRACK_SEGMENTS[s]["TRACK_MARGIN_MIN"] for s in section_of_o])
        LQR_DRIFT_COEFF_o = np.array([self.TRACK_SEGMENTS[s]["LQR_DRIFT_COEFF"] for s in section_of_o])

        alpha_o = np.zeros(len(center_o))
        best_alpha = alpha_o.copy()
        best_lap = float('inf')

        for pass_idx in range(int(self.N_MARGIN_ITERS)):
            kappa_line = _line_curvature(alpha_o)
            current_path = center_o + alpha_o[:, None] * normal_o
            
            v_o, sgl, _ = _velocity_profile(
                current_path, A_LAT_o * self.GRIP_MARGIN, A_ACCEL_o, 
                A_BRAKE_o, V_MAX_o, self.K_DF, win_o
            )
            
            a_lat = (v_o ** 2) * np.abs(kappa_line)
            drift_margin = LQR_DRIFT_COEFF_o * a_lat
            
            offtrack = np.minimum(0.5 * (self.WHEELBASE ** 2) * np.abs(kappa_line), self.OFFTRACK_CAP)
            tmarg = np.clip(TRACK_MARGIN_MIN_o + drift_margin, TRACK_MARGIN_MIN_o, self.TRACK_MARGIN_MAX)
            usable_o = np.maximum((track_half_o + self.EXTRA_TRACK_WIDTH_M) - (self.CAR_HALF_WIDTH + tmarg + offtrack), self.MIN_CORRIDOR)

            w = self.MINTIME_FLOOR + (v_o / max(np.max(v_o), 1.0)) ** float(self.MINTIME_POWER)
            alpha_o = _fista_raceline(kappa_center_o, usable_o, ds_o, w, iters=self.FISTA_ITERS, w_smooth=self.LINE_SMOOTH)
            
            lap_time = float(np.sum(sgl / np.maximum(v_o, 1e-3)))
            if lap_time < best_lap:
                best_lap = lap_time
                best_alpha = alpha_o.copy()

        alpha_o = best_alpha
        kappa_line_o = _line_curvature(alpha_o)

        u_f = _progress(center_f)
        self.section_of = self._map_segments(u_f) # Extracted for Infrastructure drawing
        
        # ---------------------------------------------------------------------
        # ALLOCATE SEGMENT PARAMETERS TO FINE CONTROL GRID
        # ---------------------------------------------------------------------
        self.A_LAT_f = np.array([self.TRACK_SEGMENTS[s]["A_LAT"] for s in self.section_of])
        self.A_ACCEL_f = np.array([self.TRACK_SEGMENTS[s]["A_ACCEL"] for s in self.section_of])
        self.A_BRAKE_f = np.array([self.TRACK_SEGMENTS[s]["A_BRAKE"] for s in self.section_of])
        self.V_MAX_f = np.array([self.TRACK_SEGMENTS[s]["V_MAX"] for s in self.section_of])
        TRACK_MARGIN_MIN_f = np.array([self.TRACK_SEGMENTS[s]["TRACK_MARGIN_MIN"] for s in self.section_of])
        LQR_DRIFT_COEFF_f = np.array([self.TRACK_SEGMENTS[s]["LQR_DRIFT_COEFF"] for s in self.section_of])

        off_f = np.interp(u_f, np.append(u_o, 1.0), np.append(alpha_o, alpha_o[0]))

        kappa_line_f = np.interp(u_f, np.append(u_o, 1.0), np.append(kappa_line_o, kappa_line_o[0]))
        kappa_line_f = _circ_smooth(np.clip(kappa_line_f, -kclamp, kclamp), win_f)
        tangent_f, normal_f = _tangents_normals(center_f, smooth_win=win_f)
        track_half_f = width_f / 2.0
        
        v_f_approx = np.interp(u_f, np.append(u_o, 1.0), np.append(v_o, v_o[0]))
        a_lat_f = (v_f_approx ** 2) * np.abs(kappa_line_f)
        drift_margin_f = LQR_DRIFT_COEFF_f * a_lat_f
        
        offtrack_f = np.minimum(0.5 * (self.WHEELBASE ** 2) * np.abs(kappa_line_f), self.OFFTRACK_CAP)
        tmarg_f = np.clip(TRACK_MARGIN_MIN_f + drift_margin_f, TRACK_MARGIN_MIN_f, self.TRACK_MARGIN_MAX)
        half_f = np.maximum(track_half_f + self.EXTRA_TRACK_WIDTH_M - (self.CAR_HALF_WIDTH + tmarg_f + offtrack_f), self.MIN_CORRIDOR)

        off_f = np.clip(off_f, -half_f, half_f)
        off_f = _circ_smooth(off_f, max(1, int(round(1.0 / self.DS_TRACK))))
        off_f = np.clip(off_f, -half_f, half_f)
        path_opt = center_f + off_f[:, None] * normal_f

        tangent = _tangents_normals(path_opt, smooth_win=win_f)[0]
        seg = _seg_lengths(path_opt)
        self.path = path_opt
        self.tangent = tangent
        self.path_3d = np.column_stack([path_opt, center_f_3d[:, 2]])

        win_ff = max(1, int(round(self.FF_SMOOTH_M / self.DS_TRACK)))
        self.kappa_signed = _circ_smooth(
            _signed_curvature(tangent, seg, smooth_win=0), win_ff)
        self.kappa_signed = np.clip(self.kappa_signed, -kclamp, kclamp)

        safe_a_lat = self.A_LAT_f * self.GRIP_MARGIN
        v_opt, s_opt, kappa_mag = _velocity_profile(
            path_opt, safe_a_lat, self.A_ACCEL_f, self.A_BRAKE_f,
            self.V_MAX_f, self.K_DF, curv_win_f
        )
        self.v_profile = v_opt
        self.curvature = kappa_mag
        self.seg = s_opt

        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))

        self._total_time = 0.0
        self._lap_time = 0.0
        self._lap_count = 0
        self._started = False
        self._start_idx = self.idx
        self._steer_lp = 0.0

    async def step(self) -> None:
        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        speed = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        n = len(self.path)

        sb = int(getattr(self, "_SB", 15))
        sf = int(getattr(self, "_SF", 250))
        win_ix = [(self.idx + k - sb) % n for k in range(sb + sf)]
        win = self.path[win_ix]
        self.idx = win_ix[int(np.argmin(np.linalg.norm(win - loc, axis=1)))]
        i = self.idx

        p = self.path[i]
        t = self.tangent[i]
        path_yaw = np.arctan2(t[1], t[0])
        dx = loc[0] - p[0]
        dy = loc[1] - p[1]
        e_y = -dx * t[1] + dy * t[0]
        e_psi = normalize_rad(yaw - path_yaw)

        # ---------------------------------------------------------------------
        # SEGMENT TIMING AND LOGIC
        # ---------------------------------------------------------------------
        current_sec = self.section_of[i]
        sec_cfg = self.TRACK_SEGMENTS[current_sec]
        
        if getattr(self, "_current_sec", None) is None:
            self._current_sec = current_sec
            self._sec_start_time = self._total_time
            
        if current_sec != self._current_sec:
            sec_time = self._total_time - self._sec_start_time
            if self._started: 
                print(f"[LAP {self._lap_count}] {self.TRACK_SEGMENTS[self._current_sec]['name']} Time: {sec_time:.3f}s")
            self._current_sec = current_sec
            self._sec_start_time = self._total_time

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
        
        # Look up Gain Scheduled LQR parameters
        K = self._lqr_gain(speed, sec_cfg["Q_E"], sec_cfg["Q_PSI"])
        
        delta_fb = float((-(K @ x_err))[0])
        delta = delta_ff + delta_fb

        steer_cmd = float(np.clip(-delta / self.MAX_STEER_RAD, -1.0, 1.0))
        beta = float(self.STEER_LP_BETA)
        self._steer_lp += beta * (steer_cmd - self._steer_lp)
        rate = float(self.STEER_RATE)
        steer = float(np.clip(self._steer_lp, self._prev_steer - rate, self._prev_steer + rate))
        self._prev_steer = steer

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
        g_lat = self.GRIP_MARGIN * (self.A_LAT_f[i] + self.K_DF * speed ** 2)
        lat_ratio = np.clip(a_lat_now / max(g_lat, 1e-3), 0.0, 1.0)
        max_throttle = float(np.sqrt(max(0.0, 1.0 - lat_ratio ** 2)))

        dv = v_target - speed
        self.integral_error = float(np.clip(self.integral_error + dv * self.CONTROL_DT, -4.0, 4.0))
        if dv >= 0.0:
            ff = dv / self.A_ACCEL_f[i]
            throttle = float(np.clip(ff + self.KP_SPEED * dv + self.KI_SPEED * self.integral_error,
                                     0.0, max_throttle))
            brake = 0.0
            if dv > 1.5:
                self.integral_error = 0.0
        else:
            throttle = 0.0
            ff = -dv / self.A_BRAKE_f[i]
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

        self._total_time += self.CONTROL_DT
        self._lap_time += self.CONTROL_DT
        
        if self._started and abs(i - self._start_idx) < 5 and self._lap_time > 30.0:
            self._lap_count += 1
            print(f">>> [LAP {self._lap_count} COMPLETE] Total Lap Time: {self._lap_time:6.2f}s <<<")
            self._lap_time = 0.0
        elif not self._started and self._lap_time > 2.0:
            self._started = True

        if self.TELEMETRY:
            self.telemetry = {
                "v": speed, "v_target": float(v_target),
                "e_y": float(e_y), "e_psi": float(e_psi),
                "steer": steer, "lat_g": float(a_lat_now / 9.81),
                "section": int(current_sec) + 1  # Feeds to the UI Dashboard!
            }
        return control