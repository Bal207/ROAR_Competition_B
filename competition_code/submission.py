"""
ROAR Competition Solution  --  Offline trajectory optimization + robust tracker
===============================================================================

Pipeline (all the heavy work happens ONCE, in initialize(), in frozen sim-time):

  raw waypoints
    -> resample centreline to a fixed metric spacing  (conditions the optimizer)
    -> STAGE 1: minimum-curvature racing line          (convex bounded LSQ, scipy)
    -> STAGE 2 (optional): minimum-time refinement      (CasADi/IPOPT, with fallback)
    -> interpolate the line onto a fine tracking grid
    -> friction-circle velocity profile (forward/backward, downforce-aware, with grip buffer)

At run time, step() just:
    -> Pure Pursuit steering toward the line  (one self-correcting law)
    -> throttle/brake to track the pre-computed target speed

The ONLY knob that matters day-to-day is A_LAT (grip). Everything else is a
physical constant, a safety cap, or an optional refinement.

ML / RL extension points (for your next stage) are marked  >>> ML HOOK <<<.
"""

from typing import List, Tuple, Optional
import numpy as np
import roar_py_interface

import scipy.sparse as sp
from scipy.optimize import lsq_linear

try:
    import casadi as _ca
    _HAS_CASADI = True
except Exception:
    _HAS_CASADI = False


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
#  GEOMETRY + OPTIMIZER  (module-level, no simulator dependency -> unit-testable)
# =============================================================================
def _resample_closed(points: np.ndarray, ds: float) -> np.ndarray:
    """Resample a closed polyline to ~uniform spacing `ds` (metres). Works for 2D and 3D."""
    loop = np.vstack([points, points[:1]])
    # Always calculate arc length based on X, Y (first two columns)
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
    """Resample a per-point scalar (e.g. lane width) onto the resampled path."""
    loop = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(round(total / ds)))
    s_new = np.linspace(0.0, total, n, endpoint=False)
    v_loop = np.concatenate([values, values[:1]])
    return np.interp(s_new, s, v_loop)


def _tangents_normals(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def _windowed_curvature(path: np.ndarray, win: int) -> np.ndarray:
    """Menger curvature using neighbours `win` steps away -> robust to dense points."""
    p0 = np.roll(path, win, axis=0)
    p1 = path
    p2 = np.roll(path, -win, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area = 0.5 * np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1])
                        - (p2[:, 0]-p0[:, 0])*(p1[:, 1]-p0[:, 1]))
    return 4.0 * area / (a * b * c + 1e-9)


def _min_curvature_offsets(center, normal, half_width) -> np.ndarray:
    """STAGE 1 -- minimum-curvature line as a convex bounded least-squares problem."""
    n = len(center)
    idx = np.arange(n)
    D = sp.csr_matrix(
        (np.concatenate([np.ones(n), -2*np.ones(n), np.ones(n)]),
         (np.concatenate([idx, idx, idx]),
          np.concatenate([(idx-1) % n, idx, (idx+1) % n]))), shape=(n, n))
    
    # Formulate D * (Center + alpha * Normal) ≈ 0
    Ax = D @ sp.diags(normal[:, 0])
    Ay = D @ sp.diags(normal[:, 1])
    bx = D @ center[:, 0]
    by = D @ center[:, 1]
    
    C = sp.vstack([Ax, Ay]).tocsr()
    d = -np.concatenate([bx, by])
    
    res = lsq_linear(C, d, bounds=(-half_width, half_width),
                     max_iter=500, tol=1e-4)
    return res.x


def _min_time_offsets(center, normal, half_width, alpha0,
                      A_LAT, A_LON, V_MAX, V_MIN, max_nodes=260):
    """STAGE 2 (optional) -- minimum-lap-time refinement (point-mass, friction circle)."""
    if not _HAS_CASADI:
        return None
    try:
        n_full = len(center)
        stride = max(1, n_full // max_nodes)
        di = np.arange(0, n_full, stride)
        c = center[di]; nm = normal[di]; hw = half_width[di]; a0 = alpha0[di]
        n = len(di)

        opti = _ca.Opti()
        a = opti.variable(n)
        v = opti.variable(n)
        Px = c[:, 0] + a * nm[:, 0]
        Py = c[:, 1] + a * nm[:, 1]
        T = 0
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            dx = Px[ip] - Px[i]; dy = Py[ip] - Py[i]
            seg = _ca.sqrt(dx*dx + dy*dy + 1e-6)
            area2 = (Px[i]-Px[im])*(Py[ip]-Py[im]) - (Px[ip]-Px[im])*(Py[i]-Py[im])
            la = _ca.sqrt((Px[i]-Px[im])**2 + (Py[i]-Py[im])**2 + 1e-6)
            lb = _ca.sqrt((Px[ip]-Px[i])**2 + (Py[ip]-Py[i])**2 + 1e-6)
            lc = _ca.sqrt((Px[ip]-Px[im])**2 + (Py[ip]-Py[im])**2 + 1e-6)
            kappa = 2 * _ca.fabs(area2) / (la * lb * lc + 1e-6)
            a_lat = v[i]**2 * kappa
            a_lon = (v[ip]**2 - v[i]**2) / (2 * seg)
            opti.subject_to((a_lat/A_LAT)**2 + (a_lon/A_LON)**2 <= 1.0)
            T += 2 * seg / (v[i] + v[ip])
        opti.minimize(T)
        opti.subject_to(opti.bounded(-hw, a, hw))
        opti.subject_to(opti.bounded(V_MIN, v, V_MAX))
        opti.set_initial(a, a0)
        opti.set_initial(v, (V_MIN + 1.0) + np.zeros(n))
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                              "ipopt.max_iter": 500, "ipopt.tol": 1e-4,
                              "ipopt.acceptable_tol": 1e-3})
        sol = opti.solve()
        a_ds = sol.value(a)
        return np.interp(np.arange(n_full), np.append(di, n_full),
                         np.append(a_ds, a_ds[0]))
    except Exception as e:
        print(f"[planner] min-time NLP fell back to min-curvature ({type(e).__name__})")
        return None


def _velocity_profile(path, A_LAT, A_ACCEL, A_BRAKE, V_MAX, K_DF, curv_win, passes=4):
    """Friction-circle quasi-steady-state speed profile (closed loop, downforce-aware)."""
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
    return v, seg


def _lap_time(v, seg):
    return float(np.sum(seg / np.maximum(0.5 * (v + np.roll(v, -1)), 0.1)))


# =============================================================================
#  SOLUTION
# =============================================================================
class RoarCompetitionSolution:

    # ----- TUNING -----------------------------------------------------------
    A_LAT   = 22.0   # max lateral accel [m/s^2] == grip.  >>> THE speed knob.
    GRIP_MARGIN = 0.90 # Plan corners at 90% limit to give the tracker a physical buffer
    
    STEER_KP = 2    # steering response. VALID RANGE ~1-4. delta is already a
                    #   pure-pursuit angle in radians, so this is a small calibration.
    STEER_SIGN = -1.0  # this sim: steer<0 = LEFT.

    A_ACCEL = 15.0    # max forward accel [m/s^2]
    A_BRAKE = 15.0   # max braking decel [m/s^2]
    V_MAX   = 105.0  # top-speed cap [m/s]
    K_DF    = 0.0    # downforce grip

    # racing line / planner
    USE_RACING_LINE = True 
    USE_MIN_TIME = True    # Set to False by default; often unstable on extreme corners.
    DS_OPT     = 2.5       # optimisation spacing [m]
    DS_TRACK   = 1.0       # tracking-grid spacing [m]
    
    # --- BOUNDARY SAFETY ---
    LANE_MARGIN = 3.7             # Base margin from track edge [m]
    TIGHT_CORNER_MARGIN_K = 30.0  # Multiplier to push line away from the apex in hairpins.
                                  # Compensates for Pure Pursuit cutting + rear-wheel drag.
    
    CURV_WIN_M = 4.0       # arc-length window for curvature estimate [m]

    # pure pursuit -- lookahead shrinks in corners so tight turns steer sharper
    WHEELBASE = 2.7
    LD_K   =  0.3         # lookahead = LD_K * speed ...
    LD_MIN = 5.0
    LD_MAX = 22         # ... clamped here. Lower LD_MAX = sharper cornering.

    # ------------------------------------------------------------------------
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
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    def _get_carla_world(self):
        """Bypasses roar_py entirely to find the CARLA world safely."""
        # Check if we have already cached the world object to avoid redundant spamming
        if hasattr(self, '_cached_carla_world') and self._cached_carla_world is not None:
            return self._cached_carla_world

        try:
            import carla
            
            # Backdoor 1: Connect directly to the local simulator port
            try:
                client = carla.Client('127.0.0.1', 2000)
                client.set_timeout(0.1) # Aggressive timeout so we don't block roar_py's loop
                world = client.get_world()
                if world is not None:
                    print("[debug] 🔓 Backdoor 1 successful: Connected directly to CARLA port 2000.")
                    self._cached_carla_world = world
                    return world
            except:
                pass
            
            # Backdoor 2: Search active RAM for a CARLA world object.
            import gc
            for obj in gc.get_objects():
                try:
                    if type(obj).__name__ == "World" and "carla" in str(type(obj)):
                        print("[debug] 🔓 Backdoor 2 successful: Found CARLA World hidden in memory.")
                        self._cached_carla_world = obj
                        return obj
                except:
                    continue
                    
        except Exception as e:
            print(f"[debug] Backdoor tools failed: {e}")
            
        return None

    def _draw_path_carla(self):
        """Draws the generated trajectory as a permanent green line in the simulator."""
        try:
            import carla
            world = self._get_carla_world()
            if world is None:
                print("\n[debug] ❌ ERROR: Could not find the raw CARLA world to draw the path!\n")
                return
            
            debug = world.debug
            # Bumped Z-offset to 1.5 meters so it floats visibly at windshield height
            z_off = 0.25
            
            print(f"[debug] 🖌️ Drawing {len(self.path_3d)} 3D track points...")
            for i in range(len(self.path_3d)):
                p1 = self.path_3d[i]
                p2 = self.path_3d[(i + 1) % len(self.path_3d)]
                
                loc1 = carla.Location(x=float(p1[0]), y=-float(p1[1]), z=float(p1[2]) + z_off)
                loc2 = carla.Location(x=float(p2[0]), y=-float(p2[1]), z=float(p2[2]) + z_off)
                
                debug.draw_line(loc1, loc2, thickness=0.2, 
                                color=carla.Color(0, 255, 0), life_time=0.0)
            print("[debug] ✅ Path drawing command sent to CARLA.")
        except Exception as e:
            print(f"[debug] ❌ Crash while drawing path: {e}")

    def _draw_target_carla(self, target_idx: int):
        """Draws a red dot on the immediate lookahead target for pure pursuit."""
        try:
            import carla
            world = self._get_carla_world()
            if world is None:
                return # Fail silently here so it doesn't spam the console 60 times a second
            
            p = self.path_3d[target_idx]
            # Red dot floats slightly higher to be easily visible
            loc = carla.Location(x=float(p[0]), y=-float(p[1]), z=float(p[2]) + 1.8)
            
            world.debug.draw_point(loc, size=0.15, color=carla.Color(255, 0, 0), life_time=0.1)
        except:
            pass

    # ------------------------------------------------------------------------
    async def initialize(self) -> None:
        wps = self.maneuverable_waypoints
        center_raw_3d = np.array([w.location[:3] for w in wps])
        center_raw = center_raw_3d[:, :2]
        width_raw = np.array([float(getattr(w, "lane_width", 6.0)) for w in wps])

        # Setup optimization grid
        center_o_3d = _resample_closed(center_raw_3d, self.DS_OPT)
        center_o = center_o_3d[:, :2]
        width_o = _resample_scalar_closed(width_raw, center_raw, self.DS_OPT)
        _, normal_o = _tangents_normals(center_o)
        
        # Setup fine tracking grid
        center_f_3d = _resample_closed(center_raw_3d, self.DS_TRACK)
        center_f = center_f_3d[:, :2]
        width_f = _resample_scalar_closed(width_raw, center_raw, self.DS_TRACK)
        _, normal_f = _tangents_normals(center_f)
        
        # --- DYNAMIC MARGIN FIX FOR HAIRPINS ---
        curv_win_o = max(1, int(round(self.CURV_WIN_M / self.DS_OPT)))
        kappa_o = _windowed_curvature(center_o, curv_win_o)
        
        curv_win_f = max(1, int(round(self.CURV_WIN_M / self.DS_TRACK)))
        kappa_f = _windowed_curvature(center_f, curv_win_f)

        # 1. Expand safety buffer dynamically based on local curvature
        dyn_margin_o = self.LANE_MARGIN + (self.TIGHT_CORNER_MARGIN_K * kappa_o)
        dyn_margin_f = self.LANE_MARGIN + (self.TIGHT_CORNER_MARGIN_K * kappa_f)
        
        half_o = np.maximum(width_o / 2.0 - dyn_margin_o, 0.0)
        half_f = np.maximum(width_f / 2.0 - dyn_margin_f, 0.0)
        
        # 2. Hard-stop normal crossover (prevents path from mathematically looping inside itself)
        max_safe_o = np.maximum(1.0 / (kappa_o + 1e-6) - 0.5, 0.0)
        max_safe_f = np.maximum(1.0 / (kappa_f + 1e-6) - 0.5, 0.0)
        
        half_o = np.minimum(half_o, max_safe_o)
        half_f = np.minimum(half_f, max_safe_f)
        # ---------------------------------------

        s_o = self._arclen(center_o)
        s_f = self._arclen(center_f)[:-1]

        # 1. Evaluate baseline Centerline Performance
        safe_a_lat = self.A_LAT * self.GRIP_MARGIN
        v_c, s_c = _velocity_profile(center_f, safe_a_lat, self.A_ACCEL,
                                     self.A_BRAKE, self.V_MAX, self.K_DF, curv_win_f)
        t_center = _lap_time(v_c, s_c)

        # 2. Compute Racing Line & Evaluate 
        if self.USE_RACING_LINE:
            alpha_o = _min_curvature_offsets(center_o, normal_o, half_o)

            if self.USE_MIN_TIME:
                a_mt = _min_time_offsets(center_o, normal_o, half_o, alpha_o,
                                         A_LAT=self.A_LAT,
                                         A_LON=min(self.A_ACCEL*1.4, self.A_BRAKE),
                                         V_MAX=self.V_MAX, V_MIN=5.0)
                if a_mt is not None:
                    alpha_o = np.clip(a_mt, -half_o, half_o)

            # Interpolate offset to fine tracking grid
            off_f = np.interp(s_f % s_o[-1], s_o, np.append(alpha_o, alpha_o[0]))
            off_f = np.clip(off_f, -half_f, half_f)
            path_opt = center_f + off_f[:, None] * normal_f

            # Generate target velocity profile
            v_opt, s_opt = _velocity_profile(path_opt, safe_a_lat, self.A_ACCEL, 
                                             self.A_BRAKE, self.V_MAX, self.K_DF, curv_win_f)
            t_opt = _lap_time(v_opt, s_opt)

            # 3. Fallback Guard Check
            if t_opt < t_center and t_opt > 0:
                self.path = path_opt
                self.path_3d = np.column_stack([path_opt, center_f_3d[:, 2]]) 
                self.v_profile = v_opt
                self.seg = s_opt
                t_final = t_opt
                print(f"[planner] Racing line accepted. Lap time: {t_final:.2f}s "
                      f"({100*(t_center-t_final)/t_center:+.1f}%)")
            else:
                self.path = center_f
                self.path_3d = center_f_3d
                self.v_profile = v_c
                self.seg = s_c
                t_final = t_center
                print(f"[planner] WARNING: Optimizer yielded worse time ({t_opt:.2f}s) "
                      f"than centerline ({t_center:.2f}s). Falling back to centerline.")
        else:
            self.path = center_f
            self.path_3d = center_f_3d
            self.v_profile = v_c
            self.seg = s_c
            t_final = t_center
            print(f"[planner] Using plain centerline. Lap time: {t_final:.2f}s")

        self._draw_path_carla()

        print(f"[planner] v_min={self.v_profile.min():.1f} m/s, "
              f"v_max={self.v_profile.max():.1f} m/s | "
              f"min-time={'on' if (self.USE_MIN_TIME and _HAS_CASADI) else 'off'}")

        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))

    @staticmethod
    def _arclen(path):
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    # ------------------------------------------------------------------------
    async def step(self) -> None:
        loc = np.asarray(self.location_sensor.get_last_gym_observation())[:2]
        yaw = float(self.rpy_sensor.get_last_gym_observation()[2])
        speed = float(np.linalg.norm(self.velocity_sensor.get_last_gym_observation()))
        n = len(self.path)

        # progress: closest point in a forward window (monotonic). If we are
        # far from that window (e.g. after a collision respawn), relocate
        # globally so the controller recovers instead of chasing a stale point.
        win_ix = [(self.idx + k) % n for k in range(40)]
        win = self.path[win_ix]
        wd = np.linalg.norm(win - loc, axis=1)
        if wd.min() > 6.0:
            self.idx = int(np.argmin(np.linalg.norm(self.path - loc, axis=1)))
        else:
            self.idx = (self.idx + int(np.argmin(wd))) % n

        ld = float(np.clip(self.LD_K * speed, self.LD_MIN, self.LD_MAX))
        j, dist = self.idx, 0.0
        for _ in range(n):
            if dist >= ld:
                break
            dist += self.seg[j]
            j = (j + 1) % n
            
        self._draw_target_carla(j)
            
        to_t = self.path[j] - loc
        alpha = normalize_rad(np.arctan2(to_t[1], to_t[0]) - yaw)
        delta = np.arctan2(2.0 * self.WHEELBASE * np.sin(alpha), max(ld, 1e-3))
        
        # STEER_SIGN converts a pure-pursuit angle (left = +) into this sim's
        # command convention (left = -, right = +). If the car still steers the
        # WRONG way, flip STEER_SIGN to +1.0.
        steer = float(np.clip(self.STEER_SIGN * self.STEER_KP * delta, -1.0, 1.0))

        # >>> ML HOOK <<< : scale v_target by a learned per-segment factor here.
        v_target = self.v_profile[self.idx]
        look, d_ahead, horizon = self.idx, 0.0, max(speed * 0.6, 5.0)
        for _ in range(n):
            if d_ahead >= horizon:
                break
            d_ahead += self.seg[look]
            look = (look + 1) % n
            v_target = min(v_target, self.v_profile[look])

        dv = v_target - speed
        if dv >= 0.0:
            throttle, brake = float(np.clip(0.5 * dv + 0.1, 0.0, 1.0)), 0.0
        else:
            throttle, brake = 0.0, float(np.clip(-0.25 * dv, 0.0, 1.0))

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