from roar_py_interface import RoarPyActor, RoarPySensor
import typing
import gymnasium as gym
import roar_py_interface
import pygame
from PIL.Image import Image
import numpy as np
from collections import deque
from typing import Optional, Dict, Any

class ManualControlViewer:
    # ----- Dashboard styling -------------------------------------------------
    DASH_W      = 380          # width of the side panel (px)
    BG          = (18, 20, 26)
    FG          = (235, 238, 245)
    MUTED       = (140, 148, 165)
    TRACK       = (52, 56, 70)
    GREEN       = (74, 222, 128)
    RED         = (248, 113, 113)
    BLUE        = (96, 165, 250)
    AMBER       = (251, 191, 36)

    MAP_H       = 232          # height of the track-map region at the bottom

    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
        self.font = None
        self.font_big = None
        self.font_small = None
        self.font_label = None
        self.last_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }
        # ----- track-map state ----------------------------------------------
        self._traj_xy = None
        self._traj_speed = None
        self._map_built = False
        self._map_box = None
        self._breadcrumb = deque(maxlen=260)

    def init_pygame(self, x, y) -> None:
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.font_big   = pygame.font.SysFont(mono, 34, bold=True)
        self.font       = pygame.font.SysFont(mono, 18)
        self.font_small = pygame.font.SysFont(mono, 14)
        self.font_label = pygame.font.SysFont(mono, 14, bold=True)

    def close(self) -> None:
        pygame.quit()

    # ---------------------------------------------------------- trajectory API
    def set_trajectory(self, path_xy, speeds) -> None:
        """Call once after the solution plans its path. path_xy is (N,2) world XY
        in the same frame the location sensor reports; speeds is (N,)."""
        self._traj_xy = np.asarray(path_xy, dtype=float)[:, :2]
        self._traj_speed = np.asarray(speeds, dtype=float)
        self._map_built = False

    @staticmethod
    def _speed_color(frac):
        frac = float(max(0.0, min(1.0, frac)))
        if frac < 0.5:
            t = frac / 0.5
            return (220, int(60 + 140 * t), 60)            # red -> yellow
        t = (frac - 0.5) / 0.5
        return (int(220 - 140 * t), int(200 + 20 * t), int(60 + 60 * t))  # yellow -> green

    def _w2m(self, x, y):
        return (int(self._map_bcx + (x - self._map_cxw) * self._map_s),
                int(self._map_bcy - (y - self._map_cyw) * self._map_s))

    def _build_map(self, x0):
        px = 20
        cw = self.DASH_W - 2 * px
        H = self.screen.get_height()
        self._map_box = (x0 + px, H - self.MAP_H - 16, cw, self.MAP_H)
        bx, by, bw, bh = self._map_box

        xy = self._traj_xy
        minx, miny = xy.min(0)
        maxx, maxy = xy.max(0)
        self._map_cxw = (minx + maxx) / 2.0
        self._map_cyw = (miny + maxy) / 2.0
        spanx = max(maxx - minx, 1e-3)
        spany = max(maxy - miny, 1e-3)
        margin = 12
        self._map_s = min((bw - 2 * margin) / spanx, (bh - 2 * margin) / spany)
        self._map_bcx = bx + bw / 2.0
        self._map_bcy = by + bh / 2.0

        n = len(xy)
        stepm = max(1, n // 500)
        idxs = list(range(0, n, stepm))
        sp = self._traj_speed
        vmin, vmax = float(sp.min()), float(sp.max())
        rng = max(vmax - vmin, 1e-3)
        self._map_px = [self._w2m(xy[i, 0], xy[i, 1]) for i in idxs]
        self._map_col = [self._speed_color((sp[i] - vmin) / rng) for i in idxs]
        self._map_built = True

    def _draw_car_marker(self, cxy, cyaw):
        cx, cy = self._w2m(cxy[0], cxy[1])
        dirx, diry = np.cos(cyaw), -np.sin(cyaw)   # y inverted in map space
        L, Wd, back = 10.0, 5.0, 4.0
        tip   = (cx + dirx * L, cy + diry * L)
        pxp, pyp = -diry, dirx
        left  = (cx - dirx * back + pxp * Wd, cy - diry * back + pyp * Wd)
        right = (cx - dirx * back - pxp * Wd, cy - diry * back - pyp * Wd)
        pygame.draw.polygon(self.screen, (80, 200, 255), [tip, left, right])

    def _draw_map(self, x0, t):
        if self._traj_xy is None:
            return
        if not self._map_built:
            self._build_map(x0)
        bx, by, bw, bh = self._map_box
        dist = t.get("dist_m")
        title = "TRACK MAP  (line · car · driven)"
        if dist is not None:
            title = f"TRACK MAP   @ {int(round(float(dist)))} m"
        self._text(title, x0 + 20, by - 18, self.MUTED, self.font_small)
        pygame.draw.rect(self.screen, (12, 13, 18), (bx, by, bw, bh), border_radius=6)
        pygame.draw.rect(self.screen, self.TRACK, (bx, by, bw, bh), 1, border_radius=6)

        pts, col = self._map_px, self._map_col
        for k in range(len(pts) - 1):
            pygame.draw.line(self.screen, col[k], pts[k], pts[k + 1], 2)
        pygame.draw.line(self.screen, col[-1], pts[-1], pts[0], 2)   # close loop

        if len(self._breadcrumb) > 1:
            bc = [self._w2m(p[0], p[1]) for p in self._breadcrumb]
            pygame.draw.lines(self.screen, (245, 245, 245), False, bc, 1)

        cxy = t.get("car_xy")
        if cxy is not None:
            try:
                self._draw_car_marker(cxy, float(t.get("car_yaw", 0.0)))
            except Exception:
                pass

        self._text("slow", bx + 6, by + bh - 16, (220, 90, 90), self.font_small)
        self._text("fast", bx + bw - 6, by + bh - 16, (90, 210, 130), self.font_small, right=True)

    # ---------------------------------------------------------- text/bar helpers
    def _text(self, s, x, y, color, font=None, right=False):
        font = font or self.font
        surf = font.render(str(s), True, color)
        rect = surf.get_rect()
        if right:
            rect.topright = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)
        return rect

    def _hbar(self, x, y, w, h, frac, color):
        frac = float(max(0.0, min(1.0, frac)))
        pygame.draw.rect(self.screen, self.TRACK, (x, y, w, h), border_radius=4)
        fw = int(frac * w)
        if fw > 0:
            pygame.draw.rect(self.screen, color, (x, y, fw, h), border_radius=4)

    def _center_bar(self, x, y, w, h, val, color):
        pygame.draw.rect(self.screen, self.TRACK, (x, y, w, h), border_radius=4)
        cx = x + w // 2
        v = float(max(-1.0, min(1.0, val)))
        if v >= 0:
            pygame.draw.rect(self.screen, color, (cx, y, int(v * w / 2), h))
        else:
            pygame.draw.rect(self.screen, color, (cx + int(v * w / 2), y, -int(v * w / 2), h))
        pygame.draw.line(self.screen, self.FG, (cx, y - 2), (cx, y + h + 2), 1)

    def _labeled_bar(self, label, val, color, x, y, cw):
        val = float(val or 0.0)
        self._text(label, x, y, self.MUTED, self.font_small)
        self._text(f"{val:4.2f}", x + cw, y, self.FG, self.font, right=True)
        y += 20
        self._hbar(x, y, cw, 16, val, color)
        return y + 30

    def _draw_dashboard(self, x0, t):
        px = 20
        W = self.DASH_W
        cw = W - 2 * px
        x = x0 + px
        x_right = x0 + W - px

        pygame.draw.rect(self.screen, self.BG, (x0, 0, W, self.screen.get_height()))
        pygame.draw.line(self.screen, self.TRACK, (x0, 0), (x0, self.screen.get_height()), 2)

        y = 18
        self._text("LIVE TELEMETRY", x, y, self.FG, self.font_label)
        if t.get("elapsed") is not None:
            self._text(f"{t['elapsed']:6.1f}s", x_right, y, self.MUTED, self.font, right=True)
        y += 32

        speed = float(t.get("speed", 0.0) or 0.0)
        self._text("SPEED", x, y, self.MUTED, self.font_small)
        self._text(f"{speed * 3.6:5.0f} km/h", x_right, y, self.MUTED, self.font_small, right=True)
        y += 18
        self._text(f"{speed:5.1f}", x, y, self.GREEN, self.font_big)
        self._text("m/s", x + 110, y + 14, self.MUTED, self.font_small)
        if t.get("target_speed") is not None:
            self._text(f"target {float(t['target_speed']):5.1f}", x_right, y + 12,
                       self.BLUE, self.font, right=True)
        y += 50

        y = self._labeled_bar("THROTTLE", t.get("throttle", 0.0), self.GREEN, x, y, cw)
        y = self._labeled_bar("BRAKE", t.get("brake", 0.0), self.RED, x, y, cw)

        steer = float(t.get("steer", 0.0) or 0.0)
        side = "LEFT" if steer < -1e-3 else ("RIGHT" if steer > 1e-3 else "—")
        self._text("STEERING", x, y, self.MUTED, self.font_small)
        self._text(f"{steer:+.2f} {side}", x_right, y, self.FG, self.font, right=True)
        y += 20
        self._center_bar(x, y, cw, 16, steer, self.AMBER)
        self._text("L", x, y + 18, self.MUTED, self.font_small)
        self._text("R", x_right, y + 18, self.MUTED, self.font_small, right=True)
        y += 46

        pygame.draw.line(self.screen, self.TRACK, (x, y), (x_right, y), 1)
        y += 14

        cp = t.get("checkpoint")
        total = t.get("total_waypoints")
        if cp is not None and total:
            self._text("CHECKPOINT", x, y, self.MUTED, self.font_small)
            self._text(f"{cp} / {total}", x_right, y, self.FG, self.font, right=True)
            y += 20
            self._hbar(x, y, cw, 14, cp / max(1, total), self.BLUE)
            self._text(f"{100 * cp / max(1, total):4.1f}%", x_right, y + 16,
                       self.MUTED, self.font_small, right=True)
            y += 36

        if t.get("lap") is not None and t.get("total_laps"):
            self._text("LAP", x, y, self.MUTED, self.font_small)
            self._text(f"{t['lap']} / {t['total_laps']}", x_right, y, self.FG, self.font, right=True)
            y += 26

        if t.get("lat_g") is not None:
            self._text("LATERAL", x, y, self.MUTED, self.font_small)
            self._text(f"{float(t['lat_g']) / 9.81:4.2f} g", x_right, y, self.AMBER, self.font, right=True)
            y += 26

        if float(t.get("collision", 0.0) or 0.0) > 100.0:
            pygame.draw.rect(self.screen, self.RED, (x, y, cw, 22), border_radius=6)
            self._text("COLLISION / RESPAWN", x + 10, y + 3, (20, 20, 20), self.font_label)
            y += 30

        # Track map pinned to the bottom of the panel.
        self._draw_map(x0, t)

    # ----------------------------------------------------------------- render
    def render(
        self,
        image: roar_py_interface.RoarPyCameraSensorData,
        occupancy_map: Optional[Image] = None,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        image_pil: Image = image.get_image()
        occupancy_map_rgb = occupancy_map.convert("RGB") if occupancy_map is not None else None

        if self.screen is None:
            base_w = image_pil.width + (occupancy_map.width if occupancy_map_rgb is not None else 0)
            self.init_pygame(base_w + self.DASH_W, image_pil.height)

        new_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            new_control['throttle'] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 0.2

        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        if occupancy_map_rgb is not None:
            occupancy_map_surface = pygame.image.fromstring(occupancy_map_rgb.tobytes(), occupancy_map_rgb.size, occupancy_map_rgb.mode).convert()

        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        dash_x = image_pil.width
        if occupancy_map_rgb is not None:
            self.screen.blit(occupancy_map_surface, (image_pil.width, 0))
            dash_x = image_pil.width + occupancy_map.width

        # Record breadcrumb of where the car actually drove (map frame world XY).
        if telemetry is not None:
            cxy = telemetry.get("car_xy")
            if cxy is not None:
                if (not self._breadcrumb or
                        abs(cxy[0] - self._breadcrumb[-1][0]) +
                        abs(cxy[1] - self._breadcrumb[-1][1]) > 0.5):
                    self._breadcrumb.append((float(cxy[0]), float(cxy[1])))

        self._draw_dashboard(dash_x, telemetry or {})

        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control

class RoarCompetitionAgentWrapper(RoarPyActor):
    def __init__(self, wrapped : RoarPyActor):
        self._wrapped = wrapped
    
    @property
    def control_timestep(self) -> float:
        return self._wrapped.control_timestep
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped.force_real_control_timestep

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._wrapped.get_sensors()

    def get_action_spec(self) -> gym.Space:
        return self._wrapped.get_action_spec()
    
    async def _apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped._apply_action(action)

    def close(self):
        pass

    def is_closed(self) -> bool:
        return self._wrapped.is_closed()

    def __del__(self):
        pass
    
    async def apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped.apply_action(action)

    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped.get_gym_observation_spec()

    async def receive_observation(self) -> typing.Dict[str, typing.Any]:
        return await self._wrapped.receive_observation()
    
    def get_last_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_observation()
    
    def get_last_gym_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_gym_observation()

    def convert_obs_to_gym_obs(self, observation : typing.Dict[str,typing.Any]) -> typing.Dict[str,typing.Any]:
        return self._wrapped.convert_obs_to_gym_obs(observation)