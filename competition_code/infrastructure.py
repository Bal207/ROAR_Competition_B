from roar_py_interface import RoarPyActor, RoarPySensor
import typing
import gymnasium as gym
import roar_py_interface
import pygame
from PIL.Image import Image
import numpy as np
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

    def init_pygame(self, x, y) -> None:
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        # Monospace keeps the numbers from jittering as they change width.
        mono = "consolas,menlo,dejavusansmono,monospace"
        self.font_big   = pygame.font.SysFont(mono, 34, bold=True)
        self.font       = pygame.font.SysFont(mono, 18)
        self.font_small = pygame.font.SysFont(mono, 14)
        self.font_label = pygame.font.SysFont(mono, 14, bold=True)

    def close(self) -> None:
        pygame.quit()

    # ---------------------------------------------------------------- helpers
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
        """val in [-1, 1]; 0 is the centre tick. Fills toward the sign."""
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
        H = self.screen.get_height()

        pygame.draw.rect(self.screen, self.BG, (x0, 0, W, H))
        pygame.draw.line(self.screen, self.TRACK, (x0, 0), (x0, H), 2)

        y = 18
        self._text("LIVE TELEMETRY", x, y, self.FG, self.font_label)
        if t.get("elapsed") is not None:
            self._text(f"{t['elapsed']:6.1f}s", x_right, y, self.MUTED, self.font, right=True)
        y += 32

        # --- Speed -----------------------------------------------------------
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

        # --- Pedals ----------------------------------------------------------
        y = self._labeled_bar("THROTTLE", t.get("throttle", 0.0), self.GREEN, x, y, cw)
        y = self._labeled_bar("BRAKE", t.get("brake", 0.0), self.RED, x, y, cw)

        # --- Steering (centred) ---------------------------------------------
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

        # --- Checkpoint / progress ------------------------------------------
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

        # --- Lap -------------------------------------------------------------
        if t.get("lap") is not None and t.get("total_laps"):
            self._text("LAP", x, y, self.MUTED, self.font_small)
            self._text(f"{t['lap']} / {t['total_laps']}", x_right, y, self.FG, self.font, right=True)
            y += 26

        # --- Lateral g (optional, from solution) -----------------------------
        if t.get("lat_g") is not None:
            self._text("LATERAL", x, y, self.MUTED, self.font_small)
            self._text(f"{float(t['lat_g']) / 9.81:4.2f} g", x_right, y, self.AMBER, self.font, right=True)
            y += 26

        # --- Collision flash -------------------------------------------------
        if float(t.get("collision", 0.0) or 0.0) > 100.0:
            pygame.draw.rect(self.screen, self.RED, (x, y, cw, 26), border_radius=6)
            self._text("COLLISION / RESPAWN", x + 10, y + 5, (20, 20, 20), self.font_label)
            y += 34

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

        # Side dashboard – sits to the right of the feed, never overlapping it.
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