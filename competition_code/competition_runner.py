import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper, ManualControlViewer
from typing import List, Type, Optional, Dict, Any
import carla
import numpy as np
import gymnasium as gym
import asyncio

# =============================================================================
#  CARLA ground-debug drawing of the planned racing line
# =============================================================================
# roar_py reports world coordinates in a right-handed frame (y points left);
# CARLA's native debug API is left-handed (y points right). So we negate Y when
# converting back. If the painted line comes out MIRRORED across the track, flip
# this to False.
DEBUG_FLIP_Y = True

def _carla_speed_color(frac: float) -> "carla.Color":
    frac = float(max(0.0, min(1.0, frac)))
    if frac < 0.5:
        t = frac / 0.5
        return carla.Color(220, int(60 + 140 * t), 60)            # red -> yellow
    t = (frac - 0.5) / 0.5
    return carla.Color(int(220 - 140 * t), int(200 + 20 * t), int(60 + 60 * t))  # -> green

def _draw_planned_line(debug_world, path_3d: np.ndarray, speeds: np.ndarray,
                       flip_y: bool = DEBUG_FLIP_Y, z_off: float = 0.3, step: int = 4):
    """Paint the planned racing line onto the track once, colored by target
    speed (red = slow apex, green = fast). Persistent (life_time = 0)."""
    n = len(path_3d)
    vmin, vmax = float(speeds.min()), float(speeds.max())
    rng = max(vmax - vmin, 1e-3)
    sy = -1.0 if flip_y else 1.0
    for i in range(0, n, step):
        j = (i + step) % n
        p0, p1 = path_3d[i], path_3d[j]
        loc0 = carla.Location(x=float(p0[0]), y=sy * float(p0[1]), z=float(p0[2]) + z_off)
        loc1 = carla.Location(x=float(p1[0]), y=sy * float(p1[1]), z=float(p1[2]) + z_off)
        debug_world.debug.draw_line(
            loc0, loc1, thickness=0.12,
            color=_carla_speed_color((speeds[i] - vmin) / rng),
            life_time=0.0)

def _draw_distance_labels(debug_world, path_3d: np.ndarray, dist_s: np.ndarray,
                          flip_y: bool = DEBUG_FLIP_Y, z_off: float = 0.6,
                          every_m: float = 50.0):
    """Paint "NNNm" markers along the line so you can read the arc-length of the
    spot where it leaves the track and turn it into a CORRIDOR_CLAMPS entry."""
    sy = -1.0 if flip_y else 1.0
    n = len(path_3d)
    next_mark = 0.0
    for i in range(n):
        if dist_s[i] >= next_mark:
            p = path_3d[i]
            loc = carla.Location(x=float(p[0]), y=sy * float(p[1]), z=float(p[2]) + z_off)
            debug_world.debug.draw_string(
                loc, f"{int(round(dist_s[i]))}m", draw_shadow=False,
                color=carla.Color(255, 255, 255), life_time=0.0)
            next_mark += every_m

class RoarCompetitionRule:
    def __init__(
        self,
        waypoints : List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_carla.RoarPyCarlaActor,
        world: roar_py_carla.RoarPyCarlaWorld
    ) -> None:
        self.waypoints = waypoints
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
        self._respawn_location = None
        self._respawn_rpy = None

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i,waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        self.waypoints = self.waypoints[closest_waypoint_idx+1:] + self.waypoints[:closest_waypoint_idx+1]
        self.furthest_waypoints_index = 0
        print(f"total length: {len(self.waypoints)}")
        self._respawn_location = self._last_vehicle_location.copy()
        self._respawn_rpy = self.vehicle.get_roll_pitch_yaw().copy()

    def lap_finished(
        self, 
        check_step = 5
    ):
        return self.furthest_waypoints_index + check_step >= len(self.waypoints)

    async def tick(
        self, 
        check_step = 15
    ):
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = (delta_vector / delta_vector_norm) if delta_vector_norm >= 1e-5 else np.zeros(3)

        previous_furthest_index = self.furthest_waypoints_index
        min_dis = np.inf
        min_index = 0
        endind_index = previous_furthest_index + check_step if (previous_furthest_index + check_step <= len(self.waypoints)) else len(self.waypoints)
        for i,waypoint in enumerate(self.waypoints[previous_furthest_index:endind_index]):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta,delta_vector_unit)
            projection = np.clip(projection,0,delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            if distance < min_dis:
                min_dis = distance
                min_index = i
        
        self.furthest_waypoints_index += min_index
        self._last_vehicle_location = current_location
        #print(f"reach waypoints {self.furthest_waypoints_index} at {self.waypoints[self.furthest_waypoints_index].location}")

    async def respawn(
        self
    ):
        self.vehicle.set_transform(
            self._respawn_location, self._respawn_rpy
        )
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            await self.world.step()
        
        self._last_vehicle_location = self.vehicle.get_3d_location()
        self.furthest_waypoints_index = 0

async def evaluate_solution(
    world : roar_py_carla.RoarPyCarlaWorld,
    solution_constructor : Type[RoarCompetitionSolution],
    max_seconds = 12000,
    enable_visualization : bool = False,
    debug_world = None,
) -> Optional[Dict[str, Any]]:
    if enable_visualization:
        viewer = ManualControlViewer()

    # Spawn vehicle and sensors to receive data
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        waypoints[0].location + np.array([0,0,1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=1024,
        image_height=768
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
        50,
        50,
        2.0,
        2.0
    )
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3)
    )

    assert camera is not None
    assert location_sensor is not None
    assert velocity_sensor is not None
    assert rpy_sensor is not None
    assert occupancy_map_sensor is not None
    assert collision_sensor is not None


    # Start to run solution 
    solution : RoarCompetitionSolution = solution_constructor(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        camera,
        location_sensor,
        velocity_sensor,
        rpy_sensor,
        occupancy_map_sensor,
        collision_sensor
    )
    rule = RoarCompetitionRule(waypoints * 3,vehicle,world) # 3 laps

    for _ in range(20):
        await world.step()
    
    rule.initialize_race()

    total_waypoints = len(rule.waypoints)
    per_lap = max(1, total_waypoints // 3)

    # Timer starts here 
    start_time = world.last_tick_elapsed_seconds
    current_time = start_time
    await vehicle.receive_observation()
    await solution.initialize()

    # ----- Hand the planned trajectory to the visualizers --------------------
    if enable_visualization and hasattr(solution, "path"):
        try:
            viewer.set_trajectory(solution.path, solution.v_profile,
                                  section_of=getattr(solution, "section_of", None))
        except Exception as e:
            print(f"Could not set track map: {e}")
    if debug_world is not None and hasattr(solution, "path_3d"):
        try:
            _draw_planned_line(debug_world,
                               np.asarray(solution.path_3d),
                               np.asarray(solution.v_profile))
            if hasattr(solution, "centerline_s"):
                _draw_distance_labels(debug_world,
                                      np.asarray(solution.path_3d),
                                      np.asarray(solution.centerline_s))
            print("Painted planned racing line + distance labels in the CARLA "
                  "world (visible in the simulator/spectator window).")
        except Exception as e:
            print(f"Could not draw ground line: {e}")

    
    while True:
        # terminate if time out
        current_time = world.last_tick_elapsed_seconds
        if current_time - start_time > max_seconds:
            vehicle.close()
            return None
        
        # receive sensors' data
        await vehicle.receive_observation()

        await rule.tick()

        # terminate if there is major collision
        collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > 100.0:
            print(f"major collision of tensity {collision_impulse_norm}")
            await rule.respawn()
        
        if rule.lap_finished():
            break

        # Step the solution first so the dashboard shows the command for this frame.
        control = await solution.step()
        control = control if isinstance(control, dict) else {}

        if enable_visualization:
            loc_xy = np.asarray(location_sensor.get_last_gym_observation())[:2]
            car_yaw = float(rpy_sensor.get_last_gym_observation()[2])
            speed = float(np.linalg.norm(
                np.asarray(velocity_sensor.get_last_gym_observation())))
            cp = rule.furthest_waypoints_index

            telemetry = {
                "throttle"        : float(control.get("throttle", 0.0)),
                "brake"           : float(control.get("brake", 0.0)),
                "steer"           : float(control.get("steer", 0.0)),
                "speed"           : speed,
                "checkpoint"      : cp,
                "total_waypoints" : total_waypoints,
                "lap"             : min(3, cp // per_lap + 1),
                "total_laps"      : 3,
                "elapsed"         : current_time - start_time,
                "collision"       : float(collision_impulse_norm),
                "car_xy"          : (float(loc_xy[0]), float(loc_xy[1])),
                "car_yaw"         : car_yaw,
            }
            # Optional extras the solution publishes (target_speed, lat_g, ...)
            extra = getattr(solution, "telemetry", None)
            if isinstance(extra, dict):
                telemetry.update(extra)

            if viewer.render(camera.get_last_observation(), telemetry=telemetry) is None:
                vehicle.close()
                return None

        await world.step()
    
    print("end of the loop")
    end_time = world.last_tick_elapsed_seconds
    vehicle.close()
    if enable_visualization:
        viewer.close()
    
    return {
        "elapsed_time" : end_time - start_time,
    }

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)

    # Native CARLA world handle used only for ground-debug drawing of the line.
    try:
        debug_world = carla_client.get_world()
    except Exception:
        debug_world = None

    evaluation_result = await evaluate_solution(
        world,
        RoarCompetitionSolution,
        max_seconds=5000,
        enable_visualization=True,
        debug_world=debug_world,
    )
    if evaluation_result is not None:
        print("Solution finished in {} seconds".format(evaluation_result["elapsed_time"]))
    else:
        print("Solution failed to finish in time")

if __name__ == "__main__":
    asyncio.run(main())