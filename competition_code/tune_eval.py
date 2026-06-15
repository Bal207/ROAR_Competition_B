"""
Headless evaluation worker for the autotuner.

Runs ONE parameter set through a full (multi-lap) session with no visualization,
and writes a JSON result the tuner reads. Designed to be launched as a subprocess
so a CARLA hang/crash never takes down the optimizer.

Usage (the tuner calls this for you):
    python tune_eval.py --params p.json --out r.json --port 2000 \
        --laps 3 --max-seconds 700 --best 389 --collision-penalty 3 --abort-factor 1.25
"""
import argparse, json, asyncio, traceback
import numpy as np
import carla
import roar_py_carla
import roar_py_interface

import submission
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper
from competition_runner import RoarCompetitionRule

COLLISION_CAP = 8   # abandon a candidate that keeps crashing


def _cleanup_stray_vehicles(client):
    """Destroy vehicles left behind by a previously killed worker."""
    try:
        cw = client.get_world()
        for a in cw.get_actors().filter("vehicle.*"):
            try:
                a.destroy()
            except Exception:
                pass
    except Exception:
        pass


async def run_eval(world, laps, max_seconds, collision_penalty, best, abort_factor):
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.dallara.dallara",
        waypoints[0].location + np.array([0, 0, 1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None

    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))

    solution = RoarCompetitionSolution(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        None, location_sensor, velocity_sensor, rpy_sensor, None, collision_sensor,
    )
    rule = RoarCompetitionRule(waypoints * laps, vehicle, world)

    for _ in range(20):
        await world.step()
    rule.initialize_race()

    total = len(rule.waypoints)
    per_lap = max(1, total // laps)
    start = world.last_tick_elapsed_seconds
    await vehicle.receive_observation()
    await solution.initialize()

    collisions = 0
    lap_cross = []
    next_split = per_lap
    finished = False
    aborted = False

    while True:
        elapsed = world.last_tick_elapsed_seconds - start
        if elapsed > max_seconds:
            break
        # Cut hopeless candidates early to save wall-clock.
        if best and best > 0 and elapsed > best * abort_factor and not finished:
            aborted = True
            break

        await vehicle.receive_observation()
        await rule.tick()

        imp = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
        if imp > 100.0:
            collisions += 1
            await rule.respawn()
            if collisions > COLLISION_CAP:
                aborted = True
                break

        if rule.furthest_waypoints_index >= next_split and len(lap_cross) < laps:
            lap_cross.append(world.last_tick_elapsed_seconds - start)
            next_split += per_lap

        if rule.lap_finished():
            finished = True
            break

        await solution.step()
        await world.step()

    elapsed = world.last_tick_elapsed_seconds - start
    progress = float(min(1.0, rule.furthest_waypoints_index / max(1, total)))
    try:
        vehicle.close()
    except Exception:
        pass

    splits = [lap_cross[0]] + [lap_cross[i] - lap_cross[i - 1] for i in range(1, len(lap_cross))]

    if finished:
        objective = elapsed + collision_penalty * collisions
    else:
        # Any non-finish ranks strictly worse than any finish (<= max_seconds),
        # but more progress is rewarded so the optimizer can climb out.
        objective = max_seconds + (1.0 - progress) * max_seconds

    return {
        "objective": float(objective),
        "elapsed": float(elapsed),
        "finished": bool(finished),
        "aborted": bool(aborted),
        "progress": progress,
        "collisions": int(collisions),
        "lap_splits": [float(s) for s in splits],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=700.0)
    ap.add_argument("--best", type=float, default=0.0)
    ap.add_argument("--collision-penalty", type=float, default=3.0)
    ap.add_argument("--abort-factor", type=float, default=1.25)
    args = ap.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    submission.OVERRIDES = params  # picked up by the solution at construction

    result = {"objective": 2 * args.max_seconds, "finished": False,
              "error": None, "params": params}
    client = None
    try:
        client = carla.Client("127.0.0.1", args.port)
        client.set_timeout(30.0)
        _cleanup_stray_vehicles(client)
        roar_py_instance = roar_py_carla.RoarPyCarlaInstance(client)
        world = roar_py_instance.world
        world.set_control_steps(0.05, 0.005)
        world.set_asynchronous(False)
        res = asyncio.run(run_eval(
            world, args.laps, args.max_seconds,
            args.collision_penalty, args.best, args.abort_factor))
        result.update(res)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        try:
            roar_py_instance.close()
        except Exception:
            pass

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[eval] obj={result['objective']:.1f} finished={result.get('finished')} "
          f"elapsed={result.get('elapsed')} collisions={result.get('collisions')} "
          f"splits={result.get('lap_splits')}")


if __name__ == "__main__":
    main()