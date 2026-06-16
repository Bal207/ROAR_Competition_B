"""
Background autotuner for the ROAR solution.

Runs forever, proposing parameter sets and scoring each by running a headless
session in a subprocess (tune_eval.py). Persists every trial to SQLite so it is
resumable and always has a best-so-far saved. Seeds the search with your proven
389 config so it starts from known-good and refines.

QUICK START
-----------
1. pip install optuna
2. Start CARLA (the same server you normally run), reachable on --port.
3. python tuner.py

Stop any time with Ctrl-C; progress is saved. Re-run to resume.
The best parameters are written to best_params.json and a ready-to-submit
submission_best.py every time the record improves.
"""
import argparse, json, os, subprocess, sys, tempfile, time
import statistics

import optuna

# ---- Bulletproof Pathing ----------------------------------------------------
# This guarantees the tuner can find its companion files from anywhere in Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- proven baseline (your 389 config) used to seed the search --------------
SEED = {
    "GRIP_MARGIN": 2.05,
    "THROTTLE_GRIP_MARGIN": 2.05,
    "K_DF": 0.0,
    "K_DF_BRAKE": 0.0,
    "A_BRAKE": 30.3,
    "A_ACCEL": 200.0,
    "V_MAX": 300.0,
    "STANLEY_K": 5.0,
    "STANLEY_K_SOFT": 2.0,
    "PID_KP": 256.0,
    "PID_KI": 0.1,
    "ACTUATOR_LAG_S": 0.4,
    "SAFETY_BUFFER": 0.65,
    "CURV_WIN_M": 5.0,
}

# ---- search space -----------------------------------------------------------
def sample(trial):
    return {
        "GRIP_MARGIN":          trial.suggest_float("GRIP_MARGIN", 1.4, 2.4),
        "THROTTLE_GRIP_MARGIN": trial.suggest_float("THROTTLE_GRIP_MARGIN", 1.0, 2.4),
        "K_DF":                 trial.suggest_float("K_DF", 0.0, 0.004),
        "K_DF_BRAKE":           trial.suggest_float("K_DF_BRAKE", 0.0, 0.003),
        "A_BRAKE":              trial.suggest_float("A_BRAKE", 15.0, 40.0),
        "A_ACCEL":              trial.suggest_float("A_ACCEL", 30.0, 220.0),
        "V_MAX":                300.0,  # effectively uncapped; not worth a dim
        "STANLEY_K":            trial.suggest_float("STANLEY_K", 0.5, 6.0),
        "STANLEY_K_SOFT":       trial.suggest_float("STANLEY_K_SOFT", 1.0, 6.0),
        "PID_KP":               trial.suggest_float("PID_KP", 32.0, 300.0),
        "PID_KI":               trial.suggest_float("PID_KI", 0.0, 1.0),
        "ACTUATOR_LAG_S":       trial.suggest_float("ACTUATOR_LAG_S", 0.1, 0.8),
        "SAFETY_BUFFER":        trial.suggest_float("SAFETY_BUFFER", 0.40, 1.20),
        "CURV_WIN_M":           trial.suggest_float("CURV_WIN_M", 3.0, 8.0),
    }


def best_finish_value(study, max_seconds):
    """Lowest objective among trials that actually finished (< max_seconds)."""
    vals = [t.value for t in study.trials
            if t.value is not None and t.value < max_seconds]
    return min(vals) if vals else 0.0


def run_worker(params, args, best):
    """Launch one headless session, return its result dict (penalty on failure)."""
    eval_script = os.path.join(BASE_DIR, "tune_eval.py")
    
    with tempfile.TemporaryDirectory() as d:
        pj = os.path.join(d, "p.json")
        rj = os.path.join(d, "r.json")
        with open(pj, "w") as f:
            json.dump(params, f)
            
        cmd = [sys.executable, eval_script,
               "--params", pj, "--out", rj,
               "--port", str(args.port),
               "--laps", str(args.laps),
               "--max-seconds", str(args.max_seconds),
               "--best", str(best),
               "--collision-penalty", str(args.collision_penalty),
               "--abort-factor", str(args.abort_factor)]
        try:
            subprocess.run(cmd, timeout=args.timeout, check=False)
            with open(rj) as f:
                return json.load(f)
        except subprocess.TimeoutExpired:
            return {"objective": 2 * args.max_seconds, "finished": False,
                    "error": "timeout"}
        except Exception as e:
            return {"objective": 2 * args.max_seconds, "finished": False,
                    "error": repr(e)}


def write_best_submission(params):
    """Bake params into a standalone submission_best.py using absolute paths."""
    submission_path = os.path.join(BASE_DIR, "submission.py")
    best_sub_path = os.path.join(BASE_DIR, "submission_best.py")
    json_path = os.path.join(BASE_DIR, "best_params.json")

    # Only attempt to modify submission.py if it exists in the folder
    if os.path.exists(submission_path):
        with open(submission_path) as f:
            src = f.read()
        line = "OVERRIDES: Dict[str, float] = " + json.dumps(params)
        out = []
        replaced = False
        for ln in src.splitlines():
            if ln.startswith("OVERRIDES") and not replaced:
                out.append(line)
                replaced = True
            else:
                out.append(ln)
        with open(best_sub_path, "w") as f:
            f.write("\n".join(out) + "\n")
    else:
        print(f"Warning: Could not find {submission_path} to build submission_best.py")

    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)


def make_objective(args):
    def objective(trial):
        params = sample(trial)
        study = trial.study
        best = best_finish_value(study, args.max_seconds)

        objs, infos = [], []
        for r in range(args.repeats):
            res = run_worker(params, args, best)
            objs.append(res["objective"])
            infos.append(res)
        obj = sum(objs) / len(objs)

        fin = infos[-1]
        trial.set_user_attr("finished", fin.get("finished"))
        trial.set_user_attr("elapsed", fin.get("elapsed"))
        trial.set_user_attr("collisions", fin.get("collisions"))
        trial.set_user_attr("splits", fin.get("lap_splits"))
        if args.repeats > 1:
            trial.set_user_attr("obj_spread", max(objs) - min(objs))
        return obj
    return objective


def improvement_callback(study, trial):
    if trial.value is None:
        return
    if study.best_trial.number == trial.number:
        params = {**SEED, **trial.params}
        write_best_submission(params)
        print(f"\n*** NEW BEST  obj={trial.value:.2f}  "
              f"(trial {trial.number}) -> submission_best.py\n{params}\n")


def probe(args):
    """Run the seed config N times to measure run-to-run variance."""
    print(f"Probing seed config {args.probe}x to measure variance...")
    vals = []
    for i in range(args.probe):
        res = run_worker(SEED, args, best=0.0)
        vals.append(res["objective"])
        print(f"  run {i+1}: obj={res['objective']:.2f} "
              f"finished={res.get('finished')} collisions={res.get('collisions')} "
              f"splits={res.get('lap_splits')}")
    if len(vals) > 1:
        print(f"\nmean={statistics.mean(vals):.2f}  "
              f"stdev={statistics.pstdev(vals):.2f}  "
              f"spread={max(vals)-min(vals):.2f}")
        print("If spread is more than ~2-3 s, run the tuner with --repeats 2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--laps", type=int, default=3,
                    help="laps per session (1 for a fast, noisier screen)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="score each candidate as the mean of this many sessions")
    ap.add_argument("--max-seconds", type=float, default=700.0)
    ap.add_argument("--collision-penalty", type=float, default=3.0)
    ap.add_argument("--abort-factor", type=float, default=1.25)
    ap.add_argument("--timeout", type=float, default=1200.0,
                    help="wall-clock seconds before a stuck session is killed")
    ap.add_argument("--study", default="roar")
    ap.add_argument("--storage", default="sqlite:///roar_tuning.db")
    ap.add_argument("--probe", type=int, default=0,
                    help="just run the seed config N times to gauge variance, then exit")
    args = ap.parse_args()

    if args.probe:
        probe(args)
        return

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True,
                                         n_startup_trials=16, seed=1)
    study = optuna.create_study(
        study_name=args.study, storage=args.storage,
        direction="minimize", sampler=sampler, load_if_exists=True)

    if len(study.get_trials(deepcopy=False)) == 0:
        study.enqueue_trial({k: v for k, v in SEED.items() if k != "V_MAX"})
        print("Seeded study with the proven baseline config.")

    print(f"Tuning on port {args.port}, laps={args.laps}, repeats={args.repeats}. "
          f"Ctrl-C to stop (progress saved).")
    study.optimize(make_objective(args), n_trials=None, timeout=None,
                   callbacks=[improvement_callback], gc_after_trial=True)


if __name__ == "__main__":
    main()