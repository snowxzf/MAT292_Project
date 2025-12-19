import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
import time, os
import pandas as pd
import re
from math import isfinite

# Timing=
start_time_total = time.time()

# Metrics
def mean_absolute_scaled_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return np.nan
    naive_forecast = np.mean(np.abs(np.diff(y_true)))
    return np.mean(np.abs(y_true - y_pred)) / (naive_forecast + 1e-12)

def chi_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(((y_true - y_pred) ** 2) / (y_pred + 1e-12))

def nash_sutcliffe_efficiency(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred)**2) / (denom + 1e-12)

def kling_gupta_efficiency(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-12)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# Utility functions
dt_sim_default = 0.5
def sigmoid(x):
    # safe sigmoid to avoid overflow 
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def get_patient_data(patient_id, file_path="tumour_data.csv"):
    '''Loads patient data from tumour_data.csv.'''
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠volumes file not found: {file_path}")
        return None, None
    pid_str = str(patient_id).zfill(3)
    df_patient = df[df["Patient"].astype(str).str.contains(pid_str, case=False, na=False)]
    if df_patient.empty:
        print("No data for patient", patient_id)
        return None, None
    weeks = []
    for w in df_patient["Week"]:
        match = re.search(r"(\d+)", str(w))
        if match:
            weeks.append(float(match.group(1)))
    weeks = np.array(weeks, dtype=float)
    volumes = df_patient["Total_mm3"].astype(float).to_numpy()
    sort_idx = np.argsort(weeks)
    weeks = weeks[sort_idx]
    volumes = volumes[sort_idx]
    return weeks, volumes

def sanitize_tumor_data(time_data, tumor_data):
    '''Cleans tumor data by removing non-finite values and handling zeros.'''
    t = np.array(time_data, dtype=float)
    v = np.array(tumor_data, dtype=float)
    valid_mask = np.isfinite(v)
    t, v = t[valid_mask], v[valid_mask]
    if len(v) == 0:
        return t, v
    if np.any(v <= 0):
        nonzero_idx = np.where(v > 0)[0]
        zero_idx = np.where(v <= 0)[0]
        if len(nonzero_idx) > 1:
            v[zero_idx] = np.interp(zero_idx, nonzero_idx, v[nonzero_idx])
        else:
            v = np.maximum(v, 10.0)
    # clip tumor volume to prevent extreme spikes
    v = np.clip(v, 1.0, 1e5)
    return t, v

# EM SDE functions
def a_of_t(t, a1, a2, alpha, t_c):
    #clip alpha*(t-tc) to avoid extreme tanh slope 
    return a1 + a2 * np.tanh(np.clip(alpha * (t - t_c), -50, 50))

def b_of_t(t, b1, b2, beta, t_b):
    # safe sigmoid used
    return b1 + b2 * sigmoid(beta * (t - t_b))

def euler_maruyama_functional_sde(params, V0, t_grid, chemo_start=3, chemo_end=21,
                                  zero_tol=1e-2, rng=None, dt_sim=dt_sim_default, V_data=None):
    """
    Euler-Maruyama SDE simulator that returns values at t_grid points.
    Numerics stabilised: clamped logs, V floor, mild drift dampening, safe diffusion denom.
    """
    a1, a2, alpha, t_c, b1, b2, beta, t_b, c, h, k0 = params
    if rng is None:
        rng = np.random.default_rng()
    t_grid = np.asarray(t_grid)
    T = float(t_grid[-1])
    # choose number of steps based on dt_sim but ensure integer 
    n_steps = max(2, int(np.ceil(T / float(dt_sim))))
    dt = T / n_steps
    # protect tiny dt from producing huge step counts 
    if dt <= 0:
        dt = 1e-6
        n_steps = max(n_steps, 1)
    dW = rng.normal(0.0, np.sqrt(dt), size=n_steps)
    V = float(max(V0, 1e-6))
    out = np.empty(len(t_grid))
    idx = 0
    ti = 0.0
    V_max = (np.max(V_data) * 1.1) if (V_data is not None and len(V_data) > 0) else 1e5
    # small damping factor so very large drift doesn't overshoot in a single EM step
    max_rel_increment = 0.5  # limits relative increment per step (50%)
    for step in range(n_steps):
        V = max(V, 1e-12)
        #  coefficients safely
        at = a_of_t(ti, a1, a2, alpha, t_c)
        bt = b_of_t(ti, b1, b2, beta, t_b)
        k = k0 if (chemo_start <= ti <= chemo_end) else 0.0
        # clamp both numerator and ratio for stability
        bt_clamped = np.clip(bt, 1e-3, max(V_max, 1e3))
        ratio = bt_clamped / max(V, 1e-12)
        ratio = np.clip(ratio, 1e-12, 1e6)   # avoid inf logs
        log_term = np.log(ratio)
        # drift and optional damping after chemo window
        drift = at * V * log_term - k * V
        if ti > chemo_end:
            drift *= max(1 - V / max(V_max, 1e-6), 0.0)
        # limit relative drift increment to avoid single-step explosions
        rel_drift = drift * dt / max(V, 1e-12)
        if abs(rel_drift) > max_rel_increment:
            # damp proportionally
            drift = np.sign(drift) * max_rel_increment * max(V, 1e-12) / dt
        diffusion = c * V / max(h + np.sqrt(max(V, 0.0)), 1e-12)
        # limit diffusion scale to avoid single-step explosion with big noise
        max_diff_allowed = max_rel_increment * max(V, 1e-12) / max(np.sqrt(dt), 1e-6)
        diffusion = np.clip(diffusion, -abs(max_diff_allowed), abs(max_diff_allowed))
        V_new = V + drift * dt + diffusion * dW[step]
        # enforce bounds & finite
        if not isfinite(V_new) or V_new > 1e6:
            V_new = min(max(V_new if isfinite(V_new) else V, 1e-12), 1e6)
        V_new = max(V_new, 1e-12)
        V = V_new
        ti += dt
        # record at appropriate output times (advance index while we've passed grid times)
        while idx < len(t_grid) and ti >= t_grid[idx] - 1e-12:
            out[idx] = V
            idx += 1
            if idx >= len(t_grid):
                break
    # if some final indices weren't filled (numerical rounding), fill with final V
    while idx < len(t_grid):
        out[idx] = V
        idx += 1
    return out

def simulate_ensemble_functional(params, V0, t_grid, chemo_start=3, chemo_end=21,
                                 M=30, dt_sim=dt_sim_default, seed=None, V_data=None):
    '''Simulates an ensemble of Euler-Maruyama SDE trajectories and returns mean, std, and all trajectories.'''
    rng = np.random.default_rng(seed)
    trajs = np.empty((M, len(t_grid)))
    for i in range(M):
        # each gets a new RNG instance, reproducibility
        subseed = int(rng.integers(0, 1<<30))
        trajs[i] = euler_maruyama_functional_sde(params, V0, t_grid,
                                                 chemo_start=chemo_start, chemo_end=chemo_end,
                                                 rng=np.random.default_rng(subseed), dt_sim=dt_sim, V_data=V_data)
    mean = np.mean(trajs, axis=0)
    std = np.std(trajs, axis=0)
    return mean, std, trajs

# Fit functions
def functional_logmse(params, time_data, observed, V0, M=30, dt_sim=dt_sim_default, seed=42):
    '''Computes log-scale mean squared error between observed and simulated mean trajectory.'''
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, time_data,
                                                   chemo_start=3, chemo_end=21,
                                                   M=M, dt_sim=dt_sim, seed=seed, V_data=observed)
    # if any non-finite, sanction heavily!!!
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj <= 0):
        return 1e30
    return np.sum((np.log(mean_traj + 1e-12) - np.log(observed + 1e-12))**2)

def functional_residuals(params, time_data, observed, V0, M=20, dt_sim=dt_sim_default, seed=123):
    '''Computes log-scale residuals between observed and simulated mean trajectory.'''
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, time_data,
                                                   chemo_start=3, chemo_end=21,
                                                   M=M, dt_sim=dt_sim, seed=seed, V_data=observed)
    # guard
    mean_traj = np.maximum(mean_traj, 1e-12)
    return np.log(mean_traj + 1e-12) - np.log(observed + 1e-12)

def fit_functional_model(time_data, observed, V0=None, seed=42):
    '''Fits the functional SDE model to observed data using differential evolution and least squares.'''
    if V0 is None: V0 = observed[0]
    obs_max = max(observed)
    bounds = [(-0.5,0.5), (-1,1), (0.01,2), (0,time_data[-1]),
              (max(1.0, obs_max*0.2), obs_max*5), (0, obs_max*10), (0.01,2), (0,time_data[-1]),
              (0,2), (1e-6,1e5), (0,0.8)]
    #  bounds are strictly increasing
    for i, (low, high) in enumerate(bounds):
        if low >= high:
            bounds[i] = (low, low + 1e-3)
    result = differential_evolution(lambda p: functional_logmse(p, time_data, observed, V0, dt_sim=dt_sim_default), bounds, seed=seed, maxiter=10)
    x0 = result.x
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ls_result = least_squares(lambda p: functional_residuals(p, time_data, observed, V0, dt_sim=dt_sim_default), x0, bounds=(lower, upper))
    return ls_result.x

# EM convergence & trajectory functions
def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=30, tol=1e-4, window=5, seed=123):
    '''Calculates the Trajectory Convergence Time (TCT) based on relative changes in the mean trajectory.'''
    t_grid = np.arange(0, T + dt, dt)
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=seed)
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    L = len(rel_changes)
    for i in range(L - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i+1+window-1]
    return None

def em_convergence_test(params, V0, T, dt_values=None, M=50, base_seed=123):
    '''Runs convergence tests for strong and weak convergence of the Euler-Maruyama method.'''
    if dt_values is None:
        dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    # reference dt: small dt for the "reference" simulation but also ensure dt_sim used consistently
    ref_dt = min(0.01, dt_values[-1]/2)
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    mean_ref, _, sims_ref = simulate_ensemble_functional(params, V0, t_ref, M=M, dt_sim=ref_dt, seed=base_seed)
    ref_final = sims_ref[:, -1]
    mean_ref_final = np.mean(ref_final)
    results = {"dt_values": [], "strong_error": [], "weak_error": [], "runtime": []}
    for dt in dt_values:
        start = time.time()
        t_grid = np.arange(0, T + dt, dt)
        #use dt as dt_sim to keep ensemble discretization consistent with t_grid spacing 
        mean_dt, _, sims_dt = simulate_ensemble_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=base_seed)
        ref_interp = np.interp(t_grid, t_ref, np.mean(sims_ref, axis=0))
        strong_err = np.mean(np.abs(mean_dt - ref_interp))
        weak_err = np.abs(np.mean(sims_dt[:, -1]) - mean_ref_final)
        results["dt_values"].append(dt)
        results["strong_error"].append(strong_err)
        results["weak_error"].append(weak_err)
        results["runtime"].append(time.time() - start)
    for k in ["dt_values","strong_error","weak_error","runtime"]:
        results[k] = np.array(results[k])
    return results

def save_em_convergence_plots(conv, patient_id, outdir="em_conv/patients"):
    os.makedirs(outdir, exist_ok=True)
    dt = conv["dt_values"]
    strong = np.clip(conv["strong_error"], 1e-12, 1e6)
    weak = np.clip(conv["weak_error"], 1e-12, 1e6)
    runtime = conv["runtime"]
    plt.figure(figsize=(6,4))
    plt.loglog(dt, strong, 'o-', label='Strong Error')
    plt.loglog(dt, weak, 's--', label='Weak Error')
    plt.xlabel("Δt"); plt.ylabel("Error")
    plt.legend()
    plt.title(f"Patient {patient_id} — EM Convergence")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"patient_{patient_id:03d}_conv.png"), dpi=300)
    plt.close()
    plt.figure(figsize=(6,4))
    plt.loglog(dt, runtime, 'd-', color='purple')
    plt.xlabel("Δt"); plt.ylabel("Runtime (s)")
    plt.title(f"Patient {patient_id} — Convergence runtime")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"patient_{patient_id:03d}_runtime.png"), dpi=300)
    plt.close()
#directories
os.makedirs("em_traj/patients", exist_ok=True)
os.makedirs("em_traj/batches", exist_ok=True)
os.makedirs("em_conv/patients", exist_ok=True)
os.makedirs("em_conv/batches", exist_ok=True)
# convergence batch buffers (per-batch & global) 
conv_batch_store = [] # list of (patient_id, conv_dict)
conv_batch_size = 10
conv_batch_index = 1
global_conv_list = [] # store conv dicts for final aggregation
# Batch Loop with Detailed Progress & Timing
batch_size = 10
metrics_results = {}
all_means, all_times, all_data, all_ids = [], [], [], []
batch_index = 1
for i in range(1, 93):
    t0 = time.time()
    print(f"\nProcessing Patient {i}")
    time_data, tum = get_patient_data(i)
    if time_data is None or tum is None:
        continue
    time_data, tumor_volume_data = sanitize_tumor_data(time_data, tum)
    if len(tumor_volume_data) < 3:
        print("insufficient valid data.")
        continue
    print("global search")
    try:
        fitted = fit_functional_model(time_data, tumor_volume_data, V0=tumor_volume_data[0], seed=42)
    except ValueError as e:
        print(f" Skipping patient {i} due to fit error: {e}")
        continue
    print("Least_squares")
    print(f"Fitted params: {np.array2string(fitted, precision=4, separator=' ')}")
    #  trajectory
    t_fine = np.linspace(0, time_data[-1], 500)
    mean_fit, std_fit, trajs = simulate_ensemble_functional(fitted, tumor_volume_data[0], t_fine,
                                                            chemo_start=3, chemo_end=21,
                                                            M=30, dt_sim=dt_sim_default, seed=42,
                                                            V_data=tumor_volume_data)
    # Metrics
    y_pred_interp = np.interp(time_data, t_fine, mean_fit)
    mase = mean_absolute_scaled_error(tumor_volume_data, y_pred_interp)
    chi2 = chi_squared(tumor_volume_data, y_pred_interp)
    nse = nash_sutcliffe_efficiency(tumor_volume_data, y_pred_interp)
    kge = kling_gupta_efficiency(tumor_volume_data, y_pred_interp)
    metrics_results[i] = {"MASE": mase, "Chi2": chi2, "NSE": nse, "KGE": kge}
    #  convergence (per-patient)
    traj_conv_time = trajectory_convergence_time_from_ensemble(fitted, tumor_volume_data[0], T=time_data[-1], M=30, dt=0.2, seed=i)
    conv = em_convergence_test(fitted, tumor_volume_data[0], T=time_data[-1], M=50)
    save_em_convergence_plots(conv, i)
    metrics_results[i].update({"TrajectoryConvergenceTime": traj_conv_time, "Convergence": conv})
    # append to batch and global conv stores 
    conv_batch_store.append((i, conv))
    global_conv_list.append(conv)
    print(f"Metrics → MASE={mase:.4f}, Chi²={chi2:.4f}, NSE={nse:.4f}, KGE={kge:.4f}, "
          f"TrajConvTime={traj_conv_time}")
    #  per-patient trajectory plot
    plt.figure(figsize=(8,4.5))
    plt.plot(t_fine, mean_fit, '-', label='Mean Trajectory')
    plt.fill_between(t_fine, mean_fit - 1.96*std_fit, mean_fit + 1.96*std_fit, alpha=0.15, label="95% CI")
    plt.scatter(time_data, tumor_volume_data, color='k', s=10, label='Data')
    plt.xlabel("Time (weeks)")
    plt.ylabel("Tumor Volume (mm³)")
    plt.ylim(0, max(np.max(mean_fit), np.max(tumor_volume_data))*1.1)  # clip y-axis
    plt.title(f"Patient {i} — Tumor Trajectory")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"em_traj/patients/patient_{i:03d}_trajectory.png", dpi=300)
    plt.close()
    # trajectory batch plotting
    all_means.append(mean_fit)
    all_times.append(t_fine)
    all_data.append((time_data, tumor_volume_data))
    all_ids.append(i)
    print(f"Patient {i} done — runtime: {time.time() - t0:.2f}s "
          f"(total elapsed: {time.time() - start_time_total:.1f}s)")
    if len(all_means) == batch_size:
        cmap = plt.cm.tab10(np.linspace(0, 1, batch_size))
        plt.figure(figsize=(10, 5))
        for j, color in enumerate(cmap):
            plt.plot(all_times[j], all_means[j], color=color, label=f"Patient {all_ids[j]}")
            plt.scatter(all_data[j][0], all_data[j][1], color=color, s=10, alpha=0.6)
        plt.axvspan(3, 21, color='gray', alpha=0.2, label='Chemo Window')
        plt.xlabel("Time (weeks)")
        plt.ylabel("Tumor Volume (mm³)")
        plt.title(f"Batch {batch_index}: Functional SDE Fits")
        plt.legend(fontsize=8)
        plt.ylim(0, np.max([np.max(m) for m in all_means])*1.1)
        plt.tight_layout()
        plt.savefig(f"em_traj/batches/batch_{batch_index:03d}.png", dpi=300)
        plt.close()
        print(f"Saved batch {batch_index} trajectory plots.")
        batch_index += 1
        all_means, all_times, all_data, all_ids = [], [], [], []
    #Convergence batching: when conv_batch_store reaches conv_batch_size, save batch plots & csv 
    if len(conv_batch_store) == conv_batch_size:
        # extract dt_values from first conv (they're identical)
        dt_vals = conv_batch_store[0][1]["dt_values"]
        strong_stack = np.vstack([c[1]["strong_error"] for c in conv_batch_store])
        weak_stack = np.vstack([c[1]["weak_error"] for c in conv_batch_store])
        runtime_stack = np.vstack([c[1]["runtime"] for c in conv_batch_store])
        patient_ids = [c[0] for c in conv_batch_store]
        # Save combined strong error batch plot with min/max envelope
        plt.figure(figsize=(8,5))
        for j in range(conv_batch_size):
            plt.loglog(dt_vals, strong_stack[j], alpha=0.6, label=f"P{patient_ids[j]}")
        # mean + envelope
        mean_strong = np.mean(strong_stack, axis=0)
        min_strong = np.min(strong_stack, axis=0)
        max_strong = np.max(strong_stack, axis=0)
        plt.fill_between(dt_vals, np.maximum(min_strong, 1e-12), max_strong, alpha=0.12, label='min-max envelope')
        plt.loglog(dt_vals, mean_strong, 'k--', linewidth=1.5, label='mean strong')
        plt.xlabel("Δt"); plt.ylabel("Strong Error")
        plt.title(f"EM Convergence — Batch {conv_batch_index}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"em_conv/batches/batch_{conv_batch_index:03d}_strong.png", dpi=300)
        plt.close()
        # Save weak error batch plot
        plt.figure(figsize=(8,5))
        for j in range(conv_batch_size):
            plt.loglog(dt_vals, weak_stack[j], alpha=0.6, label=f"P{patient_ids[j]}")
        mean_weak = np.mean(weak_stack, axis=0)
        plt.loglog(dt_vals, mean_weak, 'k--', linewidth=1.5, label='mean weak')
        plt.xlabel("Δt"); plt.ylabel("Weak Error")
        plt.title(f"EM Weak Error — Batch {conv_batch_index}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"em_conv/batches/batch_{conv_batch_index:03d}_weak.png", dpi=300)
        plt.close()
        # Save runtime batch plot
        plt.figure(figsize=(8,5))
        for j in range(conv_batch_size):
            plt.loglog(dt_vals, runtime_stack[j], alpha=0.6, label=f"P{patient_ids[j]}")
        mean_runtime = np.mean(runtime_stack, axis=0)
        plt.loglog(dt_vals, mean_runtime, 'k--', linewidth=1.5, label='mean runtime')
        plt.xlabel("Δt"); plt.ylabel("Runtime (s)")
        plt.title(f"EM Runtime — Batch {conv_batch_index}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"em_conv/batches/batch_{conv_batch_index:03d}_runtime.png", dpi=300)
        plt.close()
        #  CSV summarizing arrays for the batch
        df = pd.DataFrame({
            "dt": np.array(dt_vals)
        })
        # append mean + std columns
        df["strong_mean"] = mean_strong
        df["strong_min"] = np.min(strong_stack, axis=0)
        df["strong_max"] = np.max(strong_stack, axis=0)
        df["weak_mean"] = mean_weak
        df["runtime_mean"] = mean_runtime
        df.to_csv(f"em_conv/batches/batch_{conv_batch_index:03d}_summary.csv", index=False)
        print(f"Saved EM convergence batch {conv_batch_index} (patients: {patient_ids})")
        # clear batch store and increment
        conv_batch_store = []
        conv_batch_index += 1
#  convergence summary (global) - use global_conv_list
if len(global_conv_list) > 0:
    #  consistent dt_values
    dt_common = global_conv_list[0]["dt_values"]
    strong_all = np.vstack([c["strong_error"] for c in global_conv_list])
    weak_all = np.vstack([c["weak_error"] for c in global_conv_list])
    runtime_all = np.vstack([c["runtime"] for c in global_conv_list])
    mean_strong = np.mean(strong_all, axis=0)
    mean_weak = np.mean(weak_all, axis=0)
    mean_runtime = np.mean(runtime_all, axis=0)
    plt.figure(figsize=(7,5))
    plt.loglog(dt_common, mean_strong, 'o-', label="Mean Strong Error")
    plt.loglog(dt_common, mean_weak, 's--', label="Mean Weak Error")
    plt.xlabel("Δt"); plt.ylabel("Error")
    plt.title("Aggregate EM Convergence (All Patients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("em_conv/aggregate_convergence.png", dpi=300)
    plt.close()
    plt.figure(figsize=(7,5))
    plt.loglog(dt_common, mean_runtime, 'd-', color='purple')
    plt.xlabel("Δt"); plt.ylabel("Runtime (s)")
    plt.title("Aggregate Runtime Scaling (All Patients)")
    plt.tight_layout()
    plt.savefig("em_conv/aggregate_runtime.png", dpi=300)
    plt.close()
    print("\n Aggregate convergence summary saved.")
else:
    print("\nNo valid convergence data collected to aggregate.")

#  summary
print("\nAll metrics computed successfully.")
print(metrics_results)
print("Total runtime: %.2f s" % (time.time() - start_time_total))
