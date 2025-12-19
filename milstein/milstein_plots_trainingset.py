import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from tqdm import tqdm
import time, os, re
import pandas as pd

# Timing
script_start_time = time.time()

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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((y_true - y_pred)**2) / (denom + 1e-12)

def kling_gupta_efficiency(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-12)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# Utility functions
dt_sim = 0.5  # INCREASED from 0.2 for much faster runtime
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def get_patient_data(patient_id, file_path="tumour_data.csv"):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ volumes file not found: {file_path}")
        return None, None

    pid_str = str(patient_id).zfill(3)
    df_patient = df[df["Patient"].astype(str).str.contains(pid_str, case=False, na=False)]

    if df_patient.empty:
        print("⚠️ No available data for patient", patient_id)
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
    v = np.clip(v, 1.0, 2e5)
    return t, v

# Functional parameters with max_val constraint
def a_of_t(t, a1, a2, alpha, t_c):
    result = a1 + a2 * np.tanh(np.clip(alpha * (t - t_c), -50, 50))
    return result

def b_of_t(t, b1, b2, beta, t_b, max_val=None):
    result = b1 + b2 * sigmoid(beta * (t - t_b))
    if max_val is not None:
        result = np.minimum(result, max_val)
    return result

# Milstein step with V_max constraint
def milstein_step_functional(V, t, dt, dW, params, V_max=None):
    eps = 1e-8
    V = max(V, eps)
    a1,a2,alpha,t_c, b1,b2,beta,t_b, c,h,k0, r, m = params
    
    # Apply V_max constraint to b(t)
    at = a_of_t(t, a1, a2, alpha, t_c)
    bt = b_of_t(t, b1, b2, beta, t_b, max_val=V_max)
    
    # Clip ratios more aggressively
    ratio1 = np.clip(bt / V, 1e-6, 1e4)  # Reduced from 1e6
    ratio2 = np.clip(V / m, 1e-6, 1e4)
    
    # Calculate drift with additional safety
    log1 = np.clip(np.log(ratio1), -10, 10)
    log2 = np.clip(np.log(ratio2), -10, 10)
    
    drift = at * V * log1 - k0 * np.exp(-r * t) * V + 0.1 * V * log2
    
    # Cap drift to prevent explosions
    if V_max is not None:
        max_drift = V_max * 0.3  # Limit to 30% of max per step
        drift = np.clip(drift, -V * 0.5, max_drift)
    
    sqrtV = np.sqrt(max(V, eps))
    denom = (h + sqrtV)
    g = c * V / denom
    dg_dV = c * (1.0 / denom - (V / (denom**2)) * (1.0 / (2.0 * max(sqrtV, eps))))
    
    # Milstein update
    V_new = V + drift * dt + g * dW + 0.5 * g * dg_dV * (dW**2 - dt)
    
    # Apply hard cap
    if V_max is not None:
        V_new = min(V_new, V_max)
    
    if not np.isfinite(V_new) or V_new > 1e6:
        V_new = min(max(V_new if np.isfinite(V_new) else V, eps), V_max if V_max else 1e6)
    
    return max(V_new, eps)

# Single & ensemble simulations with V_max
def simulate_tumor_milstein_functional(params, V0, t_grid, V_max=None, seed=None):
    rng = np.random.default_rng(seed)
    V = np.empty(len(t_grid))
    V[0] = float(V0)
    for i in range(1, len(t_grid)):
        dt = t_grid[i] - t_grid[i - 1]
        dW = np.sqrt(dt) * rng.standard_normal()
        V[i] = milstein_step_functional(V[i - 1], t_grid[i - 1], dt, dW, params, V_max=V_max)
    return V

def simulate_ensemble_milstein_functional(params, V0, t_grid, M=40, V_max=None, seed=None):
    rng = np.random.default_rng(seed)
    sims = np.empty((M, len(t_grid)))
    for j in range(M):
        sims[j] = simulate_tumor_milstein_functional(params, V0, t_grid, V_max=V_max, seed=int(rng.integers(0, 1<<30)))
    mean_traj = np.mean(sims, axis=0)
    std_traj = np.std(sims, axis=0)
    return mean_traj, std_traj, sims

# Objective with penalty terms and V_max (OPTIMIZED)
def functional_logmse_milstein(params, time_data, observed, V0, M=15, dt_sim=0.5, seed=42):  # Reduced M from 25 to 15
    try:
        a1,a2,alpha,t_c, b1,b2,beta,t_b, c, h, k0, r, m = params
    except Exception:
        return 1e30
    if b1 <= 0 or h <= 0 or c < 0 or alpha <= 0 or beta <= 0 or m <= 0:
        return 1e30
    
    # Set V_max based on observed data
    obs_max = np.max(observed)
    V_max = obs_max * 5.0
    
    # Quick parameter-space penalties (no simulation needed)
    penalty = 0.0
    
    # Penalize if b1 + b2 exceeds reasonable bounds (fast check)
    b_max_possible = b1 + abs(b2)
    if b_max_possible > 3 * obs_max:
        penalty += (b_max_possible - 3*obs_max)**2 * 1e-5
    
    # Penalize extreme growth rates
    if abs(a1) > 0.2 or abs(a2) > 0.4:
        penalty += (abs(a1) - 0.2)**2 * 10 + (abs(a2) - 0.4)**2 * 10
    
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, V_max=V_max, seed=seed)
    
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj > V_max):
        return 1e30
    
    # Simplified trajectory penalties (only if trajectory seems bad)
    max_traj = np.max(mean_traj)
    if max_traj > 2 * obs_max:
        penalty += (max_traj - 2*obs_max)**2 * 1e-6
    
    interp = np.interp(time_data, t_grid, mean_traj)
    base_error = np.sum((np.log(interp + 1e-6) - np.log(observed + 1e-6))**2)
    
    return base_error + penalty

def functional_residuals_milstein(params, time_data, observed, V0, M=12, dt_sim=0.5, seed=123):  # Reduced M and increased dt
    obs_max = np.max(observed)
    V_max = obs_max * 5.0
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, V_max=V_max, seed=seed)
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.log(interp + 1e-6) - np.log(observed + 1e-6)

# Fit function with tighter bounds (OPTIMIZED)
def fit_functional_milstein(time_data, observed, V0=None, seed=42, de_maxiter=3, M_de=12):  # Further reduced
    fit_info = {"fit_time": None, "de_message": None, "param_change": None, "ls_success": None}
    if time_data is None or observed is None or len(time_data) < 3 or len(observed) < 3:
        print("⚠️ Insufficient data → skipping fit.")
        return np.full(13, np.nan), fit_info
    if V0 is None:
        V0 = observed[0]
    obs_max = max(observed)
    if not np.isfinite(obs_max) or obs_max <= 0:
        print("⚠️ Invalid tumor data (obs_max).")
        return np.full(13, np.nan), fit_info

    # Tighter bounds to prevent blowup
    bounds = [
        (-0.3, 0.3), (-0.5, 0.5), (0.01, 1.5), (0.0, time_data[-1]),  # a params: reduced range
        (max(100.0, obs_max * 0.3), obs_max * 2.0),  # b1: tighter
        (0.0, obs_max * 2.0),  # b2: much tighter (was *10)
        (0.01, 1.5), (0.0, time_data[-1]),  # beta: reduced
        (0.0, 1.5), (1e-6, 1e5), (0.0, 0.6), (0.0, 0.3),  # c, h, k0, r: reduced
        (10.0, obs_max * 2.0)  # m: tighter
    ]

    print("Running Differential Evolution (global search)...")
    pbar = tqdm(total=de_maxiter, desc="DE", leave=False)
    def _cb(xk, convergence):
        pbar.update(1)
    de_start = time.time()
    try:
        result = differential_evolution(
            lambda p: functional_logmse_milstein(p, time_data, observed, V0, M=M_de, dt_sim=dt_sim, seed=seed),
            bounds, maxiter=de_maxiter, seed=seed, callback=_cb, polish=False, workers=1  # Disable parallelization overhead
        )
        pbar.close()
        fit_info["de_message"] = result.message
    except Exception as e:
        pbar.close()
        print("DE failed:", e)
        return np.full(13, np.nan), fit_info
    de_end = time.time()

    x0 = result.x.copy()

    print("Polishing with least_squares...")
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ls_start = time.time()
    try:
        ls = least_squares(
            lambda p: functional_residuals_milstein(p, time_data, observed, V0, M=10, dt_sim=dt_sim, seed=seed+1),  # Reduced from 12
            x0, bounds=(lower, upper), xtol=1e-5, ftol=1e-5, max_nfev=100  # Relaxed tolerances, reduced max_nfev
        )
    except Exception as e:
        print("Least-squares failed:", e)
        ls = None
    ls_end = time.time()

    if ls is None:
        fitted = x0
        fit_info["ls_success"] = False
    else:
        fitted = ls.x
        fit_info["ls_success"] = bool(ls.success)

    fit_info["fit_time"] = (de_end - de_start) + (ls_end - ls_start)
    try:
        fit_info["param_change"] = np.linalg.norm(x0 - fitted)
    except Exception:
        fit_info["param_change"] = np.nan

    if np.any(np.isclose(fitted, lower)) or np.any(np.isclose(fitted, upper)):
        print("⚠️ Warning: some fitted parameters hit bounds.")

    return fitted, fit_info

# Trajectory convergence helper
def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=60, tol=1e-4, window=5, seed=1234):
    """
    Simulate ensemble, compute ensemble mean over t_grid and find earliest time
    where mean stabilizes: relative change over 'window' consecutive steps below tol.
    Returns: convergence_time (float) or None if not converged.
    """
    t_grid = np.arange(0, T + dt, dt)
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, seed=seed)
    # compute relative changes
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    # we need window consecutive rel_changes < tol
    L = len(rel_changes)
    for i in range(L - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i+1+window-1]  # return time at end of window
    return None

# Milstein convergence test (strong & weak) using fine reference
def milstein_convergence_test(params, V0, T, M=200, dt_values=None, base_seed=123):
    """
    Compute strong and weak error vs dt using a fine reference integration.
    Returns dict with arrays for dt, strong_error, weak_error, runtime.
    """
    if dt_values is None:
        dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    # reference with reasonably small dt (not too crazy to keep runtime feasible)
    ref_dt = min(0.01, dt_values[-1]/2.0, 0.01)
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    # simulate reference ensemble (M realizations)
    mean_ref, _, sims_ref = simulate_ensemble_milstein_functional(params, V0, t_ref, M=M, seed=base_seed)
    ref_final = sims_ref[:, -1]
    mean_ref_final = np.mean(ref_final)
    results = {"dt_values": [], "strong_error": [], "weak_error": [], "runtime": []}
    for dt in dt_values:
        start = time.time()
        t_grid = np.arange(0, T + dt, dt)
        mean_dt, _, sims_dt = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, seed=base_seed)
        # strong: compare mean_dt to reference mean projected onto t_grid (L1)
        ref_interp = np.interp(t_grid, t_ref, np.mean(sims_ref, axis=0))
        strong_err = np.mean(np.abs(mean_dt - ref_interp))
        # weak: compare final means
        weak_err = np.abs(np.mean(sims_dt[:, -1]) - mean_ref_final)
        results["dt_values"].append(dt)
        results["strong_error"].append(strong_err)
        results["weak_error"].append(weak_err)
        results["runtime"].append(time.time() - start)
    # convert to np arrays
    for k in ["dt_values","strong_error","weak_error","runtime"]:
        results[k] = np.array(results[k])
    return results

# Plot saving helpers
def save_milstein_convergence_plots(conv, patient_id, outdir):
    os.makedirs(outdir, exist_ok=True)
    dt = conv["dt_values"]
    strong = conv["strong_error"]
    weak = conv["weak_error"]
    runtime = conv["runtime"]

    plt.figure(figsize=(6,4))
    plt.loglog(dt, strong, 'o-', label='Strong Error')
    plt.loglog(dt, weak, 's--', label='Weak Error')
    plt.xlabel("Δt"); plt.ylabel("Error")
    plt.legend()
    plt.title(f"Patient {patient_id} — Milstein Convergence")
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

# Directories & batch config
os.makedirs("milstein_conv/batches", exist_ok=True)
os.makedirs("milstein_traj/batches", exist_ok=True)
os.makedirs("milstein_conv/patients", exist_ok=True)
os.makedirs("milstein_traj/patients", exist_ok=True)

batch_size = 10
metrics_results = {}
all_means, all_stds, all_times, all_data, all_ids = [], [], [], [], []
batch_conv_store = []
batch_traj_store = []
batch_index = 1

# Main loop over patients 1..92
for pid in range(1, 93):
    print(f"\n=== Processing Patient {pid} ===")
    t0 = time.time()
    time_data, tum = get_patient_data(pid)
    if time_data is None or tum is None:
        continue
    time_data, tumor_volume_data = sanitize_tumor_data(time_data, tum)
    if len(tumor_volume_data) < 3:
        print(" Skipping — insufficient valid data.")
        continue
    V0 = tumor_volume_data[0]
    # Fit model (DE + LS)
    fitted, fit_info = fit_functional_milstein(time_data, tumor_volume_data, V0=V0, seed=pid, de_maxiter=6, M_de=20)
    print("Fitting info:", fit_info)
    if np.all(np.isnan(fitted)):
        print(" Fit returned NaNs, skip convergence & plotting for this patient.")
        metrics_results[pid] = {"MASE": np.nan, "Chi2": np.nan, "NSE": np.nan, "KGE": np.nan, "FitInfo": fit_info, "Convergence": None}
        continue
    # simulate final with decent ensemble for plotting/metrics
    t_fine = np.linspace(time_data[0], time_data[-1], 500)
    mean_fit, std_fit, trajs = simulate_ensemble_milstein_functional(fitted, V0, t_fine, M=60, seed=pid)

    # clip only for plotting (do not change the real arrays used for metrics)
    clip_upper = np.percentile(mean_fit, 99.5)
    mean_fit_plot = np.clip(mean_fit, 0, clip_upper)
    std_fit_plot = np.clip(std_fit, 0, clip_upper)

    # metrics
    y_pred_interp = np.interp(time_data, t_fine, mean_fit)
    mase = mean_absolute_scaled_error(tumor_volume_data, y_pred_interp)
    chi2 = chi_squared(tumor_volume_data, y_pred_interp)
    nse = nash_sutcliffe_efficiency(tumor_volume_data, y_pred_interp)
    kge = kling_gupta_efficiency(tumor_volume_data, y_pred_interp)
    metrics_results[pid] = {"MASE": mase, "Chi2": chi2, "NSE": nse, "KGE": kge, "FitInfo": fit_info}

    print(f"Metrics → MASE={mase:.4f}, Chi²={chi2:.4f}, NSE={nse:.4f}, KGE={kge:.4f}")

    #Trajectory convergence time (ensemble mean stabilization) 
    try:
        traj_conv_time = trajectory_convergence_time_from_ensemble(fitted, V0, T=time_data[-1], dt=0.2, M=40, tol=1e-4, window=5, seed=pid)
    except Exception as e:
        traj_conv_time = None
        print("Trajectory convergence check failed:", e)
    metrics_results[pid]["TrajectoryConvergenceTime"] = traj_conv_time
    print("Trajectory convergence time:", traj_conv_time)

    metrics_results[pid]["ParamChange"] = fit_info.get("param_change")
    metrics_results[pid]["FitTime"] = fit_info.get("fit_time")

    try:
        # small M to keep runtime reasonable; you can increase M for more faithful results.
        conv = milstein_convergence_test(fitted, V0, T=time_data[-1], M=120)
        metrics_results[pid]["Convergence"] = conv
        # Save per-patient convergence figures into patient folder
        save_milstein_convergence_plots(conv, pid, outdir="milstein_conv/patients")
    except Exception as e:
        print("Convergence test failed:", e)
        metrics_results[pid]["Convergence"] = None
    # Save per-patient trajectory plot (small)
    try:
        plt.figure(figsize=(8,4.5))
        plt.plot(t_fine, mean_fit, '-', label="Mean fit")
        plt.fill_between(t_fine, mean_fit - 1.96*std_fit, mean_fit + 1.96*std_fit, alpha=0.15, label="95% CI")
        plt.scatter(time_data, tumor_volume_data, color='k', s=10, label="Data")
        plt.xlabel("Time (weeks)")
        plt.ylabel("Tumor Volume (mm³)")
        plt.title(f"Patient {pid} — Functional Milstein Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"milstein_traj/patients/patient_{pid:03d}_traj.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Per-patient trajectory save failed:", e)

    # Append to batch stores (for batch-of-10 saving)
    batch_conv_store.append((pid, conv))
    batch_traj_store.append((pid, t_fine, mean_fit_plot, std_fit_plot, time_data, tumor_volume_data))
    all_means.append(mean_fit_plot); all_stds.append(std_fit_plot); all_times.append(t_fine); all_data.append((time_data, tumor_volume_data)); all_ids.append(pid)
    # When we have a full batch, save aggregated plots (one convergence image + one trajectories image)
    if len(batch_conv_store) == batch_size:
        # Save batch convergence overlay in milstein_conv/batches
        plt.figure(figsize=(8,6))
        for (p_i, conv_i) in batch_conv_store:
            plt.loglog(conv_i["dt_values"], conv_i["strong_error"], marker='o', linestyle='-', alpha=0.8, label=f"P{p_i} strong")
            plt.loglog(conv_i["dt_values"], conv_i["weak_error"], marker='s', linestyle='--', alpha=0.6, label=f"P{p_i} weak")
        plt.xlabel("Δt"); plt.ylabel("Error")
        plt.title(f"Batch {batch_index}: Milstein Convergence (errors)")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(f"milstein_conv/batches/batch_{batch_index:03d}_convergence.png", dpi=300)
        plt.close()

        # Save batch trajectories in milstein_traj/batches
        cmap = plt.cm.tab10(np.linspace(0, 1, batch_size))
        plt.figure(figsize=(10,5))
        for j, color in enumerate(cmap):
            pid_j, t_j, mean_j, std_j, td_j, tv_j = batch_traj_store[j]
            plt.plot(t_j, mean_j, color=color, label=f"P{pid_j}")
            plt.fill_between(t_j, mean_j - 1.96*std_j, mean_j + 1.96*std_j, color=color, alpha=0.15)
            plt.scatter(td_j, tv_j, color=color, s=12, alpha=0.7)
        plt.axvspan(3, 21, color='gray', alpha=0.2, label='Chemo Window')
        plt.xlabel("Time (weeks)"); plt.ylabel("Tumor Volume (mm³)")
        plt.title(f"Batch {batch_index}: Functional Milstein Fits")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"milstein_traj/batches/batch_{batch_index:03d}_traj.png", dpi=300)
        plt.close()

        print(f" Saved batch {batch_index} convergence + trajectory plots.")
        batch_conv_store, batch_traj_store = [], []
        batch_index += 1
        # free memory for batch arrays
        all_means, all_stds, all_times, all_data, all_ids = [], [], [], [], []

    # quick per-patient runtime print
    print(f"Patient {pid} done — runtime: {time.time() - t0:.2f}s (total elapsed: {time.time() - script_start_time:.1f}s)")

# Aggregate convergence summary (if any)
dt_common = np.array([0.8, 0.4, 0.2, 0.1, 0.05])
strong_all, weak_all, runtime_all = [], [], []
for pid, entry in metrics_results.items():
    conv = entry.get("Convergence")
    if conv is None:
        continue
    strong_all.append(conv["strong_error"])
    weak_all.append(conv["weak_error"])
    runtime_all.append(conv["runtime"])
if strong_all:
    strong_all = np.vstack(strong_all)
    weak_all = np.vstack(weak_all)
    runtime_all = np.vstack(runtime_all)
    mean_strong = np.mean(strong_all, axis=0)
    mean_weak = np.mean(weak_all, axis=0)
    mean_runtime = np.mean(runtime_all, axis=0)

    plt.figure(figsize=(7,5))
    plt.loglog(dt_common, mean_strong, 'o-', label="Mean Strong Error")
    plt.loglog(dt_common, mean_weak, 's--', label="Mean Weak Error")
    plt.xlabel("Δt"); plt.ylabel("Error")
    plt.title("Aggregate Milstein Convergence (All Patients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("milstein_conv/aggregate_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.loglog(dt_common, mean_runtime, 'd-', color='purple')
    plt.xlabel("Δt"); plt.ylabel("Runtime (s)")
    plt.title("Aggregate Runtime Scaling (All Patients)")
    plt.tight_layout()
    plt.savefig("milstein_conv/aggregate_runtime.png", dpi=300)
    plt.close()

    print("\nAggregate convergence summary saved.")
else:
    print("\n⚠️ No valid convergence data collected to aggregate.")

# Final summary print (first 10 patients)
print("\n All done. Metrics summary (all patients):")
for k in list(metrics_results.keys()):
    print(k, metrics_results[k])
print("Total script runtime: %.2f s" % (time.time() - script_start_time))

metrics_results = {}  # Make sure this dict was filled during your per-patient loop

metrics_rows = []
for pid, data in metrics_results.items():
    row = {
        "Patient": pid,
        "MASE": data.get("MASE"),
        "Chi2": data.get("Chi2"),
        "NSE": data.get("NSE"),
        "KGE": data.get("KGE"),
        "FitTime": data.get("FitTime"),
        "ParamChange": data.get("ParamChange"),
        "TrajectoryConvergenceTime": data.get("TrajectoryConvergenceTime"),
    }

    # Milstein convergence or trajectory info (if available)
    conv = data.get("Convergence")
    if conv is not None:
        row["Conv_dt"] = "|".join(map(str, conv["dt_values"]))
        row["StrongError"] = "|".join(map(str, conv["strong_error"]))
        row["WeakError"] = "|".join(map(str, conv["weak_error"]))
        row["Runtime"] = "|".join(map(str, conv["runtime"]))
    else:
        row["Conv_dt"] = None
        row["StrongError"] = None
        row["WeakError"] = None
        row["Runtime"] = None

    metrics_rows.append(row)

df_metrics = pd.DataFrame(metrics_rows)
df_metrics.to_csv("milstein_results.csv", index=False)

print("\n Saved Milstein results → milstein_results.csv")
