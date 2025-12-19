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
dt_sim = 0.5
def sigmoid(x): # Sigmoid function
    return 1.0 / (1.0 + np.exp(-x))

def get_patient_data(patient_id, file_path="tumour_data.csv"):
    """
    Replacement for the old get_patient_data() that reads from a CSV instead of volumes.txt.
    Keeps the exact same behavior and return structure: (times, volumes)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ volumes file not found: {file_path}")
        return None, None
    #  patient ID is formatted like "Patient 001" or similar
    pid_str = str(patient_id).zfill(3)
    df_patient = df[df["Patient"].astype(str).str.contains(pid_str, case=False, na=False)]
    if df_patient.empty:
        print("⚠️ No available data for patient", patient_id)
        return None, None
    #  week numbers (handles strings like 'week-000-1', 'week-040-2', etc.)
    weeks = []
    for w in df_patient["Week"]:
        match = re.search(r"(\d+)", str(w))
        if match:
            weeks.append(float(match.group(1)))
    weeks = np.array(weeks, dtype=float)
    #  total tumour volume
    volumes = df_patient["Total_mm3"].astype(float).to_numpy()
    # Sort both arrays by week (just in case)
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
    #  zeros or negatives :
    if np.any(v <= 0):
        nonzero_idx = np.where(v > 0)[0]
        zero_idx = np.where(v <= 0)[0]
        if len(nonzero_idx) > 1:
            # Interpolate using indices (avoids weird time gaps)
            v[zero_idx] = np.interp(zero_idx, nonzero_idx, v[nonzero_idx])
        else:
            v = np.maximum(v, 10.0)
    v = np.clip(v, 1.0, 1e7)
    return t, v

#  parameters
def a_of_t(t, a1, a2, alpha, t_c):
    return a1 + a2 * np.tanh(alpha * (t - t_c))
def b_of_t(t, b1, b2, beta, t_b):
    return b1 + b2 * sigmoid(beta * (t - t_b))

def srk4_step_functional(V, t, dt, dW, params):
    """
    4-stage stochastic Runge-Kutta step for SDE:
      dV = f(V,t) dt + g(V,t) dW
    params contains the model parameters used in f and g ( 13).
    """
    eps = 1e-12
    V = max(V, eps)
    a1,a2,alpha,t_c, b1,b2,beta,t_b, c, h, k0, r, m = params
    # drift 
    def f(Vloc, tloc):
        at = a_of_t(tloc, a1, a2, alpha, t_c)
        bt = b_of_t(tloc, b1, b2, beta, t_b)
        return at * Vloc * np.log(max(bt / Vloc, 1e-12)) - k0 * np.exp(-r * tloc) * Vloc + 0.1 * Vloc * np.log(max(Vloc / m, 1e-12))
    # diffusion 
    def g(Vloc, tloc):
        sqrtV = np.sqrt(max(Vloc, eps))
        return c * Vloc / (h + sqrtV)
    # dg/dV
    def dg_dV(Vloc, tloc):
        sqrtV = np.sqrt(max(Vloc, eps))
        denom = h + sqrtV
        return c * (1.0 / denom - (Vloc / (denom**2)) * (1.0 / (2.0 * max(sqrtV, eps))))
    # simple strong-order-1.0 SRK-style integrator using 4 stages:
    # stage times
    c1, c2, c3, c4 = 0.0, 0.5, 0.5, 1.0
    #  RK weights 
    aRK = np.array([0.0, 0.5, 0.5, 1.0])
    bRK = np.array([1/6, 1/3, 1/3, 1/6])
    # stage evaluations K (drift) and L (diffusion)  then combine
    K = np.zeros(4)
    L = np.zeros(4)
    Vhat = np.zeros(4)
    # need independent normal increments for internal stochastic stages
    # decomposition  dW into correlated pieces:  two independent normals U1,U2 and construct increments
    # Stage 1
    Vhat[0] = V
    K[0] = f(Vhat[0], t + c1*dt)
    L[0] = g(Vhat[0], t + c1*dt)
    # Stage 2
    Vhat[1] = V + dt * (0.5 * K[0]) + dW * (0.5 * L[0])
    K[1] = f(Vhat[1], t + c2*dt)
    L[1] = g(Vhat[1], t + c2*dt)
    # Stage 3
    Vhat[2] = V + dt * (0.5 * K[1]) + dW * (0.5 * L[1])
    K[2] = f(Vhat[2], t + c3*dt)
    L[2] = g(Vhat[2], t + c3*dt)
    # Stage 4
    Vhat[3] = V + dt * K[2] + dW * L[2]
    K[3] = f(Vhat[3], t + c4*dt)
    L[3] = g(Vhat[3], t + c4*dt)
    V_new = V + dt * np.dot(bRK, K) + dW * np.dot(bRK, L)
    # Add small correction for  noise (improves strong order)
    # 0.5 * g * dg_dV * (dW**2 - dt) as correction (similar to Milstein) increase accurac
    g_mid = g(V, t)
    dg_mid = dg_dV(V, t)
    V_new += 0.5 * g_mid * dg_mid * (dW**2 - dt)
    V_new = np.clip(V_new, 1e-6, 1e6)
    #  clamp
    if not np.isfinite(V_new) or V_new > 1e8:
        V_new = min(max(V_new if np.isfinite(V_new) else eps, eps), 1e5)
    return max(V_new, eps)

# Single & ensemble SRK simulation (functional parameters)
def simulate_tumor_srk_functional(params, V0, t_grid, seed=None):
    '''Simulates a single trajectory of the SRK SDE with functional parameters.'''
    rng = np.random.default_rng(seed)
    V = np.empty(len(t_grid))
    V[0] = float(V0)
    for i in range(1, len(t_grid)):
        dt = t_grid[i] - t_grid[i - 1]
        dW = np.sqrt(dt) * rng.standard_normal()
        V[i] = srk4_step_functional(V[i - 1], t_grid[i - 1], dt, dW, params)
    return V

def simulate_ensemble_srk_functional(params, V0, t_grid, M=40, seed=None):
    '''Simulates an ensemble of M trajectories of the SRK SDE with functional parameters.'''
    rng = np.random.default_rng(seed)
    sims = np.empty((M, len(t_grid)))
    for j in range(M):
        sims[j] = simulate_tumor_srk_functional(params, V0, t_grid, seed=int(rng.integers(0, 1<<30)))
    mean_traj = np.mean(sims, axis=0)
    std_traj = np.std(sims, axis=0)
    return mean_traj, std_traj, sims

# Objective & Residuals (log)
def functional_logmse_srk(params, time_data, observed, V0, M=25, dt_sim=0.5, seed=42):
    '''Computes the log-MSE between observed data and model predictions for given parameters.'''
    try:
        a1,a2,alpha,t_c, b1,b2,beta,t_b, c, h, k0, r, m = params
    except Exception:
        return 1e30
    if b1 <= 0 or h <= 0 or c < 0 or alpha <= 0 or beta <= 0 or m <= 0:
        return 1e30
    # choose at least 40 steps across interval
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=seed)
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj > 1e8):
        return 1e30
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.sum((np.log(interp + 1e-6) - np.log(observed + 1e-6))**2)

def functional_residuals_srk(params, time_data, observed, V0, M=20, dt_sim=0.5, seed=123):
    '''Computes the residuals (log space) between observed data and model predictions for given parameters.'''
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=seed)
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.log(interp + 1e-6) - np.log(observed + 1e-6)

# Fit function: DE + least_squares polish
def fit_functional_srk(time_data, observed, V0=None, seed=42, de_maxiter=6, M_de=20):
    """
    Returns: fitted_params (len=13), fit_info dict with timing & convergence measures
    """
    fit_info = {"fit_time": None, "de_message": None, "param_change": None, "ls_success": None}
    if time_data is None or observed is None or len(time_data) < 3 or len(observed) < 3:
        print(" Insufficient data → skipping fit.")
        return np.full(13, np.nan), fit_info
    if V0 is None:
        V0 = observed[0]
    obs_max = max(observed)
    if not np.isfinite(obs_max) or obs_max <= 0:
        print("⚠️ Invalid tumor data (obs_max).")
        return np.full(13, np.nan), fit_info
    # bounds for 13 params: a1,a2,alpha,t_c, b1,b2,beta,t_b, c,h,k0, r, m
    bounds = [
    (-0.5, 0.5), (-1.0, 1.0), (0.01, 2.0), (0.0, time_data[-1]),
    (max(1.0, obs_max * 0.2), obs_max * 5 + 1e-3),
    (0.0, obs_max * 10 + 1e-3), (0.01, 2.0), (0.0, time_data[-1]),
    (0.0, 2.0), (1e-3, 1e5), (0.0, 0.8), (0.0, 0.8), (10.0, obs_max * 10 + 1e-3)
]
    # Differential evolution (global)
    print("Running Differential Evolution (global search)...")
    pbar = tqdm(total=de_maxiter, desc="DE", leave=False)
    def _cb(xk, convergence):
        pbar.update(1)
    de_start = time.time()
    try:
        result = differential_evolution(lambda p: functional_logmse_srk(p, time_data, observed, V0, M=M_de, dt_sim=dt_sim, seed=seed),
                                        bounds, maxiter=de_maxiter, seed=seed, callback=_cb, polish=False)
        pbar.close()
        fit_info["de_message"] = result.message
    except Exception as e:
        pbar.close()
        print("DE failed:", e)
        return np.full(13, np.nan), fit_info
    de_end = time.time()
    x0 = result.x.copy()
    #   least_squares
    print("Polishing with least_squares...")
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ls_start = time.time()
    try:
        ls = least_squares(lambda p: functional_residuals_srk(p, time_data, observed, V0, M=18, dt_sim=dt_sim, seed=seed+1),
                           x0, bounds=(lower, upper), xtol=1e-6, ftol=1e-6, max_nfev=200)
    except Exception as e:
        print("Least-squares failed:", e)
        ls = None
    ls_end = time.time()
#  process results
    if ls is None:
        fitted = x0
        fit_info["ls_success"] = False
    else:
        fitted = ls.x
        fit_info["ls_success"] = bool(ls.success)
# total time
    fit_info["fit_time"] = (de_end - de_start) + (ls_end - ls_start)
    # param change: norm(x0 - fitted)
    try:
        fit_info["param_change"] = np.linalg.norm(x0 - fitted)
    except Exception:
        fit_info["param_change"] = np.nan
# check for bounds hits
    if np.any(np.isclose(fitted, lower)) or np.any(np.isclose(fitted, upper)):
        print("⚠️ Warning: some fitted parameters hit bounds.")
    return fitted, fit_info

#  convergence helper
def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=60, tol=1e-4, window=5, seed=1234):
    """
    Simulate ensemble, compute ensemble mean over t_grid and find earliest time
    where mean stabilizes: relative change over 'window' consecutive steps below tol.
    Returns: convergence_time (float) or None if not converged.
    """
    t_grid = np.arange(0, T + dt, dt)
    mean_traj, _, _ = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=seed)
    # compute relative changes
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    # we need window consecutive rel_changes < tol
    L = len(rel_changes)
    for i in range(L - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i+1+window-1]  # return time at end of window
    return None

#  convergence test (strong & weak
def srk_convergence_test(params, V0, T, M=200, dt_values=None, base_seed=123):
    """
    Compute strong and weak error vs dt using a fine reference integration.
    Returns dict with arrays for dt, strong_error, weak_error, runtime.
    """
    if dt_values is None:
        dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    # reference with reasonably small dt (not too crazy to keep runtime feasible)
    ref_dt = min(0.01, dt_values[-1] / 2.0)
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    # simulate reference ensemble (M realizations)
    mean_ref, _, sims_ref = simulate_ensemble_srk_functional(params, V0, t_ref, M=M, seed=base_seed)
    ref_final = sims_ref[:, -1]
    mean_ref_final = np.mean(ref_final)
    results = {"dt_values": [], "strong_error": [], "weak_error": [], "runtime": []}
    for dt in dt_values:
        start = time.time()
        t_grid = np.arange(0, T + dt, dt)
        mean_dt, _, sims_dt = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=base_seed)
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
def save_srk_convergence_plots(conv, patient_id, outdir):
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
    plt.title(f"Patient {patient_id} — SRK Convergence")
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
# Directories + batch config
os.makedirs("srk_conv/batches", exist_ok=True)
os.makedirs("srk_traj/batches", exist_ok=True)
os.makedirs("srk_conv/patients", exist_ok=True)
os.makedirs("srk_traj/patients", exist_ok=True)
batch_size = 10
metrics_results = {}
all_means, all_stds, all_times, all_data, all_ids = [], [], [], [], []
batch_conv_store = []
batch_traj_store = []
batch_index = 1
# Main loop over patients
for pid in range(1, 93):
    print(f"\n=== Processing Patient {pid} ===")
    t0 = time.time()
    time_data, tum = get_patient_data(pid)
    if time_data is None or tum is None:
        continue
    time_data, tumor_volume_data = sanitize_tumor_data(time_data, tum)
    if len(tumor_volume_data) < 3:
        print("⚠️ Skipping — insufficient valid data.")
        continue
    V0 = tumor_volume_data[0]
    # Fit model (DE + LS)
    fitted, fit_info = fit_functional_srk(time_data, tumor_volume_data, V0=V0, seed=pid, de_maxiter=6, M_de=20)
    print("Fitting info:", fit_info)
    if np.all(np.isnan(fitted)):
        print("⚠️ Fit returned NaNs — skip convergence & plotting for this patient.")
        metrics_results[pid] = {"MASE": np.nan, "Chi2": np.nan, "NSE": np.nan, "KGE": np.nan, "FitInfo": fit_info, "Convergence": None}
        continue
    # simulate final with decent ensemble for plotting/metrics
    t_fine = np.linspace(time_data[0], time_data[-1], 500)
    mean_fit, std_fit, trajs = simulate_ensemble_srk_functional(fitted, V0, t_fine, M=60, seed=pid)
    # clip only for plotting (do not change the real arrays used for metrics)
    clip_upper = min(np.percentile(mean_fit, 99.5), 1e5)
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
        traj_conv_time = trajectory_convergence_time_from_ensemble(fitted, V0, T=time_data[-1], dt=0.5, M=40, tol=1e-4, window=5, seed=pid)
    except Exception as e:
        traj_conv_time = None
        print("Trajectory convergence check failed:", e)
    metrics_results[pid]["TrajectoryConvergenceTime"] = traj_conv_time
    print("Trajectory convergence time:", traj_conv_time)
    # Parameter convergence ( already stored param_change in fit_info) 
    metrics_results[pid]["ParamChange"] = fit_info.get("param_change")
    metrics_results[pid]["FitTime"] = fit_info.get("fit_time")
    # SRK convergence test (strong & weak) 
    try:
        # moderate M to keep runtime reasonable
        conv = srk_convergence_test(fitted, V0, T=time_data[-1], M=120)
        metrics_results[pid]["Convergence"] = conv
        # per-patient convergence figures into patient folder
        save_srk_convergence_plots(conv, pid, outdir="srk_conv/patients")
    except Exception as e:
        print("Convergence test failed:", e)
        metrics_results[pid]["Convergence"] = None
    #  per-patient trajectory plot (small)
    try:
        plt.figure(figsize=(8,4.5))
        plt.plot(t_fine, mean_fit, '-', label="Mean fit")
        plt.fill_between(t_fine, mean_fit - 1.96*std_fit, mean_fit + 1.96*std_fit, alpha=0.15, label="95% CI")
        plt.scatter(time_data, tumor_volume_data, color='k', s=10, label="Data")
        plt.xlabel("Time (weeks)")
        plt.ylabel("Tumor Volume (mm³)")
        plt.title(f"Patient {pid} — Functional SRK Fit")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"srk_traj/patients/patient_{pid:03d}_traj.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Per-patient trajectory save failed:", e)
    # Append to batch stores (for batch-of-10 saving)
    batch_conv_store.append((pid, conv))
    batch_traj_store.append((pid, t_fine, mean_fit_plot, std_fit_plot, time_data, tumor_volume_data))
    all_means.append(mean_fit_plot); all_stds.append(std_fit_plot); all_times.append(t_fine); all_data.append((time_data, tumor_volume_data)); all_ids.append(pid)
    # When we have a full batch, save aggregated plots (one convergence image + one trajectories image)
    if len(batch_conv_store) == batch_size:
        # Save batch convergence overlay in srk_conv/batches
        plt.figure(figsize=(8,6))
        for (p_i, conv_i) in batch_conv_store:
            plt.loglog(conv_i["dt_values"], conv_i["strong_error"], marker='o', linestyle='-', alpha=0.8, label=f"P{p_i} strong")
            plt.loglog(conv_i["dt_values"], conv_i["weak_error"], marker='s', linestyle='--', alpha=0.6, label=f"P{p_i} weak")
        plt.xlabel("Δt"); plt.ylabel("Error")
        plt.title(f"Batch {batch_index}: SRK Convergence (errors)")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(f"srk_conv/batches/batch_{batch_index:03d}_convergence.png", dpi=300)
        plt.close()
        # Save batch trajectories in srk_traj/batches
        cmap = plt.cm.tab10(np.linspace(0, 1, batch_size))
        plt.figure(figsize=(10,5))
        for j, color in enumerate(cmap):
            pid_j, t_j, mean_j, std_j, td_j, tv_j = batch_traj_store[j]
            plt.plot(t_j, mean_j, color=color, label=f"P{pid_j}")
            plt.fill_between(t_j, mean_j - 1.96*std_j, mean_j + 1.96*std_j, color=color, alpha=0.15)
            plt.scatter(td_j, tv_j, color=color, s=12, alpha=0.7)
        plt.axvspan(3, 21, color='gray', alpha=0.2, label='Chemo Window')
        plt.xlabel("Time (weeks)"); plt.ylabel("Tumor Volume (mm³)")
        plt.title(f"Batch {batch_index}: Functional SRK Fits")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f"srk_traj/batches/batch_{batch_index:03d}_traj.png", dpi=300)
        plt.close()
        print(f" Saved batch {batch_index} convergence + trajectory plots.")
        batch_conv_store, batch_traj_store = [], []
        batch_index += 1
        # free memory for batch arrays
        all_means, all_stds, all_times, all_data, all_ids = [], [], [], [], []
    # quick per-patient runtime print
    print(f"Patient {pid} done — runtime: {time.time() - t0:.2f}s (total elapsed: {time.time() - script_start_time:.1f}s)")

#  convergence summary (if any)
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
    plt.title("Aggregate SRK Convergence (All Patients)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("srk_conv/aggregate_convergence.png", dpi=300)
    plt.close()
    plt.figure(figsize=(7,5))
    plt.loglog(dt_common, mean_runtime, 'd-', color='purple')
    plt.xlabel("Δt"); plt.ylabel("Runtime (s)")
    plt.title("Aggregate Runtime Scaling (All Patients)")
    plt.tight_layout()
    plt.savefig("srk_conv/aggregate_runtime.png", dpi=300)
    plt.close()
    print("\n Aggregate convergence summary saved.")
else:
    print("\n⚠️ No valid convergence data collected to aggregate.")
# Final summary print (first 10 patients)
print("\n All done. Metrics summary (all patients):")
for k in list(metrics_results.keys())[:10]:
    print(k, metrics_results[k])
print("Total script runtime: %.2f s" % (time.time() - script_start_time))
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
    # SRK convergence (strong / weak)
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
df_metrics.to_csv("srk_results.csv", index=False)
print("\nSaved SRK results → srk_results.csv")
