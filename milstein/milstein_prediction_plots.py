import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from tqdm import tqdm
import time, os, re
import pandas as pd
from typing import List, Dict, Any, Tuple
from math import isfinite
from collections import defaultdict 
from matplotlib import cm 

COLOR_SEEN = '#3293a8'      
COLOR_UNSEEN = '#a83e32'    
COLOR_LINE = '#3244a8'      
BATCH_COLORMAP = cm.tab10 

TESTING_PATIENTS: List[int] = [31, 19, 43, 54, 77, 73, 72, 71, 67, 52]

os.makedirs("milstein_predictive_plots", exist_ok=True)
script_start_time = time.time()

def mean_absolute_scaled_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2: return np.nan
    naive_forecast = np.mean(np.abs(np.diff(y_true)))
    return np.mean(np.abs(y_true - y_pred)) / (naive_forecast + 1e-12)

def chi_squared(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(((y_true - y_pred) ** 2) / (y_pred + 1e-12))

def nash_sutcliffe_efficiency(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum((y_true - np.mean(y_true))**2)
    if denom == 0: return np.nan
    return 1 - np.sum((y_true - y_pred)**2) / (denom + 1e-12)

def kling_gupta_efficiency(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0: return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-12)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

dt_sim = 0.5  
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

def a_of_t(t, a1, a2, alpha, t_c):
    result = a1 + a2 * np.tanh(np.clip(alpha * (t - t_c), -50, 50))
    return result

def b_of_t(t, b1, b2, beta, t_b, max_val=None):
    result = b1 + b2 * sigmoid(beta * (t - t_b))
    if max_val is not None:
        result = np.minimum(result, max_val)
    return result

def milstein_step_functional(V, t, dt, dW, params, V_max=None):
    eps = 1e-8
    V = max(V, eps)
    a1,a2,alpha,t_c, b1,b2,beta,t_b, c,h,k0, r, m = params
    
    #  V_max constraint to b(t)
    at = a_of_t(t, a1, a2, alpha, t_c)
    bt = b_of_t(t, b1, b2, beta, t_b, max_val=V_max)
    
    # Clip ratios more aggressively
    ratio1 = np.clip(bt / V, 1e-6, 1e4)  
    ratio2 = np.clip(V / m, 1e-6, 1e4)
    
    #  drift with additional safety
    log1 = np.clip(np.log(ratio1), -10, 10)
    log2 = np.clip(np.log(ratio2), -10, 10)
    
    drift = at * V * log1 - k0 * np.exp(-r * t) * V + 0.1 * V * log2
    
    # Cap drift to prevent explosions
    if V_max is not None:
        max_drift = V_max * 0.3  
        drift = np.clip(drift, -V * 0.5, max_drift)
    
    sqrtV = np.sqrt(max(V, eps))
    denom = (h + sqrtV)
    g = c * V / denom
    dg_dV = c * (1.0 / denom - (V / (denom**2)) * (1.0 / (2.0 * max(sqrtV, eps))))
    
    V_new = V + drift * dt + g * dW + 0.5 * g * dg_dV * (dW**2 - dt)
    
    #  hard cap
    if V_max is not None:
        V_new = min(V_new, V_max)
    
    if not np.isfinite(V_new) or V_new > 1e6:
        V_new = min(max(V_new if np.isfinite(V_new) else V, eps), V_max if V_max else 1e6)
    
    return max(V_new, eps)

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

def functional_logmse_milstein(params, time_data, observed, V0, M=15, dt_sim=0.5, seed=42):
    try:
        a1,a2,alpha,t_c, b1,b2,beta,t_b, c, h, k0, r, m = params
    except Exception: return 1e30
    if b1 <= 0 or h <= 0 or c < 0 or alpha <= 0 or beta <= 0 or m <= 0: return 1e30
    
    obs_max = np.max(observed)
    V_max = obs_max * 5.0
    
    penalty = 0.0
    
    b_max_possible = b1 + abs(b2)
    if b_max_possible > 3 * obs_max:
        penalty += (b_max_possible - 3*obs_max)**2 * 1e-5
    
    if abs(a1) > 0.2 or abs(a2) > 0.4:
        penalty += (abs(a1) - 0.2)**2 * 10 + (abs(a2) - 0.4)**2 * 10
    
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, V_max=V_max, seed=seed)
    
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj > V_max):
        return 1e30
    
    max_traj = np.max(mean_traj)
    if max_traj > 2 * obs_max:
        penalty += (max_traj - 2*obs_max)**2 * 1e-6
    
    interp = np.interp(time_data, t_grid, mean_traj)
    base_error = np.sum((np.log(interp + 1e-6) - np.log(observed + 1e-6))**2)
    
    return base_error + penalty

def functional_residuals_milstein(params, time_data, observed, V0, M=12, dt_sim=0.5, seed=123):
    obs_max = np.max(observed)
    V_max = obs_max * 5.0
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, V_max=V_max, seed=seed)
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.log(interp + 1e-6) - np.log(observed + 1e-6)

def fit_functional_milstein(time_data, observed, V0=None, seed=42, de_maxiter=3, M_de=12):
    fit_info = {"fit_time": None, "de_message": None, "param_change": None, "ls_success": None}
    if time_data is None or observed is None or len(time_data) < 3 or len(observed) < 3:
        print("⚠️ Insufficient data → skipping fit."); return np.full(13, np.nan), fit_info
    if V0 is None: V0 = observed[0]
    obs_max = max(observed)
    if not np.isfinite(obs_max) or obs_max <= 0:
        print("⚠️ Invalid tumor data (obs_max)."); return np.full(13, np.nan), fit_info

    bounds = [
        (-0.3, 0.3), (-0.5, 0.5), (0.01, 1.5), (0.0, time_data[-1]),  # a params: reduced range
        (max(100.0, obs_max * 0.3), obs_max * 2.0),  # b1: tighter
        (0.0, obs_max * 2.0),  # b2: much tighter (was *10)
        (0.01, 1.5), (0.0, time_data[-1]),  # beta: reduced
        (0.0, 1.5), (1e-6, 1e5), (0.0, 0.6), (0.0, 0.3),  # c, h, k0, r: reduced
        (10.0, obs_max * 2.0)  # m: tighter
    ]

    print("Running Differential Evolution (global search)...")
    pbar = tqdm(total=de_maxiter, desc="DE", leave=False, disable=True)
    def _cb(xk, convergence): pbar.update(1)
    de_start = time.time()
    try:
        result = differential_evolution(
            lambda p: functional_logmse_milstein(p, time_data, observed, V0, M=M_de, dt_sim=dt_sim, seed=seed),
            bounds, maxiter=de_maxiter, seed=seed, callback=_cb, polish=False, workers=1
        )
        pbar.close(); fit_info["de_message"] = result.message
    except Exception as e:
        pbar.close(); print("DE failed:", e); return np.full(13, np.nan), fit_info
    de_end = time.time()

    x0 = result.x.copy()

    print("Polishing with least_squares...")
    lower = np.array([b[0] for b in bounds]); upper = np.array([b[1] for b in bounds])
    ls_start = time.time()
    try:
        ls = least_squares(
            lambda p: functional_residuals_milstein(p, time_data, observed, V0, M=10, dt_sim=dt_sim, seed=seed+1),
            x0, bounds=(lower, upper), xtol=1e-5, ftol=1e-5, max_nfev=100
        )
    except Exception as e:
        print("Least-squares failed:", e); ls = None
    ls_end = time.time()

    if ls is None: fitted = x0; fit_info["ls_success"] = False
    else: fitted = ls.x; fit_info["ls_success"] = bool(ls.success)

    fit_info["fit_time"] = (de_end - de_start) + (ls_end - ls_start)
    try: fit_info["param_change"] = np.linalg.norm(x0 - fitted)
    except Exception: fit_info["param_change"] = np.nan

    if np.any(np.isclose(fitted, lower)) or np.any(np.isclose(fitted, upper)):
        print("Warning: some fitted parameters hit bounds.")

    return fitted, fit_info

def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=60, tol=1e-4, window=5, seed=1234):
    return None

def milstein_convergence_test(params, V0, T, M=200, dt_values=None, base_seed=123):
    return None

def run_predictive_fitting(pid: int) -> Dict[str, Any]:
    try:
        time_data_full, tum_full = get_patient_data(pid)
        if time_data_full is None or tum_full is None: raise ValueError("Data loading failed.")
        time_data_full, tumor_volume_data_full = sanitize_tumor_data(time_data_full, tum_full)
    except Exception as e: return {'Patient': pid, 'Error': f"Data loading/sanitation failed: {e}"}
    
    midpoint = len(tumor_volume_data_full) // 2
    time_data_train = time_data_full[:midpoint]
    tumor_volume_data_train = tumor_volume_data_full[:midpoint]
    
    if len(tumor_volume_data_train) < 3 or len(tumor_volume_data_full) < 4:
        return {'Patient': pid, 'Error': 'Insufficient data points for training (need at least 3) or testing.'}

    V0 = tumor_volume_data_full[0] 
    
    fitted, fit_info = fit_functional_milstein(time_data_train, tumor_volume_data_train, V0=V0, seed=pid, de_maxiter=3, M_de=12)
    
    if np.all(np.isnan(fitted)):
        return {'Patient': pid, 'Error': 'Fit returned NaN parameters', 'FitInfo': fit_info}

    t_fine_full = np.linspace(time_data_full[0], time_data_full[-1], 500)
    V_max_sim = np.max(tumor_volume_data_full) * 5.0
    mean_fit_full, _, _ = simulate_ensemble_milstein_functional(fitted, V0, t_fine_full, M=60, V_max=V_max_sim, seed=pid)

    y_pred_interp = np.interp(time_data_full, t_fine_full, mean_fit_full)
    
    metrics = {
        'Patient': pid,
        'MASE': mean_absolute_scaled_error(tumor_volume_data_full, y_pred_interp),
        'Chi2': chi_squared(tumor_volume_data_full, y_pred_interp),
        'NSE': nash_sutcliffe_efficiency(tumor_volume_data_full, y_pred_interp),
        'KGE': kling_gupta_efficiency(tumor_volume_data_full, y_pred_interp),
        'FitTime': fit_info.get("fit_time"),
        'ParamChange': fit_info.get("param_change"), 
        'TrajectoryConvergenceTime': None, 
        't_full': time_data_full, 
        'y_full': tumor_volume_data_full, 
        't_fine_full': t_fine_full, 
        'y_pred_full': mean_fit_full, 
    }
    return metrics


def create_single_patient_plot(t_full: np.ndarray, y_full: np.ndarray, t_pred_fine: np.ndarray, y_pred_fine: np.ndarray, pid: int, file_path: str):
    midpoint = len(t_full) // 2
    plt.figure(figsize=(10, 6))

    plt.scatter(t_full[:midpoint], y_full[:midpoint], color=COLOR_SEEN, label='Data Used for Fitting (Seen)', marker='o', zorder=5)
    plt.scatter(t_full[midpoint:], y_full[midpoint:], color=COLOR_UNSEEN, label='Data Hidden from Fitting (Unseen)', marker='x', zorder=5)
    plt.plot(t_pred_fine, y_pred_fine, color=COLOR_LINE, linestyle='-', linewidth=2, label='Fitted Trajectory (Prediction)', zorder=2)

    plt.title(f'Patient {pid}: Predictive Milstein Fit vs. Data', fontsize=16)
    plt.xlabel(r'Time (Weeks)', fontsize=12)
    plt.ylabel(r'Total Tumour Volume ($\mathrm{mm}^3$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_path)
    plt.close()

def create_batch_plot(results: List[Dict[str, Any]], file_path: str):
    plt.figure(figsize=(12, 8))
    
    colors = [BATCH_COLORMAP(i / len(results)) for i in range(len(results))]
    
    handles = []; labels = []
    
    for i, result in enumerate(results):
        t_full = result['t_full']; y_full = result['y_full']
        t_pred_fine = result['t_fine_full']; y_pred_fine = result['y_pred_full']
        midpoint = len(t_full) // 2
        
        c = colors[i]
        
        h1, = plt.plot(t_full[:midpoint], y_full[:midpoint], color=c, linestyle='', marker='o', markersize=4, alpha=0.5, zorder=5)
        h2, = plt.plot(t_full[midpoint:], y_full[midpoint:], color=c, linestyle='', marker='x', markersize=5, alpha=0.5, zorder=5)
        h3, = plt.plot(t_pred_fine, y_pred_fine, color=c, linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
        
        handles.append(h3); labels.append(f'P{result["Patient"]}')
        
    generic_seen = plt.Line2D([0], [0], color='gray', linestyle='', marker='o', label='Data Used for Fitting (Seen)')
    generic_unseen = plt.Line2D([0], [0], color='gray', linestyle='', marker='x', label='Data Hidden from Fitting (Unseen)')
    
    plt.legend(
        handles=[generic_seen, generic_unseen] + handles,
        labels=['Data Used for Fitting (Seen)', 'Data Hidden from Fitting (Unseen)'] + labels,
        loc='upper right',
        fontsize=9,
        ncol=2
    )

    plt.title('Batch Predictive Milstein Trajectories (Color-coded by Patient)', fontsize=16)
    plt.xlabel(r'Time (Weeks)', fontsize=12)
    plt.ylabel(r'Total Tumour Volume ($\mathrm{mm}^3$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_path)
    plt.close()



if __name__ == '__main__':
    
    metrics_results = defaultdict(dict)
    successful_results = []
    
    print("Starting Predictive Milstein Analysis...")

    for pid in tqdm(TESTING_PATIENTS, desc="Processing Patients"):
        print(f"\n=== Processing Patient {pid} ===")
        
        result = run_predictive_fitting(pid)
        
        if 'Error' in result:
            print(f" Failed: {result['Error']}")
            metrics_results[pid] = result
            continue

        plot_file = f"milstein_predictive_plots/patient_{pid:03d}_predictive_fit.png"
        create_single_patient_plot(
            result['t_full'], result['y_full'], result['t_fine_full'], result['y_pred_full'], pid, plot_file
        )
        print(f"Metrics → MASE={result['MASE']:.4f}, KGE={result['KGE']:.4f}")
        print(f" Plot saved to {plot_file}")
        
        metrics_results[pid] = result
        successful_results.append(result)


    if successful_results:
        batch_plot_file = 'milstein_predictive_plots/batch_predictive_trajectories.png'
        create_batch_plot(successful_results, batch_plot_file)
        print(f"\n Batch plot saved to {batch_plot_file}")
    
    # 4. Save all metrics to CSV
    metrics_rows = []
    for pid, data in metrics_results.items():
        if data.get('Error'):
            row = {
                "Patient": pid, "Method": 'Milstein_Predictive', "MASE": np.nan, "Chi2": np.nan, 
                "NSE": np.nan, "KGE": np.nan, "FitTime": np.nan, "ParamChange": np.nan, 
                "TrajectoryConvergenceTime": None, "Error_Message": data.get("Error")
            }
        else:
            row = {
                "Patient": pid, "Method": 'Milstein_Predictive', "MASE": data["MASE"], 
                "Chi2": data["Chi2"], "NSE": data["NSE"], "KGE": data["KGE"], 
                "FitTime": data["FitTime"], "ParamChange": data["ParamChange"], 
                "TrajectoryConvergenceTime": data["TrajectoryConvergenceTime"],
                "Error_Message": None
            }
        row["Conv_dt"] = None; row["StrongError"] = None; row["WeakError"] = None; row["Runtime"] = None
        metrics_rows.append(row)

    df_metrics = pd.DataFrame(metrics_rows)
    output_csv = 'milstein_predictive_metrics_summary.csv'
    df_metrics.to_csv(output_csv, index=False)
    
    total_time = time.time() - script_start_time
    print(f"\n Metrics summary saved to {output_csv}")
    print(f"Total script runtime: {total_time:.2f} s")
    
    print("\n--- Predictive Milstein Fitting Summary ---")
    print(df_metrics[['Patient', 'MASE', 'KGE', 'Error_Message']])