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

os.makedirs("em_predictive_plots", exist_ok=True)
script_start_time = time.time()

start_time_total = time.time()

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

dt_sim_default = 0.5
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def get_patient_data(patient_id, file_path="tumour_data.csv"):
    '''Loads patient data from CSV file.'''
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
    # Clip tumor volume to prevent extreme spikes
    v = np.clip(v, 1.0, 1e5)
    return t, v

def a_of_t(t, a1, a2, alpha, t_c):
    return a1 + a2 * np.tanh(np.clip(alpha * (t - t_c), -50, 50))

def b_of_t(t, b1, b2, beta, t_b):
    return b1 + b2 * sigmoid(beta * (t - t_b))

def euler_maruyama_functional_sde(params, V0, t_grid, chemo_start=3, chemo_end=21,
                                  zero_tol=1e-2, rng=None, dt_sim=dt_sim_default, V_data=None):
    """
    Euler-Maruyama SDE simulator that returns values at t_grid points.
    """
    a1, a2, alpha, t_c, b1, b2, beta, t_b, c, h, k0 = params
    if rng is None:
        rng = np.random.default_rng()
    t_grid = np.asarray(t_grid)
    T = float(t_grid[-1])
    n_steps = max(2, int(np.ceil(T / float(dt_sim))))
    dt = T / n_steps
    if dt <= 0:
        dt = 1e-6
        n_steps = max(n_steps, 1)
    dW = rng.normal(0.0, np.sqrt(dt), size=n_steps)
    V = float(max(V0, 1e-6))
    out = np.empty(len(t_grid))
    idx = 0
    ti = 0.0
    V_max = (np.max(V_data) * 1.1) if (V_data is not None and len(V_data) > 0) else 1e5
    max_rel_increment = 0.5  
    for step in range(n_steps):
        V = max(V, 1e-12)
        at = a_of_t(ti, a1, a2, alpha, t_c)
        bt = b_of_t(ti, b1, b2, beta, t_b)
        k = k0 if (chemo_start <= ti <= chemo_end) else 0.0
        bt_clamped = np.clip(bt, 1e-3, max(V_max, 1e3))
        ratio = bt_clamped / max(V, 1e-12)
        ratio = np.clip(ratio, 1e-12, 1e6)   
        log_term = np.log(ratio)
        drift = at * V * log_term - k * V
        # Post-chemotherapy growth damping
        if ti > chemo_end:
            drift *= max(1 - V / max(V_max, 1e-6), 0.0)
        rel_drift = drift * dt / max(V, 1e-12)
        # Limit relative drift
        if abs(rel_drift) > max_rel_increment:
            drift = np.sign(drift) * max_rel_increment * max(V, 1e-12) / dt
        diffusion = c * V / max(h + np.sqrt(max(V, 0.0)), 1e-12)
        max_diff_allowed = max_rel_increment * max(V, 1e-12) / max(np.sqrt(dt), 1e-6)
        diffusion = np.clip(diffusion, -abs(max_diff_allowed), abs(max_diff_allowed))
        V_new = V + drift * dt + diffusion * dW[step]
        # Prevent extreme values
        if not isfinite(V_new) or V_new > 1e6:
            V_new = min(max(V_new if isfinite(V_new) else V, 1e-12), 1e6)
        V_new = max(V_new, 1e-12)
        V = V_new
        ti += dt
        # Record output at t_grid points
        while idx < len(t_grid) and ti >= t_grid[idx] - 1e-12:
            out[idx] = V
            idx += 1
            if idx >= len(t_grid): break
    while idx < len(t_grid):
        out[idx] = V
        idx += 1
    return out

def simulate_ensemble_functional(params, V0, t_grid, chemo_start=3, chemo_end=21,
                                 M=30, dt_sim=dt_sim_default, seed=None, V_data=None):
    
    '''Simulates an ensemble of trajectories for the functional SDE model.'''
    rng = np.random.default_rng(seed)
    trajs = np.empty((M, len(t_grid)))
    for i in range(M):
        subseed = int(rng.integers(0, 1<<30))
        trajs[i] = euler_maruyama_functional_sde(params, V0, t_grid,
                                                 chemo_start=chemo_start, chemo_end=chemo_end,
                                                 rng=np.random.default_rng(subseed), dt_sim=dt_sim, V_data=V_data)
    mean = np.mean(trajs, axis=0)
    std = np.std(trajs, axis=0)
    return mean, std, trajs

def functional_logmse(params, time_data, observed, V0, M=30, dt_sim=dt_sim_default, seed=42):
    '''Calculates the log-mean-squared-error between model and observed data.'''
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, time_data,
                                                   chemo_start=3, chemo_end=21,
                                                   M=M, dt_sim=dt_sim, seed=seed, V_data=observed)
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj <= 0):
        return 1e30
    return np.sum((np.log(mean_traj + 1e-12) - np.log(observed + 1e-12))**2)

def functional_residuals(params, time_data, observed, V0, M=20, dt_sim=dt_sim_default, seed=123):
    '''Calculates residuals for least squares fitting.'''
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, time_data,
                                                   chemo_start=3, chemo_end=21,
                                                   M=M, dt_sim=dt_sim, seed=seed, V_data=observed)
    mean_traj = np.maximum(mean_traj, 1e-12)
    return np.log(mean_traj + 1e-12) - np.log(observed + 1e-12)

def fit_functional_model(time_data, observed, V0=None, seed=42):
    '''Fits the functional SDE model to observed data using differential evolution and least squares.'''
    if V0 is None: V0 = observed[0]
    obs_max = max(observed)
    # bounds for 11 params
    bounds = [(-0.5,0.5), (-1,1), (0.01,2), (0,time_data[-1]),
              (max(1.0, obs_max*0.2), obs_max*5), (0, obs_max*10), (0.01,2), (0,time_data[-1]),
              (0,2), (1e-6,1e5), (0,0.8)]
    for i, (low, high) in enumerate(bounds):
        if low >= high:
            bounds[i] = (low, low + 1e-3)
    de_start = time.time()
    result = differential_evolution(lambda p: functional_logmse(p, time_data, observed, V0, dt_sim=dt_sim_default), 
                                    bounds, seed=seed, maxiter=10)
    de_end = time.time()
    x0 = result.x
    ls_start = time.time()
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    ls_result = least_squares(lambda p: functional_residuals(p, time_data, observed, V0, dt_sim=dt_sim_default), 
                              x0, bounds=(lower, upper))
    ls_end = time.time()
    fit_info = {
        'fit_time': (de_end - de_start) + (ls_end - ls_start),
        'de_message': result.message,
        'ls_success': bool(ls_result.success),
        'param_change': np.linalg.norm(x0 - ls_result.x)
    }
    return ls_result.x, fit_info
def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=30, tol=1e-4, window=5, seed=123):
    '''Estimates the trajectory convergence time from ensemble simulations.'''
    return None #  skip heavy calculation

def em_convergence_test(params, V0, T, dt_values=None, M=50, base_seed=123):
    '''Performs convergence test for Euler-Maruyama simulations at various dt values.'''
    return None #  skip heavy calculation

def run_predictive_fitting(pid: int) -> Dict[str, Any]:
    '''Runs predictive fitting for a single patient and returns performance metrics.'''
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
    # model using ONLY TRAINING data (using full time_data_train array for dt simulation)
    fitted, fit_info = fit_functional_model(time_data_train, tumor_volume_data_train, V0=V0, seed=pid)
    # Check for NaN parameters
    if np.all(np.isnan(fitted)):
        return {'Patient': pid, 'Error': 'Fit returned NaN parameters', 'FitInfo': fit_info}
# Predict on FULL data time points
    t_fine_full = np.linspace(time_data_full[0], time_data_full[-1], 500)
    mean_fit_full, _, _ = simulate_ensemble_functional(fitted, V0, t_fine_full, M=60, dt_sim=dt_sim_default, seed=pid, V_data=tumor_volume_data_full)
# Interpolate predictions to original time points
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
    '''Generates a plot for a single patient's predictive EM fit.'''
    midpoint = len(t_full) // 2
    plt.figure(figsize=(10, 6))
    plt.scatter(t_full[:midpoint], y_full[:midpoint], color=COLOR_SEEN, label='Data Used for Fitting (Seen)', marker='o', zorder=5)
    plt.scatter(t_full[midpoint:], y_full[midpoint:], color=COLOR_UNSEEN, label='Data Hidden from Fitting (Unseen)', marker='x', zorder=5)
    plt.plot(t_pred_fine, y_pred_fine, color=COLOR_LINE, linestyle='-', linewidth=2, label='Fitted Trajectory (Prediction)', zorder=2)
    plt.title(f'Patient {pid}: Predictive EM Fit vs. Data', fontsize=16)
    plt.xlabel(r'Time (Weeks)', fontsize=12)
    plt.ylabel(r'Total Tumour Volume ($\mathrm{mm}^3$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_path)
    plt.close()

def create_batch_plot(results: List[Dict[str, Any]], file_path: str):
    """Generates a single plot mapping all patient trajectories with distinct colors per patient."""
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

    plt.title('Batch Predictive EM Trajectories (Color-coded by Patient)', fontsize=16)
    plt.xlabel(r'Time (Weeks)', fontsize=12)
    plt.ylabel(r'Total Tumour Volume ($\mathrm{mm}^3$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
    metrics_results = defaultdict(dict)
    successful_results = []
    for pid in tqdm(TESTING_PATIENTS, desc="Processing Patients"):
        print(f"\n=== Processing Patient {pid} ===")
        result = run_predictive_fitting(pid)
        if 'Error' in result:
            print(f"Failed: {result['Error']}")
            metrics_results[pid] = result
            continue
        plot_file = f"em_predictive_plots/patient_{pid:03d}_predictive_fit.png"
        create_single_patient_plot(
            result['t_full'], result['y_full'], result['t_fine_full'], result['y_pred_full'], pid, plot_file
        )
        print(f"Metrics → MASE={result['MASE']:.4f}, KGE={result['KGE']:.4f}")
        print(f"Plot saved to {plot_file}")
        metrics_results[pid] = result
        successful_results.append(result)
    if successful_results:
        batch_plot_file = 'em_predictive_plots/batch_predictive_trajectories.png'
        create_batch_plot(successful_results, batch_plot_file)
        print(f"\nBatch plot saved to {batch_plot_file}")
    # Save all metrics to CSV
    metrics_rows = []
    for pid, data in metrics_results.items():
        if data.get('Error'):
            row = {
                "Patient": pid, "Method": 'EM_Predictive', "MASE": np.nan, "Chi2": np.nan, 
                "NSE": np.nan, "KGE": np.nan, "FitTime": np.nan, "ParamChange": np.nan, 
                "TrajectoryConvergenceTime": None, "Error_Message": data.get("Error")
            }
        else:
            row = {
                "Patient": pid, "Method": 'EM_Predictive', "MASE": data["MASE"], 
                "Chi2": data["Chi2"], "NSE": data["NSE"], "KGE": data["KGE"], 
                "FitTime": data["FitTime"], "ParamChange": data["ParamChange"], 
                "TrajectoryConvergenceTime": data["TrajectoryConvergenceTime"],
                "Error_Message": None
            }
        row["Conv_dt"] = None; row["StrongError"] = None; row["WeakError"] = None; row["Runtime"] = None
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    output_csv = 'em_predictive_metrics_summary.csv'
    df_metrics.to_csv(output_csv, index=False)
    total_time = time.time() - script_start_time
    print(f"\nMetrics summary saved to {output_csv}")
    print(f"Total script runtime: {total_time:.2f} s")
    print("\n--- Predictive EM Fitting Summary ---")
    print(df_metrics[['Patient', 'MASE', 'KGE', 'Error_Message']])