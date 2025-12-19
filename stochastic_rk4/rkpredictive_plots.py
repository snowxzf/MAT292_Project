import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares
from tqdm import tqdm
import time, os, re
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict 
from matplotlib import cm 

COLOR_SEEN = '#3293a8'      
COLOR_UNSEEN = '#a83e32'    
COLOR_LINE = '#3244a8'      
BATCH_COLORMAP = cm.tab10 

TESTING_PATIENTS: List[int] = [31, 19, 43, 54, 77, 73, 72, 71, 67, 52]

os.makedirs("srk_predictive_plots", exist_ok=True)
os.makedirs("srk_conv_predictive", exist_ok=True) 
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

# Utility functio (EXACTLY as provided in rk.py)
dt_sim = 0.5
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_patient_data(patient_id, file_path="tumour_data.csv"):
    '''retrieves patient data from CSV file given patient ID.'''
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
    # Sort by weeks
    sort_idx = np.argsort(weeks)
    weeks = weeks[sort_idx]
    volumes = volumes[sort_idx]

    return weeks, volumes

def sanitize_tumor_data(time_data, tumor_data):
    '''Cleans tumor data by removing non-finite values, interpolating non-positive values, and clipping extreme values.'''
    t = np.array(time_data, dtype=float)
    v = np.array(tumor_data, dtype=float)
    valid_mask = np.isfinite(v)
    t, v = t[valid_mask], v[valid_mask]
    if len(v) == 0: return t, v
    if np.any(v <= 0):
        nonzero_idx = np.where(v > 0)[0]
        zero_idx = np.where(v <= 0)[0]
        if len(nonzero_idx) > 1:
            v[zero_idx] = np.interp(zero_idx, nonzero_idx, v[nonzero_idx])
        else:
            v = np.maximum(v, 10.0)
    v = np.clip(v, 1.0, 1e7)
    return t, v

def a_of_t(t, a1, a2, alpha, t_c): # time-varying growth rate
    return a1 + a2 * np.tanh(alpha * (t - t_c))
def b_of_t(t, b1, b2, beta, t_b): # time-varying carrying capacity
    return b1 + b2 * sigmoid(beta * (t - t_b))

def srk4_step_functional(V, t, dt, dW, params):
    '''Performs a single SRK4 step for the functional SDE model.'''
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
    c1, c2, c3, c4 = 0.0, 0.5, 0.5, 1.0
    bRK = np.array([1/6, 1/3, 1/3, 1/6])
    K = np.zeros(4); L = np.zeros(4); Vhat = np.zeros(4)
    # Stage 1
    Vhat[0] = V; K[0] = f(Vhat[0], t + c1*dt); L[0] = g(Vhat[0], t + c1*dt)
    # Stage 2
    Vhat[1] = V + dt * (0.5 * K[0]) + dW * (0.5 * L[0]); K[1] = f(Vhat[1], t + c2*dt); L[1] = g(Vhat[1], t + c2*dt)
    # Stage 3
    Vhat[2] = V + dt * (0.5 * K[1]) + dW * (0.5 * L[1]); K[2] = f(Vhat[2], t + c3*dt); L[2] = g(Vhat[2], t + c3*dt)
    # Stage 4
    Vhat[3] = V + dt * K[2] + dW * L[2]; K[3] = f(Vhat[3], t + c4*dt); L[3] = g(Vhat[3], t + c4*dt)
    V_new = V + dt * np.dot(bRK, K) + dW * np.dot(bRK, L)
    # Milstein-type correction
    g_mid = g(V, t); dg_mid = dg_dV(V, t)
    V_new += 0.5 * g_mid * dg_mid * (dW**2 - dt)
    V_new = np.clip(V_new, 1e-6, 1e6)
    # safeguard against non-finite or excessively large values
    if not np.isfinite(V_new) or V_new > 1e8:
        V_new = min(max(V_new if np.isfinite(V_new) else eps, eps), 1e5)
    return max(V_new, eps)

def simulate_tumor_srk_functional(params, V0, t_grid, seed=None):
    '''simulates a single trajectory of the SRK functional SDE model.'''
    rng = np.random.default_rng(seed)
    V = np.empty(len(t_grid))
    V[0] = float(V0)
    for i in range(1, len(t_grid)):
        dt = t_grid[i] - t_grid[i - 1]
        dW = np.sqrt(dt) * rng.standard_normal() 
        V[i] = srk4_step_functional(V[i - 1], t_grid[i - 1], dt, dW, params)
    return V

def simulate_ensemble_srk_functional(params, V0, t_grid, M=40, seed=None):
    '''simulates an ensemble of M trajectories of the SRK functional SDE model.'''
    rng = np.random.default_rng(seed)
    sims = np.empty((M, len(t_grid)))
    for j in range(M):
        sims[j] = simulate_tumor_srk_functional(params, V0, t_grid, seed=int(rng.integers(0, 1<<30)))
    mean_traj = np.mean(sims, axis=0)
    std_traj = np.std(sims, axis=0)
    return mean_traj, std_traj, sims

def functional_logmse_srk(params, time_data, observed, V0, M=25, dt_sim=0.5, seed=42):
    '''computes the log-MSE between observed data and model predictions for given parameters.'''
    try:
        a1,a2,alpha,t_c, b1,b2,beta,t_b, c, h, k0, r, m = params
    except Exception: return 1e30
    if b1 <= 0 or h <= 0 or c < 0 or alpha <= 0 or beta <= 0 or m <= 0: return 1e30
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=seed)
    if np.any(~np.isfinite(mean_traj)) or np.any(mean_traj > 1e8): return 1e30
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.sum((np.log(interp + 1e-6) - np.log(observed + 1e-6))**2)

def functional_residuals_srk(params, time_data, observed, V0, M=20, dt_sim=0.5, seed=123):
    ''' computes residuals between log-transformed observed data and model predictions for given parameters.'''
    t_grid = np.linspace(time_data[0], time_data[-1], max(40, int((time_data[-1]-time_data[0]) / dt_sim)))
    mean_traj, _, _ = simulate_ensemble_srk_functional(params, V0, t_grid, M=M, seed=seed)
    interp = np.interp(time_data, t_grid, mean_traj)
    return np.log(interp + 1e-6) - np.log(observed + 1e-6)


def fit_functional_srk(time_data, observed, V0=None, seed=42, de_maxiter=6, M_de=20):
    '''fits the functional SRK model parameters to tumor volume data using differential evolution and least squares.'''
    fit_info = {"fit_time": None, "de_message": None, "param_change": None, "ls_success": None}
    if time_data is None or observed is None or len(time_data) < 3 or len(observed) < 3:
        print("⚠️ Insufficient data → skipping fit."); return np.full(13, np.nan), fit_info
    if V0 is None: V0 = observed[0]
    obs_max = max(observed)
    if not np.isfinite(obs_max) or obs_max <= 0:
        print(" Invalid tumor data (obs_max)."); return np.full(13, np.nan), fit_info
    # Define parameter bounds
    bounds = [
    (-0.5, 0.5), (-1.0, 1.0), (0.01, 2.0), (0.0, time_data[-1]),
    (max(1.0, obs_max * 0.2), obs_max * 5 + 1e-3),
    (0.0, obs_max * 10 + 1e-3), (0.01, 2.0), (0.0, time_data[-1]),
    (0.0, 2.0), (1e-3, 1e5), (0.0, 0.8), (0.0, 0.8), (10.0, obs_max * 10 + 1e-3)
    ]
    # Differential Evolution
    print("Running Differential Evolution (global search)...")
    pbar = tqdm(total=de_maxiter, desc="DE", leave=False, disable=True) 
    def _cb(xk, convergence): pbar.update(1)
    de_start = time.time()
    try:
        result = differential_evolution(lambda p: functional_logmse_srk(p, time_data, observed, V0, M=M_de, dt_sim=dt_sim, seed=seed),
                                        bounds, maxiter=de_maxiter, seed=seed, callback=_cb, polish=False)
        pbar.close(); fit_info["de_message"] = result.message
    except Exception as e:
        pbar.close(); print("DE failed:", e); return np.full(13, np.nan), fit_info
    de_end = time.time()
    # Least Squares polishing
    x0 = result.x.copy()
    print("Polishing with least_squares...")
    lower = np.array([b[0] for b in bounds]); upper = np.array([b[1] for b in bounds])
    ls_start = time.time()
    try:
        ls = least_squares(lambda p: functional_residuals_srk(p, time_data, observed, V0, M=18, dt_sim=dt_sim, seed=seed+1),
                           x0, bounds=(lower, upper), xtol=1e-6, ftol=1e-6, max_nfev=200)
    except Exception as e:
        print("Least-squares failed:", e); ls = None
    ls_end = time.time()
    # Compile results
    if ls is None: fitted = x0; fit_info["ls_success"] = False
    else: fitted = ls.x; fit_info["ls_success"] = bool(ls.success)
    # Record fit info
    fit_info["fit_time"] = (de_end - de_start) + (ls_end - ls_start)
    try: fit_info["param_change"] = np.linalg.norm(x0 - fitted)
    except Exception: fit_info["param_change"] = np.nan
    # Check for parameters hitting bounds
    if np.any(np.isclose(fitted, lower)) or np.any(np.isclose(fitted, upper)):
        print("⚠️ Warning: some fitted parameters hit bounds.")
    return fitted, fit_info

def run_predictive_fitting(pid: int) -> Dict[str, Any]:
    '''Runs predictive fitting for a given patient ID using SRK SDE model.'''
    try:
        time_data_full, tum_full = get_patient_data(pid)
        if time_data_full is None or tum_full is None: raise ValueError("Data loading failed.")
        time_data_full, tumor_volume_data_full = sanitize_tumor_data(time_data_full, tum_full)
    except Exception as e: return {'Patient': pid, 'Error': f"Data loading/sanitation failed: {e}"}
    # Split Data
    midpoint = len(tumor_volume_data_full) // 2
    time_data_train = time_data_full[:midpoint]
    tumor_volume_data_train = tumor_volume_data_full[:midpoint]
    # Check data sufficiency
    if len(tumor_volume_data_train) < 3 or len(tumor_volume_data_full) < 4:
        return {'Patient': pid, 'Error': 'Insufficient data points for training (need at least 3) or testing.'}
# Initial condition
    V0 = tumor_volume_data_full[0] 
    # Fit and Predict
    fitted, fit_info = fit_functional_srk(time_data_train, tumor_volume_data_train, V0=V0, seed=pid, de_maxiter=6, M_de=20)
    if np.all(np.isnan(fitted)):
        return {'Patient': pid, 'Error': 'Fit returned NaN parameters', 'FitInfo': fit_info}
# Generate fine prediction over full time
    t_fine_full = np.linspace(time_data_full[0], time_data_full[-1], 500)
    mean_fit_full, _, _ = simulate_ensemble_srk_functional(fitted, V0, t_fine_full, M=60, seed=pid)
    y_pred_interp = np.interp(time_data_full, t_fine_full, mean_fit_full)
    metrics = {
        'Patient': pid,
        'MASE': mean_absolute_scaled_error(tumor_volume_data_full, y_pred_interp),
        'Chi2': chi_squared(tumor_volume_data_full, y_pred_interp),
        'NSE': nash_sutcliffe_efficiency(tumor_volume_data_full, y_pred_interp),
        'KGE': kling_gupta_efficiency(tumor_volume_data_full, y_pred_interp),
        'FitInfo': fit_info,
        'ParamChange': fit_info.get("param_change"), 
        'FitTime': fit_info.get("fit_time"),
        'TrajectoryConvergenceTime': None, 
        't_full': time_data_full, 
        'y_full': tumor_volume_data_full, 
        't_fine_full': t_fine_full, 
        'y_pred_full': mean_fit_full, 
    }
    return metrics


def create_single_patient_plot(t_full: np.ndarray, y_full: np.ndarray, t_pred_fine: np.ndarray, y_pred_fine: np.ndarray, pid: int, file_path: str):
    '''Generates a plot for a single patient comparing observed data and fitted trajectory.'''
    midpoint = len(t_full) // 2
    plt.figure(figsize=(10, 6))
    plt.scatter(t_full[:midpoint], y_full[:midpoint], color=COLOR_SEEN, label='Data Used for Fitting (Seen)', marker='o', zorder=5)
    plt.scatter(t_full[midpoint:], y_full[midpoint:], color=COLOR_UNSEEN, label='Data Hidden from Fitting (Unseen)', marker='x', zorder=5)
    plt.plot(t_pred_fine, y_pred_fine, color=COLOR_LINE, linestyle='-', linewidth=2, label='Fitted Trajectory (Prediction)', zorder=2)
    plt.title(f'Patient {pid}: Predictive SRK Fit vs. Data', fontsize=16)
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
    # Plot each patient's data and fitted trajectory
    handles = []
    labels = []
    for i, result in enumerate(results):
        t_full = result['t_full']
        y_full = result['y_full']
        t_pred_fine = result['t_fine_full']
        y_pred_fine = result['y_pred_full']
        midpoint = len(t_full) // 2
        c = colors[i]
        
        h1, = plt.plot(t_full[:midpoint], y_full[:midpoint], color=c, linestyle='', marker='o', markersize=4, alpha=0.5, zorder=5)
        
        h2, = plt.plot(t_full[midpoint:], y_full[midpoint:], color=c, linestyle='', marker='x', markersize=5, alpha=0.5, zorder=5)
        
        h3, = plt.plot(t_pred_fine, y_pred_fine, color=c, linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)
        handles.append(h3) 
        labels.append(f'P{result["Patient"]}')
    generic_seen = plt.Line2D([0], [0], color='gray', linestyle='', marker='o', label='Data Used for Fitting (Seen)')
    generic_unseen = plt.Line2D([0], [0], color='gray', linestyle='', marker='x', label='Data Hidden from Fitting (Unseen)')
    plt.legend(
        handles=[generic_seen, generic_unseen] + handles,
        labels=['Data Used for Fitting (Seen)', 'Data Hidden from Fitting (Unseen)'] + labels,
        loc='upper right',
        fontsize=9,
        ncol=2
    )
    plt.title('Batch Predictive SRK Trajectories (Color-coded by Patient)', fontsize=16)
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
            print(f" Failed: {result['Error']}")
            metrics_results[pid] = result
            continue
        plot_file = f"srk_predictive_plots/patient_{pid:03d}_predictive_fit.png"
        create_single_patient_plot(
            result['t_full'], result['y_full'], result['t_fine_full'], result['y_pred_full'], pid, plot_file
        )
        print(f"Metrics → MASE={result['MASE']:.4f}, KGE={result['KGE']:.4f}")
        print(f" Plot saved to {plot_file}")
        metrics_results[pid] = result
        successful_results.append(result)
    if successful_results:
        batch_plot_file = 'srk_predictive_plots/batch_predictive_trajectories.png'
        create_batch_plot(successful_results, batch_plot_file)
        print(f"\n Batch plot saved to {batch_plot_file}")
    metrics_rows = []
    for pid, data in metrics_results.items():
        if data.get('Error'):
            row = {
                "Patient": pid, "Method": 'SRK_Predictive', "MASE": np.nan, "Chi2": np.nan, 
                "NSE": np.nan, "KGE": np.nan, "FitTime": np.nan, "ParamChange": np.nan, 
                "TrajectoryConvergenceTime": None, "Error_Message": data.get("Error")
            }
        else:
            row = {
                "Patient": pid, "Method": 'SRK_Predictive', "MASE": data["MASE"], 
                "Chi2": data["Chi2"], "NSE": data["NSE"], "KGE": data["KGE"], 
                "FitTime": data["FitTime"], "ParamChange": data["ParamChange"], 
                "TrajectoryConvergenceTime": data["TrajectoryConvergenceTime"],
                "Error_Message": None
            }
        row["Conv_dt"] = None; row["StrongError"] = None; row["WeakError"] = None; row["Runtime"] = None
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    output_csv = 'srk_predictive_metrics_summary.csv'
    df_metrics.to_csv(output_csv, index=False)
    total_time = time.time() - script_start_time
    print(f"\n Metrics summary saved to {output_csv}")

    print(f"Total script runtime: {total_time:.2f} s")
    print("\n--- Predictive SRK Fitting Summary ---")
    print(df_metrics[['Patient', 'MASE', 'KGE', 'Error_Message']])