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
dt_sim_default = 0.1
DATA_FILENAME = "tumour_data.csv"

# Directories for output
os.makedirs("em_predictive_plots", exist_ok=True)
script_start_time = time.time()

# Metrics
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
    if np.std(y_true) == 0: return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# Data 
def get_patient_data(pid: int):
    '''Loads patient data from tumour_data.csv.'''
    try:
        df = pd.read_csv(DATA_FILENAME)
        p_str = f"Patient-{pid:03d}"
        p_data = df[df['Patient'] == p_str].copy()
        if p_data.empty: return None, f"Patient {p_str} not found"
        
        def week_to_day(w):
            nums = re.findall(r'\d+', str(w))
            return float(nums[0]) * 7 if nums else 0.0
            
        p_data['Day'] = p_data['Week'].apply(week_to_day)
        p_data = p_data.sort_values('Day')
        return (p_data['Day'].values, p_data['Total_mm3'].values), None
    except Exception as e: return None, str(e)

def sanitize_tumor_data(t, v):
    '''Removes non-finite entries from tumor data.'''
    valid = np.isfinite(t) & np.isfinite(v)
    return t[valid], v[valid]

# SDE Model Components
def a_of_t(V, t, params):
    rho, K = params[:2]
    return rho * V * (1 - (V / K))

def b_of_t(V, t, params):
    sigma = params[2]
    return sigma * V

def euler_maruyama_functional_sde(params, V0, t_eval, dt_sim=0.1, seed=None):
    '''Simulates SDE using Euler-Maruyama method for functional SDE.'''
    if seed is not None: np.random.seed(seed)
    T_max = t_eval[-1]
    num_steps = int(np.ceil(T_max / dt_sim))
    V = np.zeros(num_steps + 1); V[0] = V0
    times = np.linspace(0, num_steps * dt_sim, num_steps + 1)
    for i in range(num_steps):
        drift = a_of_t(V[i], times[i], params)
        diffusion = b_of_t(V[i], times[i], params)
        dW = np.random.normal(0, np.sqrt(dt_sim))
        V[i+1] = max(1e-8, V[i] + drift * dt_sim + diffusion * dW)
    return np.interp(t_eval, times, V)

def simulate_ensemble_functional(params, V0, t_eval, M=30, dt_sim=0.1, seed=123):
    '''Simulates an ensemble of SDE trajectories and computes mean and std deviation.'''
    sims = np.array([euler_maruyama_functional_sde(params, V0, t_eval, dt_sim, seed + i) for i in range(M)])
    return np.mean(sims, axis=0), np.std(sims, axis=0), sims

# Fitting
def functional_residuals(params, t_data, V_data, V0, seed):
    '''Computes residuals between model mean and data for fitting.'''
    mean_v, _, _ = simulate_ensemble_functional(params, V0, t_data, M=30, dt_sim=0.1, seed=seed)
    return mean_v - V_data

def fit_functional_model(t_data, V_data, V0, seed=123):
    '''Fits the functional SDE model to data using DE + LS.'''
    start = time.time()
    bounds = [(1e-4, 0.5), (max(V_data)*0.5, max(V_data)*5.0), (1e-4, 0.5)]
    res_de = differential_evolution(lambda p: np.sum(functional_residuals(p, t_data, V_data, V0, seed)**2), bounds, seed=seed, maxiter=50)
    res_ls = least_squares(functional_residuals, res_de.x, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), args=(t_data, V_data, V0, seed))
    return res_ls.x, {"fit_time": time.time() - start, "param_change": np.linalg.norm(res_ls.x - res_de.x)}

# Convergence Logic (TCT & Conv Test)
def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=60, tol=1e-4, window=5, seed=123):
    '''Determines convergence time of trajectory from ensemble simulations.'''
    t_grid = np.arange(0, T + dt, dt)
    mean_traj, _, _ = simulate_ensemble_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=seed)
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    for i in range(len(rel_changes) - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i+1+window-1]
    return T #  max T if no convergence found

def em_convergence_test(params, V0, T, dt_values=None, M=60, base_seed=123):
    '''Performs convergence test for Euler-Maruyama method.'''
    if dt_values is None: dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    ref_dt = 0.01
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    mean_ref, _, sims_ref = simulate_ensemble_functional(params, V0, t_ref, M=M, dt_sim=ref_dt, seed=base_seed)
    mean_ref_final = np.mean(sims_ref[:, -1])
    res = {"dt_values": [], "strong_error": [], "weak_error": [], "runtime": []}
    for dt in dt_values:
        start_t = time.time()
        t_grid = np.arange(0, T + dt, dt)
        mean_dt, _, sims_dt = simulate_ensemble_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=base_seed)
        ref_interp = np.interp(t_grid, t_ref, mean_ref)
        res["dt_values"].append(dt)
        res["strong_error"].append(np.mean(np.abs(mean_dt - ref_interp)))
        res["weak_error"].append(np.abs(np.mean(sims_dt[:, -1]) - mean_ref_final))
        res["runtime"].append(time.time() - start_t)
    return res

# Main logic
def run_predictive_fitting(pid: int) -> Dict[str, Any]:
    '''Runs predictive fitting for a given patient ID and computes metrics.'''
    data_res, err = get_patient_data(pid)
    if err: return {'Patient': pid, 'Error': err}
    t_full, y_full = sanitize_tumor_data(data_res[0], data_res[1])
    midpoint = len(y_full) // 2
    t_train, y_train = t_full[:midpoint], y_full[:midpoint]
    if len(t_train) < 3: return {'Patient': pid, 'Error': 'Insufficient training data'}
    V0 = y_full[0] 
    fitted, fit_info = fit_functional_model(t_train, y_train, V0=V0, seed=pid)
    t_fine = np.linspace(t_full[0], t_full[-1], 500)
    mean_fit, _, _ = simulate_ensemble_functional(fitted, V0, t_fine, M=60, seed=pid)
    y_pred_interp = np.interp(t_full, t_fine, mean_fit)
    tct = trajectory_convergence_time_from_ensemble(fitted, V0, t_full[-1], M=60, seed=pid)
    conv = em_convergence_test(fitted, V0, t_full[-1], M=60, base_seed=pid)
    return {
        'Patient': pid,
        'MASE': mean_absolute_scaled_error(y_full, y_pred_interp),
        'Chi2': chi_squared(y_full, y_pred_interp),
        'NSE': nash_sutcliffe_efficiency(y_full, y_pred_interp),
        'KGE': kling_gupta_efficiency(y_full, y_pred_interp),
        'FitTime': fit_info.get("fit_time"),
        'ParamChange': fit_info.get("param_change"),
        'TrajectoryConvergenceTime': tct, 
        'ConvergenceResults': conv
    }

if __name__ == '__main__':
    metrics_results = {}
    for pid in tqdm(TESTING_PATIENTS, desc="EM Predictive Run"):
        metrics_results[pid] = run_predictive_fitting(pid)
    metrics_rows = []
    for pid, data in metrics_results.items():
        row = {"Patient": pid, "Method": 'EM_Predictive', "Error_Message": data.get("Error")}
        if not data.get('Error'):
            row.update({
                "MASE": data["MASE"], "Chi2": data["Chi2"], "NSE": data["NSE"], "KGE": data["KGE"], 
                "FitTime": data["FitTime"], "ParamChange": data["ParamChange"], 
                "TrajectoryConvergenceTime": data["TrajectoryConvergenceTime"]
            })
            conv = data.get('ConvergenceResults')
            row["Conv_dt"] = "|".join(map(str, conv["dt_values"]))
            row["StrongError"] = "|".join(map(str, conv["strong_error"]))
            row["WeakError"] = "|".join(map(str, conv["weak_error"]))
            row["Runtime"] = "|".join(map(str, conv["runtime"]))
        else:
            for k in ["MASE", "Chi2", "NSE", "KGE", "FitTime", "ParamChange", "TrajectoryConvergenceTime", "Conv_dt", "StrongError", "WeakError", "Runtime"]:
                row[k] = np.nan
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv('em_predictive_metrics_summary.csv', index=False)
    print("\nEM Results saved with TCT data.")