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
DATA_FILENAME = "tumour_data.csv"

# Directories for output
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
    if np.std(y_true) == 0: return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def get_patient_data(pid: int):
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
    valid = np.isfinite(t) & np.isfinite(v)
    return t[valid], v[valid]

def drift_mil(V, t, params):
    rho, K = params[:2]
    return rho * V * (1 - (V / K))

def diff_mil(V, t, params):
    sigma = params[2]
    return sigma * V

def d_diff_mil(V, t, params):
    sigma = params[2]
    return sigma

def milstein_functional_sde(params, V0, t_eval, dt_sim=0.1, V_max=1e7, seed=None):
    if seed is not None: np.random.seed(seed)
    T_max = t_eval[-1]
    num_steps = int(np.ceil(T_max / dt_sim))
    V = np.zeros(num_steps + 1); V[0] = V0
    times = np.linspace(0, num_steps * dt_sim, num_steps + 1)
    
    for i in range(num_steps):
        f = drift_mil(V[i], times[i], params)
        g = diff_mil(V[i], times[i], params)
        gp = d_diff_mil(V[i], times[i], params)
        dW = np.random.normal(0, np.sqrt(dt_sim))
        
        # Milstein correction term: 0.5 * g * g' * (dW^2 - dt)
        V[i+1] = V[i] + f * dt_sim + g * dW + 0.5 * g * gp * (dW**2 - dt_sim)
        V[i+1] = np.clip(V[i+1], 1e-8, V_max)
    
    return np.interp(t_eval, times, V)

def simulate_ensemble_milstein_functional(params, V0, t_eval, M=30, dt_sim=0.1, V_max=1e7, seed=123):
    sims = np.array([milstein_functional_sde(params, V0, t_eval, dt_sim, V_max, seed + i) for i in range(M)])
    return np.mean(sims, axis=0), np.std(sims, axis=0), sims

def milstein_residuals(params, t_data, V_data, V0, V_max, seed):
    mean_v, _, _ = simulate_ensemble_milstein_functional(params, V0, t_data, M=30, dt_sim=0.1, V_max=V_max, seed=seed)
    return mean_v - V_data

def fit_functional_milstein(t_data, V_data, V0, seed=123, de_maxiter=50, M_de=15):
    start = time.time()
    V_max = np.max(V_data) * 10.0
    bounds = [(1e-4, 0.5), (max(V_data)*0.5, max(V_data)*5.0), (1e-4, 0.5)]
    res_de = differential_evolution(lambda p: np.sum(milstein_residuals(p, t_data, V_data, V0, V_max, seed)**2), bounds, seed=seed, maxiter=de_maxiter)
    res_ls = least_squares(milstein_residuals, res_de.x, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), args=(t_data, V_data, V0, V_max, seed))
    return res_ls.x, {"fit_time": time.time() - start, "param_change": np.linalg.norm(res_ls.x - res_de.x)}

def trajectory_convergence_time_from_ensemble(params, V0, T, dt=0.1, M=60, tol=1e-4, window=5, seed=1234):
    t_grid = np.arange(0, T + dt, dt)
    mean_traj, _, _ = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=seed)
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    for i in range(len(rel_changes) - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i+1+window-1]
    return None

def milstein_convergence_test(params, V0, T, M=60, dt_values=None, base_seed=123):
    if dt_values is None: dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]
    ref_dt = 0.01
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    mean_ref, _, sims_ref = simulate_ensemble_milstein_functional(params, V0, t_ref, M=M, dt_sim=ref_dt, seed=base_seed)
    mean_ref_final = np.mean(sims_ref[:, -1])

    results = {"dt_values": [], "strong_error": [], "weak_error": [], "runtime": []}
    for dt in dt_values:
        start_t = time.time()
        t_grid = np.arange(0, T + dt, dt)
        mean_dt, _, sims_dt = simulate_ensemble_milstein_functional(params, V0, t_grid, M=M, dt_sim=dt, seed=base_seed)
        ref_interp = np.interp(t_grid, t_ref, mean_ref)
        strong_err = np.mean(np.abs(mean_dt - ref_interp))
        weak_err = np.abs(np.mean(sims_dt[:, -1]) - mean_ref_final)
        results["dt_values"].append(dt)
        results["strong_error"].append(strong_err)
        results["weak_error"].append(weak_err)
        results["runtime"].append(time.time() - start_t)
    return results

def run_predictive_fitting(pid: int) -> Dict[str, Any]:
    data_res, err = get_patient_data(pid)
    if err: return {'Patient': pid, 'Error': err}
    time_data_full, tumor_volume_data_full = sanitize_tumor_data(data_res[0], data_res[1])
    midpoint = len(tumor_volume_data_full) // 2
    time_data_train, tumor_volume_data_train = time_data_full[:midpoint], tumor_volume_data_full[:midpoint]
    
    if len(time_data_train) < 3: return {'Patient': pid, 'Error': 'Insufficient training data'}
    V0 = tumor_volume_data_full[0] 
    
    fitted, fit_info = fit_functional_milstein(time_data_train, tumor_volume_data_train, V0=V0, seed=pid)
    t_fine_full = np.linspace(time_data_full[0], time_data_full[-1], 500)
    mean_fit_full, _, _ = simulate_ensemble_milstein_functional(fitted, V0, t_fine_full, M=60, seed=pid)
    y_pred_interp = np.interp(time_data_full, t_fine_full, mean_fit_full)
    
    tct = trajectory_convergence_time_from_ensemble(fitted, V0, time_data_full[-1], M=60, seed=pid)
    conv = milstein_convergence_test(fitted, V0, time_data_full[-1], M=60, base_seed=pid)
    
    return {
        'Patient': pid,
        'MASE': mean_absolute_scaled_error(tumor_volume_data_full, y_pred_interp),
        'Chi2': chi_squared(tumor_volume_data_full, y_pred_interp),
        'NSE': nash_sutcliffe_efficiency(tumor_volume_data_full, y_pred_interp),
        'KGE': kling_gupta_efficiency(tumor_volume_data_full, y_pred_interp),
        'FitTime': fit_info.get("fit_time"),
        'ParamChange': fit_info.get("param_change"),
        'TrajectoryConvergenceTime': tct,
        'ConvergenceResults': conv
    }

if __name__ == '__main__':
    metrics_results = {}
    for pid in tqdm(TESTING_PATIENTS, desc="Milstein Predictive Run"):
        metrics_results[pid] = run_predictive_fitting(pid)

    metrics_rows = []
    for pid, data in metrics_results.items():
        row = {"Patient": pid, "Method": 'Milstein_Predictive', "Error_Message": data.get("Error")}
        if not data.get('Error'):
            row.update({"MASE": data["MASE"], "Chi2": data["Chi2"], "NSE": data["NSE"], "KGE": data["KGE"], 
                        "FitTime": data["FitTime"], "ParamChange": data["ParamChange"], 
                        "TrajectoryConvergenceTime": data["TrajectoryConvergenceTime"]})
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
    output_csv = 'milstein_predictive_metrics_summary.csv'
    df_metrics.to_csv(output_csv, index=False)
    print(f"\n Milstein results saved to {output_csv}")