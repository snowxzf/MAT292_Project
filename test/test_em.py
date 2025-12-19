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

#aesthetics
COLOR_SEEN = '#3293a8'      
COLOR_UNSEEN = '#a83e32'    
COLOR_LINE = '#3244a8'      
BATCH_COLORMAP = cm.tab10 
TESTING_PATIENTS: List[int] = [19]
DATA_FILENAME = "tumour_data.csv"
OUTPUT_DIR = "em_predictive_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# metrics
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
    return 1 - np.sum((y_true - y_pred)**2) / (denom + 1e-12) if denom != 0 else np.nan

def kling_gupta_efficiency(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0: return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-12)
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-12)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# EM model
def sigmoid(x): #creates smooth transition
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def a_of_t(t, a1, a2, alpha, t_c): #growth rate using tanh function
    return a1 + a2 * np.tanh(np.clip(alpha * (t - t_c), -50, 50))

def b_of_t(t, b1, b2, beta, t_b): #carrying capacity
    return b1 + b2 * sigmoid(beta * (t - t_b))

def euler_maruyama_step(V, t, dt, dW, params, V_max):
    '''calculates the tumour size for the next tiny slice of time. 
     takes the current size, adds "drift" (predictable growth/treatment effect), and adds "diffusion" (random noise/fluctuations).'''
    eps = 1e-12
    V = max(V, eps)
    a1, a2, alpha, t_c, b1, b2, beta, t_b, c, h, k0 = params
    at = a_of_t(t, a1, a2, alpha, t_c)
    bt = b_of_t(t, b1, b2, beta, t_b)
    
    k = k0 if (3 <= t <= 21) else 0.0
    drift = at * V * np.log(np.clip(bt/V, 1e-12, 1e6)) - k * V
    if t > 21: drift *= max(1 - V/V_max, 0.0)
    
    diffusion = c * V / (h + np.sqrt(max(V, 0.0)) + 1e-12)
    V_new = V + drift * dt + diffusion * dW
    return np.clip(V_new, eps, V_max)

def simulate_ensemble(params, V0, t_grid, M=10, seed=123, V_max=None):
    '''the model has random noise, running it once isn't enough. 
    runs the simulation M times (an "ensemble") and returns the average path and the standard deviation'''
    rng = np.random.default_rng(seed)
    if V_max is None: V_max = 1e7
    trajs = np.zeros((M, len(t_grid)))
    for j in range(M):
        sub_rng = np.random.default_rng(int(rng.integers(0, 1<<30)))
        v = np.zeros(len(t_grid)); v[0] = V0
        for i in range(1, len(t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            dW = sub_rng.normal(0, np.sqrt(dt))
            v[i] = euler_maruyama_step(v[i-1], t_grid[i-1], dt, dW, params, V_max)
        trajs[j] = v
    return np.mean(trajs, axis=0), np.std(trajs, axis=0), trajs

# tct
def calculate_tct(mean_traj, t_grid, tol=1e-4, window=5):
    '''the "Time to Constant Tumour." It looks across the prediction to find the point where the 
    size stops changing significantly (within a specific tolerance) for a few days in a row'''
    mean_traj = np.maximum(mean_traj, 1e-8)
    rel_changes = np.abs(np.diff(mean_traj) / mean_traj[:-1])
    for i in range(len(rel_changes) - window + 1):
        if np.all(rel_changes[i:i+window] < tol):
            return t_grid[i]
    return t_grid[-1]

# convergence
def run_convergence_suite(params, V0, T, seed):
    '''runs the simulation with big time steps and tiny time steps to ensure the math stays stable and 
    "converges" on a consistent answer regardless of the step size'''
    dt_vals = [1, 0.5]
    ref_dt = 0.01
    t_ref = np.arange(0, T + ref_dt, ref_dt)
    
    # reference simulation
    m_ref, _, sims_ref = simulate_ensemble(params, V0, t_ref, M=20, seed=seed)
    ref_final_mean = np.mean(sims_ref[:, -1])
    
    conv_data = {"dt": [], "strong": [], "weak": [], "runtime": []}
    
    for dt in dt_vals:
        start_t = time.time()
        t_grid = np.arange(0, T + dt, dt)
        m_dt, _, sims_dt = simulate_ensemble(params, V0, t_grid, M=20, seed=seed)
        elapsed = time.time() - start_t
        
        conv_data["dt"].append(dt)
        conv_data["strong"].append(np.mean(np.abs(m_dt - np.interp(t_grid, t_ref, m_ref))))
        conv_data["weak"].append(np.abs(np.mean(sims_dt[:, -1]) - ref_final_mean))
        conv_data["runtime"].append(elapsed)
        
    return conv_data

# fitting
def fit_em_model(t_train, y_train, V0, seed):
    '''fits the EM model to the training data by optimizing the parameters to minimize the 
    difference between the log of predicted and observed tumour sizes'''
    obs_max = max(y_train)
    bounds = [(-0.5, 0.5), (-1.0, 1.0), (0.01, 2.0), (0, t_train[-1]), (obs_max*0.2, obs_max*5),
              (0, obs_max*10), (0.01, 2.0), (0, t_train[-1]), (0, 2.0), (1e-6, 1e5), (0, 0.8)]
    
    start = time.time()
    res_de = differential_evolution(lambda p: np.sum((np.log(simulate_ensemble(p, V0, t_train, M=10, seed=seed)[0]+1e-6) - np.log(y_train+1e-6))**2), 
                                    bounds, seed=seed, maxiter=5, polish=False)
    res_ls = least_squares(lambda p: np.log(simulate_ensemble(p, V0, t_train, M=10, seed=seed)[0]+1e-6) - np.log(y_train+1e-6), 
                           res_de.x, bounds=([b[0] for b in bounds], [b[1] for b in bounds]), max_nfev=50)
    return res_ls.x, {"fit_time": time.time() - start}

# main
if __name__ == '__main__':
    df = pd.read_csv(DATA_FILENAME)
    final_results = []

    for pid in tqdm(TESTING_PATIENTS, desc="EM Full Consistency Run"):
        p_df = df[df['Patient'].astype(str).str.contains(f"{pid:03d}")].copy()
        if p_df.empty: continue
        
        p_df['Day'] = p_df['Week'].apply(lambda w: float(re.findall(r'\d+', str(w))[0]) * 7)
        p_df = p_df.sort_values('Day')
        t_full, y_full = p_df['Day'].values, p_df['Total_mm3'].values
        
        mid = len(y_full) // 2
        t_train, y_train, V0 = t_full[:mid], y_full[:mid], y_full[0]
        
        # 1. fit
        fitted, info = fit_em_model(t_train, y_train, V0, pid)
        
        # 2. predict (Full range)
        t_fine = np.linspace(t_full[0], t_full[-1], 200)
        mean_pred, _, _ = simulate_ensemble(fitted, V0, t_fine, M=20, seed=pid, V_max=max(y_full)*5)
        y_interp = np.interp(t_full, t_fine, mean_pred)
        
        # 3. metrics and convergence
        conv = run_convergence_suite(fitted, V0, t_full[-1], pid)
        
        res = {
            'Patient': pid,
            'Method': 'Euler-Maruyama',
            'MASE': mean_absolute_scaled_error(y_full, y_interp),
            'KGE': kling_gupta_efficiency(y_full, y_interp),
            'NSE': nash_sutcliffe_efficiency(y_full, y_interp),
            'Chi2': chi_squared(y_full, y_interp),
            'TCT': calculate_tct(mean_pred, t_fine),
            'FitTime': info['fit_time'],
            'AvgStrongErr': np.mean(conv['strong']),
            'AvgWeakErr': np.mean(conv['weak']),
            'AvgConvRuntime': np.mean(conv['runtime']),
            't_fine': t_fine, 'y_pred': mean_pred, 't_full': t_full, 'y_full': y_full
        }
        final_results.append(res)
        
        #  Plots
        plt.figure(figsize=(8, 5))
        plt.scatter(t_full[:mid], y_full[:mid], color=COLOR_SEEN, label='Seen')
        plt.scatter(t_full[mid:], y_full[mid:], color=COLOR_UNSEEN, marker='x', label='Unseen')
        plt.plot(t_fine, mean_pred, color=COLOR_LINE, label='EM Predictive')
        plt.title(f"Patient {pid} Consistency Fit"); plt.legend(); plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/em_p{pid}.png"); plt.close()

    # Save  CSV
    pd.DataFrame(final_results).drop(['t_fine', 'y_pred', 't_full', 'y_full'], axis=1).to_csv(f"{OUTPUT_DIR}/em_consistent_metrics.csv", index=False)
    
    print(f"\nEM Analysis Complete. Consistent metrics saved to {OUTPUT_DIR}/em_consistent_metrics.csv")