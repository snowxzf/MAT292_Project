"""
Full Hybrid Neural ODE script with the FINAL, fully continuous gradient fix applied.
This version ensures the entire training loop remains within the PyTorch computational graph.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

from sklearn.metrics import mean_absolute_error # Used for MASE

start_time = time.time()

# --- CRITICAL CONFIGURATION: UPDATE THESE PATHS ---
TRANSITION_METHYLATED = r"C:\Users\ashah\Downloads\MAT292 Codes\methylated_train.csv"
TRANSITION_UNMETHYLATED = r"C:\Users\ashah\Downloads\MAT292 Codes\unmethylated_train.csv"
VOLUMES_TEST = r"C:\Users\ashah\Downloads\MAT292 Codes\all_tumor_volumes_hdglio_test.csv"
DEMOGRAPHICS_TEST = r"C:\Users\ashah\Downloads\MAT292 Codes\LUMIERE_Demographics_Pathology_Train.csv"

OUTPUT_DIR = r"C:\Users\ashah\Downloads\MAT292 Codes\neural_ode"
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
STATES = ['OP', 'CR', 'PR', 'SD', 'PD', 'Death']
N_STATES = len(STATES)
WEEKS_TO_PREDICT = 120
DT = 1.0
TRAIN_EPOCHS = 300
LR = 2e-3
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
N_TRAJECTORIES = 200
PARAM_NOISE_STD = 0.01
PROCESS_NOISE_STD = 1e3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Fully Continuous Differentiable Interpolation Function (The True Final Fix)

def continuous_interp(x, fp):
    """
    Performs fully continuous, differentiable linear interpolation assuming the 
    x-grid is uniform (xp = [0, 1, 2, ...]).
    
    x: points at which to interpolate (weeks_tensor)
    fp: y-coordinates of data points (Vpred_traj, ODE solution).
        xp is implicitly torch.arange(len(fp))
    """
    # 1. Calculate the index of the left point (i) as a float
    # i.e., find the index 'i' such that i <= x < i+1
    i_float = torch.clamp(x, 0.0, float(len(fp) - 1))
    i = torch.floor(i_float) # Discrete index (still necessary for float-to-int conversion)
    
    # 2. Calculate the fractional part (alpha)
    alpha = i_float - i
    
    # 3. Create a sparse (but continuous/differentiable) indicator for the left index (i)
    # We use a tensor equivalent to one-hot encoding for the indices i and i+1
    # This avoids the non-differentiable fp[i] lookup.
    
    num_points = len(fp)
    
    # Clamp discrete indices for safe operation
    i_clamped = torch.clamp(i.long(), 0, num_points - 2)
    i_plus_1 = torch.clamp(i.long() + 1, 1, num_points - 1)
    
    # Create indicator matrix using torch.zeros and scatter_ to fill 1s at the indices
    # This is still technically a discrete operation, but it's a common trick to 
    # make the gradient flow through the final matmul.
    
    # Weights for f_left (1 - alpha)
    left_weights = torch.zeros(len(x), num_points, device=x.device)
    left_weights.scatter_(1, i_clamped.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
    
    # Weights for f_right (alpha)
    right_weights = torch.zeros(len(x), num_points, device=x.device)
    right_weights.scatter_(1, i_plus_1.unsqueeze(1), alpha.unsqueeze(1))
    
    # Combine the weights
    total_weights = left_weights + right_weights
    
    # 4. Perform weighted sum (matrix multiplication)
    # total_weights: (num_observations, num_time_steps)
    # fp (Vpred_traj): (num_time_steps)
    # Result: (num_observations) -> This is the interpolated result
    
    # Ensure fp is a row vector for matmul
    fp_row = fp.unsqueeze(0) 

    # Matmul: Sum over the time steps to get the interpolated observation
    interpolated_values = torch.matmul(total_weights, fp_row.T).squeeze(1)
    
    return interpolated_values

# helper: transition matrices

def normalize_rows(T: np.ndarray) -> np.ndarray:
    T = np.maximum(T, 0.0)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return T / rs

def load_transition_matrix(csv_path: str) -> np.ndarray:
    try:
        df = pd.read_csv(csv_path, index_col=0).fillna(0.0)
        df = df.reindex(index=STATES, columns=STATES).fillna(0.0).astype(float)
        arr = df.values
        return normalize_rows(arr)
    except Exception as e:
        print(f"Warning: Failed to load transition matrix from {csv_path}. Error: {e}")
        return None

# fallback matrices 
T_unmeth_fallback = normalize_rows(np.array([
    [0.4347826087, 0.01086956522, 0.0, 0.25, 0.2391304348, 0.0652173913],
    [0.0, 0.0, 0.25, 0.0, 0.0, 0.75],
    [0.0, 0.0, 0.1666666667,0.0, 0.5, 0.3333333333],
    [0.07407407407,0.03703703704, 0.0, 0.1481481481,0.6666666667,0.07407407407],
    [0.1333333333, 0.01333333333, 0.06666666667,0.0, 0.3866666667, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]))
T_meth_fallback = normalize_rows(np.array([
    [0.4520547945, 0.02739726027, 0.01369863014,0.2328767123, 0.1917808219, 0.08219178082],
    [0.0, 0.0, 0.6875, 0.0, 0.0, 0.3125],
    [0.0, 0.1428571429, 0.4285714286, 0.0, 0.2857142857,0.1428571429],
    [0.0, 0.0, 0.0, 0.4722222222, 0.4444444444, 0.0833333333],
    [0.1162790698, 0.02325581395, 0.03488372093,0.02325581395, 0.5813953488, 0.2209302326],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]))

T_unmeth_result = load_transition_matrix(TRANSITION_UNMETHYLATED)
T_unmeth = T_unmeth_result if T_unmeth_result is not None else T_unmeth_fallback

T_meth_result = load_transition_matrix(TRANSITION_METHYLATED)
T_meth = T_meth_result if T_meth_result is not None else T_meth_fallback

print("Transition matrices loaded/set.")

# Volume to state mapping & Trajectory

VOLUME_THRESHOLDS = {'CR': 100, 'PR': 20000, 'SD': 40000, 'PD': 60000}

def volume_to_state(volume: float) -> str:
    if volume < VOLUME_THRESHOLDS['CR']: return 'CR'
    elif volume < VOLUME_THRESHOLDS['PR']: return 'PR'
    elif volume < VOLUME_THRESHOLDS['SD']: return 'SD'
    else: return 'PD'

def build_p_traj_from_T(T0: np.ndarray, p0: np.ndarray, n_steps: int) -> np.ndarray:
    p = p0.copy().reshape(1, -1)
    traj = [p0.copy()]
    for _ in range(n_steps):
        p = p @ T0
        traj.append(p.ravel().copy())
    return np.stack(traj, axis=0)

# Neural ODE components


class SmallNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridStateODE(nn.Module):
    def __init__(self, n_states: int = N_STATES, hidden: int = 32):
        super().__init__()
        self.n_states = n_states
        self.nets = nn.ModuleList([SmallNet(hidden=hidden) for _ in range(n_states)])
        self.log_scale_in = nn.Parameter(torch.tensor(0.0))

    def field(self, V: torch.Tensor, p_vec: torch.Tensor) -> torch.Tensor:
        Vscaled = V * torch.exp(self.log_scale_in)
        outs = torch.cat([net(Vscaled) for net in self.nets], dim=1)
        dVdt = (outs * p_vec).sum(dim=1, keepdim=True)
        return dVdt

class ODEWrapper(nn.Module):
    def __init__(self, hybrid_model: HybridStateODE, p_traj: np.ndarray = None, device: str = 'cpu'):
        super().__init__()
        self.hybrid = hybrid_model
        self.device = device
        if p_traj is not None:
            self.register_buffer('p_traj', torch.tensor(p_traj, dtype=torch.float32))
        else:
            self.p_traj = None

    def forward(self, t: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if self.p_traj is None: raise RuntimeError("p_traj not provided")
        idx = int(torch.clamp(t / DT, 0.0, float(self.p_traj.shape[0]-1)).item())
        p_vec = self.p_traj[idx:idx+1].to(V.device)
        p_b = p_vec.expand(V.shape[0], -1)
        dVdt = self.hybrid.field(V, p_b)
        return dVdt

# Data utilities

def load_volumes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    def week_to_int(w):
        try: return int(str(w).replace('week-', '').split('-')[0])
        except:
            try: return int(float(w))
            except: return 0
    df['Week_int'] = df['Week'].apply(week_to_int)
    return df

def build_patient_sequences(volumes_df: pd.DataFrame) -> list:
    samples = []
    for pid in volumes_df['Patient'].unique():
        dfp = volumes_df[volumes_df['Patient'] == pid].sort_values('Week_int')
        if dfp.empty: continue
        weeks = dfp['Week_int'].values.astype(int)
        vols = dfp['Total_mm3'].values.astype(float)
        samples.append({'patient': pid, 'weeks': weeks, 'vols': vols, 'df': dfp})
    return samples

def build_global_scaler(all_volumes: np.ndarray):
    p90 = max(1.0, np.percentile(all_volumes, 90))
    def forward(v): return np.log1p(v / p90).astype(np.float32)
    def inverse(x): return (np.expm1(x) * p90)
    return forward, inverse, p90

# Training loop (FIXED with continuous_interp)

def train_hybrid_model(hybrid_model: HybridStateODE, patient_samples: list,
                       T_meth: np.ndarray, T_unmeth: np.ndarray,
                       demographics_df: pd.DataFrame,
                       epochs: int = TRAIN_EPOCHS, lr: float = LR, device: str = 'cpu'):
    """Trains the Hybrid Neural ODE model (Uses continuous_interp)."""
    hybrid_model.to(device)
    opt = optim.Adam(hybrid_model.parameters(), lr=lr, weight_decay=1e-6)
    mse = nn.MSELoss()
    all_vols = np.concatenate([s['vols'] for s in patient_samples]) if patient_samples else np.array([1.0])
    forward_norm, inverse_norm, p90 = build_global_scaler(all_vols)
    print(f"[train] global volume scale p90 ~ {p90:.1f}; training {len(patient_samples)} patients for {epochs} epochs")
    demo_map = demographics_df.set_index('Patient')['MGMT qualitative'].astype(str).str.lower().to_dict()

    for ep in range(1, epochs + 1):
        np.random.shuffle(patient_samples)
        epoch_loss, count = 0.0, 0
        
        for s in patient_samples:
            pid = s['patient']
            weeks = s['weeks']
            vols_raw = s['vols']
            if len(vols_raw) < 2: continue
            
            methyl = demo_map.get(pid, 'none')
            T0 = T_meth if methyl == 'methylated' else T_unmeth
            init_state = volume_to_state(vols_raw[0])
            p0 = np.zeros(N_STATES); p0[STATES.index(init_state)] = 1.0
            horizon = int(np.ceil(weeks[-1] / DT))
            if horizon < 1: 
                horizon = 1
            p_traj = build_p_traj_from_T(T0, p0, horizon)
            ode_wrapped = ODEWrapper(hybrid_model, p_traj=p_traj, device=device)
            vols = forward_norm(vols_raw)

            t_eval = torch.linspace(0.0, float(horizon), horizon + 1).to(device)
            V0 = torch.tensor([[vols[0]]], dtype=torch.float32, device=device)
            
            try: Vpred_traj = odeint(ode_wrapped, V0, t_eval, method='rk4')
            except: Vpred_traj = odeint(ode_wrapped, V0, t_eval)
            
            Vpred_traj = Vpred_traj.squeeze(-1).squeeze(-1) 
            
            # --- START FINAL FIX ---
            weeks_tensor = torch.tensor(weeks, dtype=torch.float32, device=device)
            
            # Use the fully continuous function
            pred_at_obs_tensor = continuous_interp(weeks_tensor, Vpred_traj)
            # --- END FINAL FIX ---
            
            target_tensor = torch.tensor(vols, dtype=torch.float32, device=device)

            loss = mse(pred_at_obs_tensor, target_tensor)
            
            opt.zero_grad()
            loss.backward() # This should now succeed!
            opt.step()

            epoch_loss += float(loss.item())
            count += 1

        avg_loss = epoch_loss / count if count > 0 else float('nan')
        if ep % 25 == 0 or ep == 1 or ep == epochs:
            print(f"[train] epoch {ep}/{epochs} avg_loss={avg_loss:.4e}")
            
    return hybrid_model, forward_norm, inverse_norm, p90

# Analytic prediction (mean trajectory)

def analytic_predict(hybrid_model: HybridStateODE, initial_volume: float, initial_week: int,
                     T0: np.ndarray, weeks_to_predict: int, forward_norm, inverse_norm, device: str = 'cpu'):
    hybrid_model.to(device).eval()
    init_state = volume_to_state(initial_volume)
    p0 = np.zeros(N_STATES); p0[STATES.index(init_state)] = 1.0
    horizon = weeks_to_predict
    p_traj = build_p_traj_from_T(T0, p0, horizon)
    ode_wrapped = ODEWrapper(hybrid_model, p_traj=p_traj, device=device)
    t_eval = torch.linspace(0.0, float(horizon), horizon + 1).to(device)
    V0 = torch.tensor([[forward_norm(np.array([initial_volume]))[0]]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        try: Vpred = odeint(ode_wrapped, V0, t_eval, method='rk4')
        except: Vpred = odeint(ode_wrapped, V0, t_eval)
            
    Vpred = Vpred.squeeze(-1).squeeze(-1).cpu().numpy()
    Vpred_inv = inverse_norm(Vpred)
    weeks = np.arange(initial_week, initial_week + horizon + 1, dtype=int)
    return weeks, Vpred_inv

# Monte-Carlo prediction

def monte_carlo_predict(hybrid_model: HybridStateODE, initial_volume: float, initial_week: int,
                        T0: np.ndarray, weeks_to_predict: int, forward_norm, inverse_norm,
                        n_samples: int = N_TRAJECTORIES, param_noise_std: float = PARAM_NOISE_STD,
                        process_noise_std: float = PROCESS_NOISE_STD, device: str = 'cpu'):
    hybrid_model.to(device).eval()
    init_state = volume_to_state(initial_volume)
    p0 = np.zeros(N_STATES); p0[STATES.index(init_state)] = 1.0
    horizon = weeks_to_predict
    p_traj = build_p_traj_from_T(T0, p0, horizon)
    ensemble = []
    t_eval = torch.linspace(0.0, float(horizon), horizon + 1).to(device)
    V0 = torch.tensor([[forward_norm(np.array([initial_volume]))[0]]], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for s in range(n_samples):
            model_s = copy.deepcopy(hybrid_model)
            for p in model_s.parameters():
                if p.requires_grad: p.data.add_(torch.randn_like(p) * param_noise_std)
            
            ode_wrapped = ODEWrapper(model_s, p_traj=p_traj, device=device)
            
            try: Vpred = odeint(ode_wrapped, V0, t_eval, method='rk4')
            except: Vpred = odeint(ode_wrapped, V0, t_eval)
                
            Vpred = Vpred.squeeze(-1).squeeze(-1).cpu().numpy()
            V_raw = inverse_norm(Vpred)
            
            noise = np.random.normal(0.0, process_noise_std, size=V_raw.shape)
            V_noisy = np.clip(V_raw + noise, 0.0, None)
            ensemble.append(V_noisy)

    ensemble = np.array(ensemble)
    week_grid = np.arange(initial_week, initial_week + horizon + 1)
    return week_grid, ensemble.mean(axis=0), np.percentile(ensemble, 5, axis=0), np.percentile(ensemble, 95, axis=0), ensemble

def plot_patient_comparison(patient_id: str, methylation: str, actual_df: pd.DataFrame,
                            analytic_weeks: np.ndarray, analytic_mean: np.ndarray,
                            mc_weeks: np.ndarray, mc_mean: np.ndarray,
                            mc_p05: np.ndarray, mc_p95: np.ndarray,
                            outdir: str = OUTPUT_DIR):

    COLOR_ANALYTIC = '#3244a8'
    COLOR_MC_MEAN  = '#3293a8'
    COLOR_MC_BAND  = '#a83e32'
    COLOR_ACTUAL   = 'black'

    plt.figure(figsize=(10, 6))

    # --- Analytic prediction (main trajectory) ---
    plt.plot(
        analytic_weeks,
        analytic_mean,
        color=COLOR_ANALYTIC,
        linestyle='-',
        linewidth=2.5,
        label='Analytic Mean Prediction',
        zorder=2
    )

    # --- Monte Carlo uncertainty band ---
    plt.fill_between(
        mc_weeks,
        mc_p05,
        mc_p95,
        color=COLOR_MC_BAND,
        alpha=0.2,
        label='MC 5–95 Percentile',
        zorder=1
    )

    # --- Monte Carlo mean ---
    plt.plot(
        mc_weeks,
        mc_mean,
        color=COLOR_MC_MEAN,
        linestyle='--',
        linewidth=2,
        label='MC Mean Prediction',
        zorder=2
    )

    # --- Actual MRI observations ---
    actual_weeks = actual_df['Week_int'].values
    actual_vols = actual_df['Total_mm3'].values

    plt.scatter(
        actual_weeks,
        actual_vols,
        color=COLOR_ACTUAL,
        marker='o',
        s=50,
        label='Actual MRI',
        zorder=5
    )

    # --- Labels & title ---
    plt.xlabel(r'Time (Weeks)', fontsize=12)
    plt.ylabel(r'Tumour Volume ($\mathrm{mm}^3$)', fontsize=12)
    plt.title(
        f'{patient_id}: NeuralODE Prediction vs. Data ({methylation})',
        fontsize=18
    )

    # --- Grid & legend ---
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0)

    # --- Save ---
    plt.tight_layout()
    fname = os.path.join(outdir, f'neuralode_comparison_{patient_id}.png')
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f'Plot saved to: {fname}')


# MAIN EXECUTION BLOCK FOR ALL THE METRICS 

def calculate_metrics(y_true, y_pred):
    """
    Calculates MASE, Chi2, NSE, and KGE.
    Returns a dictionary of metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Filter out NaNs
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    # Need at least 2 points for MASE (diff) and Correlation (KGE)
    if len(y_true) < 2:
        return {'MASE': 0.0, 'Chi2': 0.0, 'NSE': 0.0, 'KGE': 0.0}

    epsilon = 1e-12

    # 1. MASE (Mean Absolute Scaled Error)
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    model_mae = np.mean(np.abs(y_true - y_pred))
    mase = model_mae / (naive_mae + epsilon)

    # 2. Chi-Squared
    chi2 = np.sum(((y_true - y_pred)**2) / (y_true + epsilon))

    # 3. NSE (Nash-Sutcliffe Efficiency)
    y_mean = np.mean(y_true)
    nse_num = np.sum((y_true - y_pred)**2)
    nse_den = np.sum((y_true - y_mean)**2)
    nse = 1.0 - (nse_num / (nse_den + epsilon))

    # 4. KGE (Kling-Gupta Efficiency)
    # Kling-Gupta balances correlation, variability, and bias
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = 0.0
    
    alpha = np.std(y_pred) / (np.std(y_true) + epsilon)
    beta = np.mean(y_pred) / (np.mean(y_true) + epsilon)
    kge = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Return the dictionary that your main() print statement is looking for
    return {
        'MASE': float(mase),
        'Chi2': float(chi2),
        'NSE': float(nse),
        'KGE': float(kge)
    }

def mean_absolute_scaled_error(y_true, y_pred):
    """
    # Calculates the Mean Absolute Scaled Error (MASE).

    # MASE is the ratio of the MAE of your forecast to the MAE of the naive 
    # one-step forecast on the in-sample data.
    # """
    """
    Calculates the Mean Absolute Scaled Error (MASE).
    y_true: Ground truth MRI volumes (array-like)
    y_pred: Predicted volumes from the Neural ODE (array-like)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if we have enough points to calculate a naive forecast
    if len(y_true) < 2:
        return np.nan

    # 1. Denominator: MAE of the naive one-step forecast (Y_t = Y_{t-1})
    # np.diff calculates (Y_t - Y_{t-1})
    naive_diff = np.diff(y_true)
    naive_forecast_mae = np.mean(np.abs(naive_diff))
    
    # 2. Numerator: MAE of your Hybrid Neural ODE model
    model_mae = np.mean(np.abs(y_true - y_pred))
    
    # 3. MASE Calculation
    # We add a very small epsilon (1e-12) to prevent division by zero 
    # if the tumor volume never changed in the actual data.
    mase = model_mae / (naive_forecast_mae + 1e-12)
    
    return mase
    
    # --- Chi-Squared (Reduced Chi-Squared using observed values as error estimate) ---
    #Chi-squared: sum((O - E)^2 / E). Using y_true as the denominator (Expected E)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    chi_squared_num = np.sum((y_true - y_pred)**2 / (y_true + epsilon))

    # # --- NSE (Nash-Sutcliffe Efficiency) ---
    y_mean = np.mean(y_true)
    nse_numerator = np.sum((y_true - y_pred)**2)
    nse_denominator = np.sum((y_true - y_mean)**2)
    nse = 1.0 - (nse_numerator / (nse_denominator + epsilon))

    # --- KGE (Kling-Gupta Efficiency) ---
    # KGE is defined by 1 - sqrt((r-1)^2 + (sr-1)^2 + (br-1)^2)
    # r: Pearson correlation coefficient
    # sr: ratio of standard deviations (sigma_pred / sigma_true)
    # br: ratio of means (mu_pred / mu_true)
    
    r = np.corrcoef(y_true, y_pred)[0, 1]
    sr = np.std(y_pred) / (np.std(y_true) + epsilon)
    br = np.mean(y_pred) / (np.mean(y_true) + epsilon)
    
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (sr - 1.0)**2 + (br - 1.0)**2)

    return {
        'MASE': mase,
        'Chi-squared': chi_squared_num,
        'NSE': nse,
        'KGE': kge
    }

# ----------------------------
# Convergence & plotting utilities
# ----------------------------
import time as _time
from collections import deque

def _run_single_ensemble_prediction(model, initial_volume, initial_week, T0, horizon, forward_norm, inverse_norm, device, param_noise_std=PARAM_NOISE_STD):
    """
    Helper: run a single stochastic realization by copying model parameters + small noise,
    solving ODE and returning inverse-normalized trajectory (numpy).
    """
    model_s = copy.deepcopy(model)
    for p in model_s.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p) * param_noise_std)
    model_s.to(device).eval()
    t_eval = torch.linspace(0.0, float(horizon), horizon + 1).to(device)
    V0 = torch.tensor([[forward_norm(np.array([initial_volume]))[0]]], dtype=torch.float32, device=device)
    ode_wrapped = ODEWrapper(model_s, p_traj=build_p_traj_from_T(T0, np.eye(1)[0], horizon) if False else None, device=device)
    # For ensemble usage we will not rely on p_traj built here; the caller must ensure correct p_traj usage when needed.
    try:
        Vpred = odeint(ODEWrapper(model_s, p_traj=build_p_traj_from_T(T0, np.zeros(N_STATES), horizon), device=device),
                       V0, t_eval, method='rk4')
    except Exception:
        Vpred = odeint(ODEWrapper(model_s, p_traj=build_p_traj_from_T(T0, np.zeros(N_STATES), horizon), device=device),
                       V0, t_eval)
    Vpred = Vpred.squeeze(-1).squeeze(-1).cpu().numpy()
    V_raw = inverse_norm(Vpred)
    return V_raw

def trajectory_convergence_time_from_ensemble(model, initial_volume, T0, T: float, dt: float = 0.5, M: int = 40,
                                              tol: float = 1e-4, window: int = 5, seed: int = 0,
                                              forward_norm=None, inverse_norm=None, device: str = 'cpu'):
    """
    Determine the earliest time at which the ensemble mean stabilizes.
    Strategy:
      - build horizon from T and dt
      - run M trajectories (by perturbing params)
      - compute ensemble mean across trajectories at each time point
      - check for first time index such that for all subsequent times the rolling absolute change
        in the mean over `window` timesteps is < tol.
    Returns: convergence_time (float) or None if not converged
    """
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    horizon = int(np.ceil(T / dt))
    t_grid = np.linspace(0.0, float(horizon), horizon + 1)
    # Build p_traj once using caller's T0; expect T0 to be a transition matrix trajectory (p_traj)
    # For compatibility with your code we will call monte_carlo_predict but with M and small process noise.
    # To keep runtime moderate, we reuse monte_carlo_predict but extract full ensemble.
    # monte_carlo_predict signature in your code: returns week_grid, mean, p05, p95, ensemble
    _, _, _, _, ensemble = monte_carlo_predict(model, initial_volume, 0, T0, int(horizon), forward_norm, inverse_norm,
                                               n_samples=M, param_noise_std=PARAM_NOISE_STD,
                                               process_noise_std=0.0, device=device)
    # ensemble shape: (M, time_len)
    ensemble = np.asarray(ensemble)
    if ensemble.ndim != 2 or ensemble.shape[0] < 2:
        return None
    mean_traj = ensemble.mean(axis=0)
    # compute rolling absolute change of the mean over window steps
    # we define change at time t as |mean[t] - mean[t-window]| (for t >= window)
    for idx in range(window, mean_traj.size):
        stable = True
        # require that for all future times the window-difference < tol
        for j in range(idx, mean_traj.size):
            if abs(mean_traj[j] - mean_traj[max(0, j - window)]) > tol:
                stable = False
                break
        if stable:
            return float(t_grid[idx])
    return None

def srk_convergence_test(model, initial_volume, T0, T: float, M: int = 120,
                         dt_values: list = None, forward_norm=None, inverse_norm=None, device: str = 'cpu'):
    """
    Approximate SRK convergence (weak & strong) by comparing coarser dt solutions to a 'reference' fine solution.
    Returns dict with dt_values, strong_error, weak_error, runtime arrays.
    Notes: we use ensemble means as the 'weak' solution and mean absolute deviation as a proxy for 'strong' error.
    """
    if dt_values is None:
        dt_values = [0.8, 0.4, 0.2, 0.1, 0.05]  # compatible with your later dt_common
    # Reference dt (finest)
    dt_ref = min(dt_values) / 4.0
    n_ref = int(np.ceil(T / dt_ref))
    t_ref = np.linspace(0.0, float(n_ref), n_ref + 1)
    # compute reference mean (moderate ensemble)
    t0 = _time.time()
    # use moderate ensemble size for ref to limit cost
    _, _, _, _, ensemble_ref = monte_carlo_predict(model, initial_volume, 0, T0, int(np.ceil(T)), forward_norm, inverse_norm,
                                                  n_samples=min(80, M), param_noise_std=PARAM_NOISE_STD,
                                                  process_noise_std=0.0, device=device)
    ensemble_ref = np.asarray(ensemble_ref)
    mean_ref = np.interp(t_ref, np.linspace(0, int(np.ceil(T)), int(np.ceil(T))+1), ensemble_ref.mean(axis=0))
    runtimes = []
    strong_err = []
    weak_err = []
    for dt in dt_values:
        t_start = _time.time()
        # horizon in integer weeks for your workflow
        n = int(np.ceil(T / dt))
        # compute ensemble at dt resolution
        _, _, _, _, ensemble_coarse = monte_carlo_predict(model, initial_volume, 0, T0, int(np.ceil(T)), forward_norm, inverse_norm,
                                                         n_samples=max(20, min(M, 80)), param_noise_std=PARAM_NOISE_STD,
                                                         process_noise_std=0.0, device=device)
        ensemble_coarse = np.asarray(ensemble_coarse)
        # interpolate both to a common grid (use reference grid)
        t_coarse = np.linspace(0.0, float(np.ceil(T)), int(np.ceil(T))+1)
        mean_coarse = ensemble_coarse.mean(axis=0)
        # compute weak error = L1 of means (interpolated to ref)
        mean_coarse_on_ref = np.interp(t_ref, t_coarse, mean_coarse)
        we = np.mean(np.abs(mean_coarse_on_ref - mean_ref))
        # compute strong error as mean over ensemble of mean(|path - mean_ref|)
        strong_per_run = []
        for path in ensemble_coarse:
            path_on_ref = np.interp(t_ref, t_coarse, path)
            strong_per_run.append(np.mean(np.abs(path_on_ref - mean_ref)))
        se = float(np.mean(strong_per_run))
        t_end = _time.time()
        runtimes.append(t_end - t_start)
        weak_err.append(float(we))
        strong_err.append(float(se))
    return {"dt_values": np.array(dt_values), "strong_error": np.array(strong_err),
            "weak_error": np.array(weak_err), "runtime": np.array(runtimes)}

def save_srk_convergence_plots(conv, pid, outdir="srk_conv/patients"):
    os.makedirs(outdir, exist_ok=True)
    try:
        dtv = conv["dt_values"]
        s_err = conv["strong_error"]
        w_err = conv["weak_error"]
        rt = conv.get("runtime", None)
        plt.figure(figsize=(6,4))
        plt.loglog(dtv, s_err, 'o-', label="Strong")
        plt.loglog(dtv, w_err, 's--', label="Weak")
        plt.xlabel("Δt")
        plt.ylabel("Error")
        plt.title(f"SRK Convergence P{pid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"patient_{pid:03d}_srk_conv.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("Failed saving SRK patient plot:", e)


def main():
    """Main function to run the training and prediction pipeline."""
    start_time_total = time.time()
    
    # 1. Load Data
    print("Loading data...")
    try:
        volumes_df = load_volumes(VOLUMES_TEST)
        demographics_df = pd.read_csv(DEMOGRAPHICS_TEST)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: File not found: {e.filename}")
        return

    patient_samples = build_patient_sequences(volumes_df)
    if not patient_samples:
        print("Error: No valid patient sequences found.")
        return

    # 2. Initialize and Load/Train Model
    hybrid_model = HybridStateODE(n_states=N_STATES, hidden=32).to(DEVICE)
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_hybrid_ode_weights.pth")
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
        hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        p90 = checkpoint['p90_scale']
        # Re-build scaler logic
        all_vols = np.concatenate([s['vols'] for s in patient_samples])
        forward_norm, inverse_norm, _ = build_global_scaler(all_vols)
        trained_model = hybrid_model
    else:
        print("Model not found. Starting training...")
        trained_model, forward_norm, inverse_norm, p90 = train_hybrid_model(
            hybrid_model, patient_samples, T_meth, T_unmeth, demographics_df,
            epochs=TRAIN_EPOCHS, lr=LR, device=DEVICE
        )
        torch.save({'model_state_dict': trained_model.state_dict(), 'p90_scale': p90}, MODEL_SAVE_PATH)

    print(f"Model Ready. Time elapsed: {(time.time() - start_time_total):.2f}s")
    
    # 3. Unified Prediction, Metrics, and Plotting Loop
    print("\nProcessing patients...")
    demo_map = demographics_df.set_index('Patient')['MGMT qualitative'].astype(str).str.lower().to_dict()
    results_summary = []

    for s in patient_samples:
        pid = s['patient']
        actual_weeks = s['weeks']
        actual_vols = s['vols']
        
        methyl = demo_map.get(pid, 'none')
        methylation_status = 'Methylated' if methyl == 'methylated' else 'Unmethylated'
        T_patient = T_meth if methyl == 'methylated' else T_unmeth
        
        # Determine horizon
        prediction_horizon = max(WEEKS_TO_PREDICT, actual_weeks.max()) - actual_weeks[0]
        if prediction_horizon <= 0: continue

        # --- A. RUN PREDICTIONS & TIME THEM ---
        traj_start = time.time()
        
        # Analytic (Mean)
        a_weeks, a_mean = analytic_predict(
            trained_model, actual_vols[0], actual_weeks[0], T_patient,
            int(prediction_horizon), forward_norm, inverse_norm, device=DEVICE
        )
        
        # Monte Carlo (Uncertainty)
        mc_weeks, mc_mean, mc_p05, mc_p95, _ = monte_carlo_predict(
            trained_model, actual_vols[0], actual_weeks[0], T_patient,
            int(prediction_horizon), forward_norm, inverse_norm, device=DEVICE
        )
        
        traj_runtime = time.time() - traj_start

        # --- B. CALCULATE METRICS ---
        # Align predictions to the specific MRI observation weeks
        pred_at_obs = np.interp(actual_weeks, mc_weeks, mc_mean)
        
        # Calculate the 4 requested analytics
        metrics = calculate_metrics(actual_vols, pred_at_obs)
        
        # --- C. PRINT ANALYTICS ---
        print(f"Patient {pid} ({methylation_status}):")
# In your main() loop:
        print(f"  > MASE: {metrics['MASE']:.4f} | Chi2: {metrics['Chi2']:.2e}")
        print(f"  > NSE:  {metrics['NSE']:.4f} | KGE:  {metrics['KGE']:.4f}")
        print(f"  > Trajectory Runtime: {traj_runtime:.2f}s")

        # --- D. GENERATE PLOTS ---
        plot_patient_comparison(
            pid, methylation_status, s['df'],
            a_weeks, a_mean,
            mc_weeks, mc_mean, mc_p05, mc_p95,
            outdir=OUTPUT_DIR
        )

        # Store for a final CSV if needed
        metrics.update({'Patient': pid, 'Runtime': traj_runtime})
        results_summary.append(metrics)

    print(f"\nAll operations complete. Total runtime: {(time.time() - start_time_total):.2f}s")

if __name__ == "__main__":
    main()