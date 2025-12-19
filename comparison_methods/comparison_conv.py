import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from typing import List, Dict, Any


NUMERICAL_DIR = "numerical_stats"
NEURAL_DIR = "neuralstats"
PLOT_OUTPUT_DIR = "test_plots"
# Ensure output directory exists
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)


def read_num_summary(filename):
    """Reads numerical predictive summary CSVs."""
    path = os.path.join(NUMERICAL_DIR, filename)
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    if 'em' in filename.lower(): m = 'EM'
    elif 'milstein' in filename.lower(): m = 'Milstein'
    elif 'srk' in filename.lower(): m = 'SRK'
    else: m = 'Unknown'
    df['Method'] = m
    return df

def get_neuralode_df():
    """Parses NeuralODE metrics and convergence from text files."""
    s_path = os.path.join(NEURAL_DIR, "test_stats.txt")
    if not os.path.exists(s_path): return pd.DataFrame()
    with open(s_path, 'r') as f: content = f.read()
    rows = []
    pat = r"Patient Patient-(\d+).*?> MASE: ([\d\.\-]+) \| Chi2: ([\d\.\-e\+]+).*?> NSE:\s+([\d\.\-]+) \| KGE:\s+([\d\.\-]+).*?> Trajectory Runtime: ([\d\.\-]+)s"
    for match in re.finditer(pat, content, re.DOTALL):
        rows.append({
            'Patient': int(match.group(1)), 'MASE': float(match.group(2)), 
            'Chi2': float(match.group(3)), 'NSE': float(match.group(4)), 
            'KGE': float(match.group(5)), 'FitTime': float(match.group(6))
        })
    df_s = pd.DataFrame(rows)
    c_path = os.path.join(NEURAL_DIR, "test_conv.txt")
    if os.path.exists(c_path):
        with open(c_path, 'r') as f: content = f.read()
        rows_c = []
        pat_c = r"\[Patient-(\d+)\]: Conv Time=([\d\.\-]+),.*?Weak Err=([\d\.\-e\+]+),.*?Strong Err=([\d\.\-e\+]+), Runtime=([\d\.\-]+)s"
        for m in re.finditer(pat_c, content):
            rows_c.append({
                'Patient': int(m.group(1)), 
                'TrajectoryConvergenceTime': float(m.group(2)),
                'WeakError': float(m.group(3)), 
                'StrongError': float(m.group(4)), 
                'Runtime': float(m.group(5))
            })
        df_c = pd.DataFrame(rows_c)
        df_s = pd.merge(df_s, df_c, on='Patient', how='outer')
    df_s['Method'] = 'NeuralODE'
    return df_s

def load_data():
    """Consolidates and standardizes all convergence data."""
    dfs = [read_num_summary(f) for f in ['em_predictive_metrics_summary.csv', 
                                              'milstein_predictive_metrics_summary.csv', 
                                              'srk_predictive_metrics_summary.csv']]
    dfs.append(get_neuralode_df())
    master = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    master = master.loc[:, ~master.columns.duplicated()].copy()
    renames = {'FitTime': 'Overall_FitTime', 'Runtime': 'Total_Convergence_Runtime',
               'StrongError': 'Strong_Error_Metric', 'WeakError': 'Weak_Error_Metric'}
    master.rename(columns={k: v for k, v in renames.items() if k in master.columns}, inplace=True)
    metrics = ['Strong_Error_Metric', 'Weak_Error_Metric', 'Total_Convergence_Runtime', 'TrajectoryConvergenceTime']
    for col in metrics:
        if col in master.columns:
            series = master[col].astype(str).str.split('|').str[-1]
            master[col] = pd.to_numeric(series, errors='coerce')
    return master


def identify_and_clip(df_melted):
    """Applies IQR clipping for visual and statistical consistency."""
    processed = []
    for method in df_melted['Method'].unique():
        for metric in df_melted['Metric'].unique():
            mask = (df_melted['Method'] == method) & (df_melted['Metric'] == metric)
            subset = df_melted[mask].copy()
            vals = subset['Value'].dropna()
            if len(vals) >= 4:
                Q1, Q3 = vals.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                subset['Value'] = np.clip(subset['Value'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            processed.append(subset)
    return pd.concat(processed)

def do_analysis(df):
    """Main analysis loop: Plots and prints summary tables."""
    metrics_map = {
        'Strong_Error_Metric': 'Strong Convergence Error',
        'Weak_Error_Metric': 'Weak Convergence Error',
        'TrajectoryConvergenceTime': 'Trajectory Convergence Time',
        'Total_Convergence_Runtime': 'Convergence Runtime Comparison'
    }
    melted = df.melt(id_vars=['Method'], value_vars=[c for c in metrics_map.keys() if c in df.columns], var_name='M_Key', value_name='Value')
    melted['Metric'] = melted['M_Key'].map(metrics_map)
    clipped = identify_and_clip(melted)
    # Plots
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    target_order = ['Strong_Error_Metric', 'Weak_Error_Metric', 'TrajectoryConvergenceTime', 'Total_Convergence_Runtime']
    for i, col_key in enumerate(target_order):
        ax = axes.flatten()[i]
        label = metrics_map[col_key]
        subset = clipped[clipped['Metric'] == label]
        if subset.empty: continue
        sns.boxplot(x='Method', y='Value', hue='Method', data=subset, ax=ax, palette='Set2', fliersize=0)
        if ax.get_legend() is not None: ax.get_legend().remove()
        means = subset.groupby('Method')['Value'].mean()
        ax.scatter(np.arange(len(means)), means.values, marker='o', color='red', s=100, zorder=5)
        ax.set_title(label, fontsize=18, fontweight='bold')
        ax.set_ylabel("Clipped Value", fontsize=16)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.suptitle("Comparative Analysis of Model Convergence and Stability", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "formal_convergence_comparison.png"), dpi=300)
    plt.close()
    # RAW 
    print("\n" + "="*115 + "\nRAW CONVERGENCE SUMMARY (Full Cohort Mean ± Std)\n" + "="*115)
    raw_means = df.groupby('Method')[target_order].mean()
    raw_stds = df.groupby('Method')[target_order].std()
    raw_rows = []
    for mth in raw_means.index:
        entry = {'Method': mth}
        for m_col in target_order:
            mv, sv = raw_means.at[mth, m_col], raw_stds.at[mth, m_col]
            fmt = "{:.2e}" if (abs(mv) > 1000 or (0 < abs(mv) < 0.01)) else "{:.4f}"
            entry[m_col] = f"{fmt.format(mv)} ± {fmt.format(sv)}"
        raw_rows.append(entry)
    print(pd.DataFrame(raw_rows).to_string(index=False))
    # CLIPPED 
    print("\n" + "="*115 + "\nCLIPPED CONVERGENCE SUMMARY (Outliers Removed via IQR)\n" + "="*115)
    clipped_stats = clipped.pivot_table(index='Method', columns='M_Key', values='Value', aggfunc=['mean', 'std'])
    clipped_rows = []
    for mth in clipped_stats.index:
        entry = {'Method': mth}
        for m_col in target_order:
            mv = clipped_stats.loc[mth, ('mean', m_col)]
            sv = clipped_stats.loc[mth, ('std', m_col)]
            fmt = "{:.2e}" if (abs(mv) > 1000 or (0 < abs(mv) < 0.01)) else "{:.4f}"
            entry[m_col] = f"{fmt.format(mv)} ± {fmt.format(sv)}"
        clipped_rows.append(entry)
    print(pd.DataFrame(clipped_rows).to_string(index=False))

if __name__ == '__main__':
    data = load_data()
    if not data.empty:
        do_analysis(data)
        print(f"\nConvergence tables printed above.")