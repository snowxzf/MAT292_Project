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
    """Parses NeuralODE metrics from text files."""
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
    df_s['Method'] = 'NeuralODE'
    return df_s

def load_data():
    """Consolidates and prepares all model data into a single DataFrame."""
    dfs = [read_num_summary(f) for f in [
        'em_predictive_metrics_summary.csv', 
        'milstein_predictive_metrics_summary.csv', 
        'srk_predictive_metrics_summary.csv'
    ]]
    dfs.append(get_neuralode_df())
    master = pd.concat([d for d in dfs if not d.empty], ignore_index=True)
    master = master.loc[:, ~master.columns.duplicated()].copy()
    master.rename(columns={'FitTime': 'Overall_FitTime'}, inplace=True)
    metrics = ['MASE', 'Chi2', 'NSE', 'KGE', 'Overall_FitTime']
    for col in metrics:
        if col in master.columns:
            series = master[col].astype(str).str.split('|').str[0]
            master[col] = pd.to_numeric(series, errors='coerce')
    return master


def identify_and_clip(df_melted):
    """Performs per-method IQR clipping and identifies specific outliers."""
    processed = []
    outlier_log = [] #  to track clipped patients
    for method in df_melted['Method'].unique():
        for metric in df_melted['Metric'].unique():
            mask = (df_melted['Method'] == method) & (df_melted['Metric'] == metric)
            subset = df_melted[mask].copy()
            vals = subset['Value'].dropna()
            if len(vals) >= 4:
                Q1, Q3 = vals.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                #  which patients are outside these bounds
                outliers = subset[(subset['Value'] < lower_bound) | (subset['Value'] > upper_bound)]
                for _, row in outliers.iterrows():
                    outlier_log.append({
                        'Method': method, 
                        'Metric': metric, 
                        'Patient': row.get('Patient', 'Unknown'),
                        'Value': row['Value'],
                        'IQR_Bound': f"[{lower_bound:.2e}, {upper_bound:.2e}]"
                    })
                subset['Value'] = np.clip(subset['Value'], lower_bound, upper_bound)
            processed.append(subset)
    return pd.concat(processed), pd.DataFrame(outlier_log)

def do_analysis(df):
    """Generates formal paper-style comparison table, box plots, and outlier report."""
    metrics_map = {
        'MASE': 'Mean Absolute Scaled Error (MASE)', 
        'Chi2': 'Chi-Squared ($\chi^2$)', 
        'NSE': 'Nash-Sutcliffe Efficiency (NSE)', 
        'KGE': 'Kling-Gupta Efficiency (KGE)'
    }
    # Patient ID 
    melted = df.melt(id_vars=['Method', 'Patient'], value_vars=[c for c in metrics_map.keys() if c in df.columns], 
                     var_name='M_Key', value_name='Value')
    melted['Metric'] = melted['M_Key'].map(metrics_map)
    clipped, outlier_report = identify_and_clip(melted)
    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    target_order = ['MASE', 'Chi2', 'NSE', 'KGE']
    for i, col_key in enumerate(target_order):
        ax = axes.flatten()[i]
        label = metrics_map[col_key]
        subset = clipped[clipped['Metric'] == label]
        if subset.empty: continue
        sns.boxplot(x='Method', y='Value', hue='Method', data=subset, ax=ax, palette='Set2', fliersize=0)
        if ax.get_legend() is not None: ax.get_legend().remove()
        m_vals = subset.groupby('Method')['Value'].mean().reindex(sorted(subset['Method'].unique()))
        ax.scatter(np.arange(len(m_vals)), m_vals.values, marker='o', color='red', s=100, label='Mean', zorder=5)
        ax.set_title(label, fontsize=18, fontweight='bold')
        ax.set_ylabel("Clipped Value", fontsize=16)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.suptitle("Comparative Analysis of Predictive Model Performance", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "formal_metrics_comparison.png"), dpi=300)
    plt.close()
    # Outlier  
    if not outlier_report.empty:
        print("\n" + "!"*110 + "\nOUTLIER IDENTIFICATION REPORT (Clipped Patients)\n" + "!"*110)
        print(outlier_report.sort_values(['Method', 'Metric']).to_string(index=False))
        print("!"*110)
    # Tables
    agg_cols = ['MASE', 'Chi2', 'NSE', 'KGE', 'Overall_FitTime']
    available = [m for m in agg_cols if m in df.columns]
    means, stds = df.groupby('Method')[available].mean(), df.groupby('Method')[available].std()
    print("\nRAW PERFORMANCE SUMMARY (Full Cohort Mean ± Std)")
    report_rows = []
    for method in means.index:
        entry = {'Method': method}
        for m in available:
            mv, sv = means.at[method, m], stds.at[method, m]
            fmt = "{:.2e}" if abs(mv) > 1000 or (0 < abs(mv) < 0.001) else "{:.4f}"
            entry[m] = f"{fmt.format(mv)} ± {fmt.format(sv)}"
        report_rows.append(entry)
    print(pd.DataFrame(report_rows).to_string(index=False))
    table_clipped = clipped.pivot_table(index='Method', columns='M_Key', values='Value', aggfunc=['mean', 'std'])
    print("\nCLIPPED PERFORMANCE SUMMARY (Outliers Removed via IQR)")
    clipped_rows = []
    for method in table_clipped.index:
        entry = {'Method': method}
        for m_key in target_order:
            if m_key in table_clipped.columns.get_level_values(1):
                mv, sv = table_clipped.loc[method, ('mean', m_key)], table_clipped.loc[method, ('std', m_key)]
                fmt = "{:.2e}" if abs(mv) > 1000 or (0 < abs(mv) < 0.001) else "{:.4f}"
                entry[m_key] = f"{fmt.format(mv)} ± {fmt.format(sv)}"
        clipped_rows.append(entry)
    print(pd.DataFrame(clipped_rows).to_string(index=False))


if __name__ == '__main__':
    data = load_data()
    if not data.empty:
        do_analysis(data)