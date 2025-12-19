# MAT292_Project

# Glioblastoma Tumor Growth: Numerical & Comparative Analysis

This repository contains the implementation of various **Stochastic Differential Equation (SDE)** solvers and a comparative analysis framework to evaluate tumor growth forecasting. It compares classical numerical methods against a **Neural ODE** approach using longitudinal MRI data from the LUMIERE dataset.

## Segmentation Process and Extracting Tumour Volumes 
These tumour volumes are used for the rest of the numerical and neural models, thus it is important that this was done first. However this may take up to hours to fully finish, thus we recommend using the completed files "all_tumor_volumes_hdglio_test.csv" and "all_tumor_volumes_hdglio_train.csv". 

The following packages are necessary for successful segmentation (linux):
- HD-GLIO, nnunet, nibabel,numpy, pandas, scipy,SimpleITK, pyradiomics collageradiomics, FSL, FreeSurfer, torch, tqdm

To complete MRI segmentation, run "runhdglio.sh"  and "run.py", ensure the directories in the script are changed to the correct ones with complete patient data. 
**Usage**:
```bash
bash runhdglio.sh
```
It should output: 
- Registered MRI sequences
- Tumor segmentation masks
- Volume measurements
- Registration matrices

The bash files "create_volumes.sh" and "extract_volumes.sh" were used to extract and put these volumes all into one file for easier access for numerical and neural models. 

## Neural Model 

1. Markov State Transition Matrices

   

3. Hybrid Neural Model 
## Numerical Overview
The project implements a **Stochastic Logistic Growth Model** to predict tumor volume trajectories. It accounts for biological randomness using Monte Carlo ensembles and compares three numerical integration schemes of varying complexity:
1. **Euler-Maruyama (EM)** - First-order baseline (Strong order 0.5).
2. **Milstein Method** - Second-order Itô correction (Strong order 1.0).
3. **Strong Runge-Kutta (SRK)** - Derivative-free higher-order scheme (Strong order 1.0).

---

## Requirements

Ensure you have Python 3.8+ installed. You can install the necessary libraries using:

```bash
pip install numpy pandas matplotlib seaborn scipy tqdm
```
## Project Structure
1. SDE Numerical Solvers (Predictive Runs): 

The repository distinguishes between "Training" and "Predictive" (Testing) scripts to evaluate model generalization:
Training Scripts (*_training.py): These process the 80-patient training cohort. They are used to generate the Aggregate Convergence and Runtime Scaling plots. This phase establishes the theoretical discretization error (Strong/Weak convergence orders) for each solver.

Predictive Scripts (*predictive.py): These evaluate the 10-patient test set using a 50/50 temporal split.Calibration: The solver uses the first 50% of a patient's timepoints to estimate parameters ($\rho, \sigma, K$) via Differential Evolution.Forecasting: The solver then simulates the remaining 50% of the timeline to measure predictive accuracy against the "unseen" MRI observations.


empredictive.py: Implementation of the Euler-Maruyama method (strong order 0.5).
milsteinpredictive.py: Implementation of the Milstein method (strong order 1.0).
rkpredictive.py: Implementation of the Strong Runge-Kutta (SRK) method (derivative-free order 1.0).


2. Training & Visualization Scripts: Used for processing the 80-patient training set and generating aggregate convergence and scaling plots across various discretization steps ($\Delta t$).emplots_training.pymilstein_plots_trainingset.pyrk_trainingset.py3.

Note: The tumour_data.csv file (derived from the LUMIERE dataset) is already included within the branched folders of this repository. You do not need to provide external volumetric data to run the solvers.
```bash
.
├── tumour_data.csv          # Pre-processed volumetric data for all patients
├── numerical_stats/         # Auto-generated: Aggregated CSV metrics (MASE, Chi2, etc.)
├── neuralstats/             # Expected input: Place 'test_stats.txt' here
├── test_plots/              # Auto-generated: Final paper-style comparison figures
├── [method]_predictive_plots/ # Auto-generated: Individual patient trajectory fits
└── [method]_conv/           # Auto-generated: Discretization error & runtime scaling plots
```
After execution, the following directories will be populated with results:

Trajectory Fits: milstein_predictive_plots/, srk_predictive_plots/, and em_predictive_plots/ will contain PNG files for every patient. These show the 50/50 split, the ensemble mean, and the 95% confidence intervals derived from the Monte Carlo runs.

Performance Metrics: test_plots/ contains the final box plots. These visualizations specifically utilize IQR Clipping to remove statistical outliers (like Patient 43 and 77), allowing for a clear visual comparison of the "typical" performance across all four models.

Stability is evaluated via Trajectory Convergence Time (TCT). The scripts calculate the earliest week $t^*$ where the relative change in the ensemble mean stays below a tolerance of $10^{-4}$ for at least 5 consecutive weeks. This metric is used to contrast the rapid stabilization of Neural ODEs against the high-volatility tail behavior of the EM baseline.

## Verification Script
To verify the numerical implementation without running the full cohort, use the standalone script:
1. Ensure `tumour_data.csv` is in the root directory.
2. Run `python verify_patient_67.py`.
3. Check the `verification_results/` folder for the summary CSV and comparison plot.
**How it differs from the main solvers:**
- **Speed:** Uses **NumPy Vectorization** to solve the Stochastic Differential Equations. It processes the 60-path Monte Carlo ensemble 10x faster than the standard iterative scripts. Therefore, keep in mind that the shape of the graph is different than what the report used.
- **Scope:** While the main scripts (`empredictive.py`, etc.) process the full 10-patient test cohort, this script targets a single representative patient to allow for rapid verification of the numerical logic.
Each numerical script (empredictive.py, etc.) generates a summary CSV in the numerical_stats/ folder. Below is a breakdown of what the columns represent:
1. Identification & PerformancePatient / Method: The patient ID and the specific SDE solver used.
2. MASE (Mean Absolute Scaled Error): The primary accuracy metric. A value near 1.0 means the model is roughly as accurate as a "naive" guess. Lower is better.
3. Chi2 ($\chi^2$): Measures the structural deviation of the fit from clinical data. High values indicate a divergence in growth trends.
4. NSE / KGE: Efficiency metrics where 1.0 is a perfect fit. Negative values indicate that the model mean is less predictive than the historical average.FitTime / ParamChange: Tracks the computational cost of the two-stage calibration (Differential Evolution + Least Squares).
5. Stability & ConvergenceTrajectoryConvergenceTime (TCT): The time point (in weeks) where the tumor volume stabilizes within a $10^{-4}$ tolerance.
6. Conv_dt: The refinement grid used for discretization analysis (e.g., 0.8|0.4|0.2|0.1|0.05).
7. StrongError / WeakError: These columns contain multiple values separated by a pipe (|). Each value corresponds to the discretization steps in Conv_dt.Strong Error: Measures path-wise accuracy. It should decrease linearly as $\Delta t$ gets smaller. Weak Error: Measures the accuracy of the statistical mean.
8. Runtime: The wall-clock time (seconds) taken to simulate the ensemble at each refinement level.

**Example Data Interpretation:** In the provided example for Patient 31:MASE (0.97): Indicates a very strong predictive fit on the unseen test data.TCT (1330.0): Shows that this specific simulation did not reach a stable plateau within the clinical window (often signifying active late-stage growth).StrongError (2523|2325|...|1915): Shows a steady decrease in error as the time step is refined from $0.8$ down to $0.05$, proving numerical stability.
## Final Comparative Analysis
These scripts consolidate numerical results with Neural ODE outputs to generate the formal paper-style visualizations and tables used in the report.

1. comparison_metrics.py: Generates comparative box plots for MASE, $\chi^2$, NSE, and KGE.
2. comparison_conv.py: Generates Strong/Weak error and Trajectory Convergence Time (TCT) comparisons.

## How to Use
1. Prepare DataPlace your tumour_data.csv in the root directory.Ensure your Neural ODE results are saved in a directory named neuralstats/ (specifically neuralstats/test_stats.txt).
2. Run Numerical Simulations
3. Execute the predictive scripts for each method to generate per-patient statistics:
  ```bash
python milsteinpredictive.py
python rkpredictive.py
python empredictive.py
```
Outputs: CSV summaries in numerical_stats/ and trajectory plots in method-specific folders (e.g., milstein_predictive_plots/).
4. Run Comparative AnalysisOnce the numerical stats are generated and neural stats are in place, run the comparison scripts:
```bash
python comparison_metrics.py
python comparison_conv.py
```
Outputs: Professional box plots and aggregate summary tables saved to the test_plots/ directory.
## Methodology Summary
Monte Carlo Ensembles: All numerical solvers run $M=60$ independent trajectories per patient to rigorously account for non-linear stochasticity.

IQR Clipping: The analysis scripts perform automatic Interquartile Range clipping to ensure aggregate means reflect the "typical" cohort performance without being skewed by catastrophic outliers (e.g., Patients 43 and 77).

Convergence Analysis: Stability is quantified via Trajectory Convergence Time (TCT), identifying the week at which the ensemble mean reaches a stable state within a $10^{-4}$ tolerance threshold.

## Repository Authors
Jessica Fu, Sara Parvaresh Rizi, and Aarya Shah University of Toronto | MAT292 Final Project GitHub: https://github.com/snowxzf/MAT292_Project
