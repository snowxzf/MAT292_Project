# Glioblastoma Tumour Growth: Numerical & Comparative Analysis

This repository contains the implementation of various **Stochastic Differential Equation (SDE)** solvers and a comparative analysis framework to evaluate tumour growth forecasting. It compares classical numerical methods against a **Neural ODE** approach using longitudinal MRI data from the LUMIERE dataset.
NOTE: For grading the MAT292 Project, for numerical methods, please watch for the section that is explicitly for TA testing.

## Segmentation Process and Extracting Tumour Volumes 
These tumour volumes are used for the rest of the numerical and neural models, thus it is important that this was done first. However this may take up to hours to fully finish, thus we recommend using the completed files "all_tumor_volumes_hdglio_test.csv" and "all_tumor_volumes_hdglio_train.csv". 

The following packages are necessary for successful segmentation (linux):
- HD-GLIO, nnU-Net, nibabel,numpy, pandas, scipy,SimpleITK, pyradiomics collageradiomics, FSL, FreeSurfer, torch, tqdm

To complete MRI segmentation, run "runhdglio.sh"  and "run.py", ensure the directories in the script are changed to the correct ones with complete patient data. 
**Usage**:
```bash
bash runhdglio.sh
```
It should output: 
- Registered MRI sequences
- Tumour segmentation masks
- Volume measurements
- Registration matrices

The bash files "create_volumes.sh" and "extract_volumes.sh" were used to extract and put these volumes all into one file for easier access for numerical and neural models. 

## Neural Model 

1. Markov State Transition Matrices (statespace.py)
This script builds state transition probability matrices for disease progression modeling based on MGMT methylation status, a key biomarker that predicts treatment response in glioblastoma. It first loads and cleans clinical rating data (CR, PR, SD, PD, OP states) and separates patients into methylated vs. unmethylated cohorts. It then tracks all state-to-state transitions for each patients and calculates normalized transition probabilities. 

Input: LUMIERE_ExpertRating_Train.csv, LUMIERE_Demographics_Pathology_Train.csv

Output: methylated_train.csv, unmethylated_train.csv

After the Markov transition probability matrices are outputted, they are used in the hybrid neural model. 

2. Hybrid Neural Model

This section implements a hybrid neural ODE model that combines the previously produced Markov state transition matrices with deep learning to output a continuous tumour volume trajectory based on a patient's methylation status. 

Package requirements can be installed using:
```bash
pip install torch torchdiffeq numpy pandas matplotlib scikit-learn
```

Input files: 
- `methylated_train.csv` / `unmethylated_train.csv` - Transition matrices from `statespace.py`
- `all_tumor_volumes_hdglio_test.csv` 
- `LUMIERE_Demographics_Pathology_Train.csv`
-  `final_hybrid_ode_weights.pth`

The final_hybrid_ode_weights.pth file contains the trained data that is used in hybrid_neural.py to produce the tumour growth plots. It is recommended that one uses this file when running to save time rather than training the model themselves. While running, **ensure these files are in the same folder and directories are updated correctly**. 

Once all of this is complete, run the script hybrid_neural.py:
```bash
python hybrid_neural.py
```
This will load the pre-trained model from 'final_hybrid_ode_weights.pth' and generate prediction withs 200 Monte Carlo trajectories per patient. It then calculates metrices: MASE, Chi-squared, NSE, KGE, and creates comparison plots with uncertainty bounds. Specifically `neuralode_comparison_PatientID.png` which gives individual patieent prediction plots, which is used to compare with numerical method plots. 

## Numerical Overview
The project implements a **Stochastic Logistic Growth Model** to predict tumour volume trajectories. It accounts for biological randomness using Monte Carlo ensembles and compares three numerical integration schemes of varying complexity:
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

**Numerical Method Comparisons**: The project evaluates three solvers with different mathematical properties:
1. EM: Our baseline. It uses a simple first-order approximation but can be sensitive to large time steps. Data from this method is taken from em_predictive_data.py, and plots from em_predictive_plots.py within euler_maruyama. 
2. Milstein: Specifically addresses "multiplicative noise" ($\sigma V$). It includes an Itô correction term ($0.5 \sigma^2 V$), which provides higher-order strong convergence. Data from this method is taken from milstein_prediction_data.py, and plots from milstein_prediction_plots.py within milstein. 
3. SRK: A derivative-free multi-stage method. It uses a predictor-corrector approach to achieve high-order accuracy without needing to analytically compute the derivative of the diffusion term. Data from this method is taken from rkpredictive_data.py, and plots from rkpredictive_plots.py within stochastic_rk4. 

4. Training & Visualization Scripts: Used for processing the 80-patient training set and generating aggregate convergence and scaling plots across various discretization steps ($\Delta t$). Data was taken from the tumour_data.csv file (derived from the LUMIERE dataset) is already included within the branched folders of this repository. You do not need to provide external volumetric data to run the solvers.
5. Comparison: within the comparison_methods folder is the following structure:
```bash
.
├── tumour_data.csv          # Pre-processed volumetric data for all patients
├── numerical_stats/         # Aggregated CSV metrics (MASE, Chi2, etc.)
├── neuralstats/             # Output from neural ODE
├── test_plots/              # final paper-style comparison figures
├── comparison_conv.py # script to compare convergence of methods
└── comparison_metrics.py          # script to compare metrics of methods
```
After execution, the following directories will be populated with results:

Trajectory Fits: milstein_predictive_plots/, srk_predictive_plots/, and em_predictive_plots/ will contain PNG files for every patient. These show the 50/50 split, the ensemble mean, and the 95% confidence intervals derived from the Monte Carlo runs.

Performance Metrics: test_plots/ contains the final box plots. These visualizations specifically utilize IQR Clipping to remove statistical outliers (like Patient 43 and 77), allowing for a clear visual comparison of the "typical" performance across all four models.

Stability is evaluated via Trajectory Convergence Time (TCT). The scripts calculate the earliest week $t^*$ where the relative change in the ensemble mean stays below a tolerance of $10^{-4}$ for at least 5 consecutive weeks. This metric is used to contrast the rapid stabilization of Neural ODEs against the high-volatility tail behavior of the EM baseline.

## Data Source and Preprocessing
The included tumour_data.csv is derived from the LUMIERE dataset (Longitudinal Glioblastoma MRI with expert RANO evaluation). We performed automated segmentation using HD-GLIO and nnU-Net, extracting volumetric data across four MRI sequences (T1, T1c, T2, and FLAIR). This ensures the numerical models are grounded in high-fidelity, longitudinal clinical observations.

## Verification Script (**FOR TAS**)
To facilitate quick grading, the scripts test_em.py, test_milstein.py, and test_srk.py have been pre-configured as "High-Efficiency" versions within the folder test. These scripts isolate a single patient and use optimized hyperparameters to significantly reduce wall-clock runtime without compromising the underlying mathematical logic.
We have implemented two primary changes to the numerical configurations to ensure each script finishes in around 5-10 minutes total for all three scripts:
1. Discretization ($\Delta t$): Increased to $\Delta t = 0.5, 1$. By using a larger step size, the number of iterations per simulation is reduced.
3. Ensemble Size ($M$):Standard: $M = 60$ paths (For stable 95% Confidence Intervals).Grading Mode: $M = 20$ paths. This reduces the Monte Carlo computational load by 66%.
Each script targets Patient 19 by default. Upon execution, the scripts will automatically create and populate the following directories:

- em_predictive_outputs/

- milstein_predictive_outputs/

- srk_predictive_outputs/

Each directory will contain:
1. Predictive Plot: A .png visualization showing the Seen data (used for fitting), the Unseen data (held out for testing), and the resulting SDE trajectory.
2. Metrics CSV: A summary containing the MASE, $\chi^2$, NSE, and KGE scores, along with the calculated Trajectory Convergence Time (TCT).

If you wish to verify the model against other patients from our test cohort, simply modify the TESTING_PATIENTS list at the top of each script: (Change this to any of the 10 test patients: [31, 19, 43, 54, 77, 73, 72, 71, 67, 52])
```bash
TESTING_PATIENTS: List[int] = [...]
```

Please follow these steps to verify the numerical models and predictive accuracy:
1. Environment Check: Ensure numpy, matplotlib, scipy, pandas, and tqdm are installed in your Python environment.
2. Execution: Run the three test scripts in any order:

- python test_em.py

- python test_milstein.py

- python test_srk.py

3. Directory Verification: Confirm that three new folders have been created:

- em_predictive_outputs/

- milstein_predictive_outputs/

- srk_predictive_outputs/

4. Visual Inspection: Open the .png files in each folder. Verify that:

- Seen Data (Blue circles) represents the training set.

- Unseen Data (Red crosses) represents the test set.

- The Solid Line shows the SDE model's mean forecast.

5. Data Verification: Open the .csv files. Check for MASE, NGE, KSE, Chi-squared, TCT convergence, strong/weak convergence, runtime convergence. 

**Example Data Interpretation:** For example:
MASE (0.97): Indicates a very strong predictive fit on the unseen test data.TCT (1330.0): Shows that this specific simulation did not reach a stable plateau within the clinical window (often signifying active late-stage growth).StrongError (2523|2325|...|1915): Shows a steady decrease in error as the time step is refined from $0.8$ down to $0.05$, proving numerical stability.

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
