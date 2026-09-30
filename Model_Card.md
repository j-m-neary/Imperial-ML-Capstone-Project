MODEL CARD FOR THE BBO OPTIMISATION APPROACH

MODEL OVERVIEW

Approach Name: Anisotropic Gaussian Process Bayesian Optimiser with ARD Kernel Tuning (AGP-BO)

Model Type: Probabilistic Non-parametric Surrogate Regressor with Constrained Acquisition Search

Model Version: Version 2.0 (Iteration Round 10)

Core Framework: Python 3.12, Scikit-Learn (GaussianProcessRegressor), SciPy, NumPy, Pandas

INTENDED USE

Primary Intended Uses:
Optimisation of expensive-to-evaluate, continuous black-box objective functions where gradient information is unavailable and evaluation budgets are strictly limited (N < 100).

Relevant Practical Domains:
Hyperparameter optimisation in deep learning, tuning industrial chemical formulations, automated robotics trajectory control, and calibrating complex biological/epidemiological simulations.

Out-of-Scope / Prohibited Uses:

High-throughput optimisation problems where evaluations are inexpensive (where evolutionary strategies or Nelder-Mead simplex methods are superior).

Non-continuous, discrete, or mixed-integer problems without appropriate kernel encoding.

Highly non-stationary landscapes characterised by sharp discontinuities, step functions, or cliffs where stationary RBF kernels fail.

OPTIMISATION DETAILS AND STRATEGY EVOLUTION

Strategy Architecture:

Surrogate Model: Composite Constant Kernel multiplied by an Anisotropic Radial Basis Function (RBF) with dimension-specific length scales:
k(x, x') = sigma_f^2 * exp(-0.5 * sum((x_d - x'_d)^2 / l_d^2))

Length-Scale Bounds: Bounds set to [1e-3, 1e5] with 15 optimizer restarts to prevent premature convergence to poor local minima.

Target Handling: normalize_y=True and alpha = 1e-3 regularisation noise floor.

Acquisition Mechanics:

Expected Improvement (EI): Configured with exploration jitter xi = 0.005 to identify promising adjacent regions.

Upper Confidence Bound (UCB): Exploitative tuning using low kappa values (kappa = 0.5 to 0.8) to track proven ascent ridges.

Safeguards: Candidates restricted to interior bounds ([0.05, 0.95]) and filtered using Euclidean distance thresholds (delta >= 0.012 to 0.020) to eliminate duplicate evaluations.

Ten-Round Strategy Evolution:

Rounds 1-3: Broad variance-driven exploration establishing baseline bounds.

Rounds 4-7: ARD hyperparameter divergence; isolating dominant drivers (e.g. x1 in Function 1; x3 in Function 3 and 5) and letting invariant dimensions float.

Rounds 8-10: Strict exploitation along verified ridges, relying on low-kappa UCB and narrow Gaussian candidate pools.

PERFORMANCE AND RESULTS SUMMARY
Evaluations were measured by maximum observed scalar yield (y_max) and simple regret reduction across the ten rounds:

Function 1 (2D Plume): Baseline -0.004545 -> Peak +3.4017e-10 (ARD: x1 sensitive at l1 ~ 0.014; x2 invariant at l2 ~ 57,000)

Function 2 (2D Synthesis): Baseline -0.056647 -> Peak +0.629679 (ARD: x1 narrow corridor at l1 ~ 0.017; x2 smooth at l2 ~ 1.48)

Function 3 (3D Climate): Baseline -0.134425 -> Peak -0.009522 (ARD: x3 primary driver at l3 ~ 0.102; x1 invariant at l1 = 1,000)

Function 4 (4D Rover): Baseline -16.920548 -> Peak +0.609786 (ARD: isotropic manifold, l_d ~ 0.65 to 0.75 across all dimensions)

Function 5 (4D Vaccine): Baseline 8.377400 -> Peak +3048.662950 (ARD: x3 sensitive driver at l3 ~ 0.031; x2 invariant at l2 ~ 1,560)

Function 6 (5D Formulation): Baseline -1.699914 -> Peak -0.426447 (ARD: active curvature along x2 at l2 ~ 0.26 and x5 at l5 ~ 0.30)

Function 7 (6D Tuning): Baseline 0.412950 -> Peak +1.587444 (ARD: active drivers x1, x4, x5; invariant dimensions x2, x3, x6)

Function 8 (8D High-Dim): Baseline 9.707293 -> Peak +9.965739 (ARD: high plateau landscape; x8 invariant at l8 = 100,000)

ASSUMPTIONS AND LIMITATIONS

Smoothness and Stationarity: Assumes the underlying response surface is continuous and stationary. Non-stationary behaviour (such as phase transitions) causes the RBF kernel to oversmooth true peaks.

Single Basin Assumption: Late-stage exploitation assumes that the highest peak found in early rounds belongs to the basin of the global optimum. In highly multimodal landscapes, the algorithm can become trapped in suboptimal local peaks.

Dimensionality Bottlenecks: In 6D and 8D spaces, sample sizes under 50 leave the space sparsely mapped. Predictions in unsampled regions revert to the prior mean.

ETHICAL CONSIDERATIONS AND TRANSPATENCY

Reproducibility: All decisions are deterministic and auditable by setting fixed random seeds (random_state=42), explicit bounding boxes, and published code routines.

Safety in Real-World Deployment: When deploying surrogate optimisation to physical environments (e.g. livestock biosecurity, automated dosing, or agricultural machinery), unexpected model uncertainty or ungrounded extrapolation can cause operational failure. Transparent model cards ensure operators understand domain boundaries, structural assumptions, and where human-in-the-loop oversight is required.

Adequacy of Structure: The current structure balances mathematical rigour with operational clarity. Additional narrative detail would dilute the critical technical insights (ARD mechanics, length-scale contraction, and distance filtering) that govern performance.
