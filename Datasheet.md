# DATASHEET FOR THE BBO CAPSTONE PROJECT DATASET

## MOTIVATION

### Why was the dataset created?
This dataset was generated to support the Capstone Project within the Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence. It provides an empirical benchmark for active learning, Gaussian Process (GP) regression, and sequential black-box optimisation (BBO) algorithms across unknown, expensive-to-evaluate scalar response surfaces.

### What task does it support?
The dataset supports the task of iterative global optimisation under a strict finite-horizon evaluation budget ($T = 13$ rounds). It records sequential parameter choices and feedback across thirteen distinct querying rounds, demonstrating how surrogate models identify active subspaces, manage the curse of dimensionality, and balance exploration against terminal exploitation.

### Who created the dataset and who funded it?
The underlying objective functions and evaluation environment were provided by the Imperial College London programme team. The iterative queries, candidate selections, and empirical logs were produced, documented, and analysed by Dr Joseph Neary.

---

## COMPOSITION

### What do the instances comprise?
The dataset consists of paired continuous coordinate vectors and scalar outputs across eight distinct synthetic or semi-synthetic functions, encompassing a total of 13 sequential query rounds following initial baseline observations:

* **Function 1: 2D Spatial Plume Source Localisation** (22 observations, domain $[0, 1]^2$)
* **Function 2: 2D Chemical Compound Synthesis Yield** (22 observations, domain $[0, 1]^2$)
* **Function 3: 3D Climate Simulator Calibration** (22 observations, domain $[0, 1]^3$)
* **Function 4: 4D Autonomous Rover Trajectory Control** (22 observations, domain $[0, 1]^4$)
* **Function 5: 4D Bioreactor Vaccine Yield Maximisation** (22 observations, domain $[0, 1]^4$)
* **Function 6: 5D Multi-Component Formulation Blending** (22 observations, domain $[0, 1]^5$)
* **Function 7: 6D Hyperparameter Tuning Benchmark** (22 observations, domain $[0, 1]^6$)
* **Function 8: 8D High-Dimensional Systems Control** (47 observations, domain $[0, 1]^8$)

### What is the format of the data?
Data are stored as binary NumPy arrays (`.npy` files) in separate subdirectories for each function (`Data/Raw/function_i/`):

* `initial_inputs.npy`: 2D `float64` array of shape $(N, D)$ representing normalised parameter coordinates.
* `initial_outputs.npy`: 1D `float64` array of shape $(N,)$ containing evaluated scalar returns.

### Are there gaps, missing data, or structural biases?
There are no missing values or corrupted fields. However, substantial structural biases and coverage gaps are intentionally present:

* **Curse of Dimensionality:** In higher dimensions ($D \ge 5$), sample sizes of $N = 22$ (and $N = 47$ for Function 8) leave the vast majority of the unit hypercube volume completely unsampled.
* **Interior Domain Clipping:** Extreme boundary zones (coordinates $< 0.02$ or $> 0.98$) are deliberately under-represented to prevent numerical edge-pooling artefacts.
* **Exploitation and Crest Bias:** Due to the active learning setup, late-stage queries are densely clustered around discovered local summits and high-yield ridges (e.g., the $x_3 \ge 0.96, x_4 \ge 0.97$ subspace in Function 5, and the $y > 9.968$ plateau in Function 8). The dataset is not an independent and identically distributed (i.i.d.) uniform sample of the spaces.

---

## COLLECTION PROCESS

### How were the queries generated?
Queries were collected sequentially over thirteen iterative feedback cycles using a decaying exploration-exploitation acquisition policy:

* **Rounds 1–4 (Initial Global Exploration):** Baseline initialisations supplemented by Latin Hypercube Sampling (LHS) and global Upper Confidence Bound (UCB, $\kappa = 1.96$) / Expected Improvement (EI, $\xi = 0.05$) to explore global epistemic uncertainty.
* **Rounds 5–8 (Active Subspace Mapping & ARD Calibration):** Gaussian Process Regressors with Automatic Relevance Determination (ARD) were fitted to identify sensitive versus invariant parameters. Vectorised Euclidean distance exclusion filters ($\Delta x \ge 0.005$ to $0.012$) were enforced to eliminate redundant near-duplicate sampling.
* **Rounds 9–11 (Trust-Region Perturbation Shells):** Transition to contracted local Gaussian perturbation shells centred on empirical maxima ($x \sim \mathcal{N}(x^*, \sigma^2 I)$) with contracting radii ($\sigma = 0.05 \to 0.006$).
* **Rounds 12–13 (Finite-Horizon Terminal Harvest):** Complete collapse of exploration bonuses ($\kappa = 0.0, \xi = 0.0$). Proposals converged strictly onto the posterior mean maximum ($\arg\max \mu(x)$) to harvest the global summits.

### Over what time frame was the data collected?
Data points were accumulated sequentially over a 13-round evaluation period, concluding with the terminal Round 13 evaluation cycle.

### What sampling mechanisms were used?
Latin Hypercube Sampling, uniform Monte Carlo proposal pools, and local Gaussian perturbation shells filtered through an Anisotropic Gaussian Process surrogate model and acquisition policies.

---

## PREPROCESSING AND USES

### Have any transformations been applied?
* **Input Normalisation:** Coordinate inputs are bounded within the normalised unit hypercube $[0.0, 1.0]^D$.
* **Target Standardisation:** Scalar outputs are internally standardised (`normalize_y=True`) during Gaussian Process fitting to maintain zero mean and unit variance across heterogeneous output scales.
* **Regularisation Noise Floor:** A numerical regularisation nugget (`alpha = 1e-3`) was added to the covariance matrix diagonal to maintain numerical stability during Cholesky decomposition.

### What are the intended uses?
* Benchmarking sample-efficient Bayesian optimisation algorithms under tight query budgets ($T \le 13$).
* Analysing Automatic Relevance Determination (ARD) length-scale convergence and dimensional reduction in expensive black-box landscapes.
* Studying the transition from global exploration to local trust-region exploitation in finite-horizon active learning.

### What are inappropriate uses?
* **Global Surface Reconstruction:** The dataset should not be used to train unweighted global regression surrogates intended to interpolate the entire hypercube, as non-optimal regions are sparse and under-sampled.
* **Combinatorial or Discrete Optimisation:** The dataset reflects continuous Euclidean parameters and is unsuited for categorical or graph-based optimisation tasks.

---

## DISTRIBUTION AND MAINTENANCE

### Where is the dataset hosted?
The dataset is hosted in a public GitHub repository:  
`https://github.com/j-m-neary/Imperial-ML-Capstone-Project`  
(Located within the `Data/Raw/` subdirectories).

### What are the terms of use and licensing?
Distributed under the MIT Academic Open Source Licence for educational and scientific research evaluation.

### Who maintains the dataset?
Maintained by Dr Joseph Neary, with full version tracking documented via Git commits.
