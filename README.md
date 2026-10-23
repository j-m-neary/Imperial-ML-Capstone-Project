# Imperial-ML-Capstone-Project: Black-Box Optimisation (BBO)

**Programme:** Imperial College London – Professional Certificate in Machine Learning & AI  
**Author:** Dr Joseph Neary | Senior Lecturer in Livestock Health & Welfare, University of Liverpool  
**Repository:** `Imperial-ML-Capstone-Project`  
**Status:** Completed (13 Iterative Rounds Concluded)  
**Application Focus:** Translating Bayesian Black-Box Optimisation to sample-efficient experimental design, livestock infectious disease surveillance, precision dairy automation, and pasture management.

---

## Governance & Transparency Documentation

In accordance with Module 21 (Transparency and Interpretability) standards, comprehensive governance artefacts accompany this repository:
* **Datasheet for the BBO Dataset (`DATASHEET.md`):** Documents dataset provenance, binary NumPy tensor schemas, 13-round active-learning sequential collection histories, normalisation workflows, and structural coverage boundaries.
* **Model Card for the BBO Optimiser (`MODEL_CARD.md`):** Details the Anisotropic Gaussian Process Regressor (AGP-BO) architecture, Automatic Relevance Determination (ARD) length-scale dynamics, time-decaying acquisition schedules, terminal exploitation policies, and diagnostic deployment guardrails.

---

## 1. Project Overview & Clinical/Epidemiological Context

This capstone project addresses the challenge of discovering the global maximums of eight continuous, expensive-to-evaluate "black-box" functions under incomplete knowledge across 2D to 8D search spaces. In real-world machine learning and applied sciences—such as deep learning hyperparameter tuning, bioprocess yield calibration, and epidemiological intervention design—closed-form analytical equations are rarely available. Instead, practitioners must sequentially query an expensive system, observe scalar feedback, and systematically steer proposals toward optimal configurations.

Coming from a veterinary medicine, epidemiology, and livestock production background, this optimisation challenge directly mirrors complex field trials. In clinical veterinary trials, precision livestock farming, and herd-level disease interventions (such as managing pathogen-microbiome interactions during cattle commingling or optimising rotational grazing), empirical field evaluations are severely constrained by animal welfare, biological latency, and substantial economic costs. Building a sample-efficient Bayesian Optimisation framework bridges core machine learning theory with pragmatic experimental design, establishing advanced competencies in Gaussian Process (GP) regression, Automatic Relevance Determination (ARD), and finite-horizon acquisition policies.

---

## 2. Inputs, Outputs, and Operational Constraints

The project sequentially optimises eight distinct functions ($f_1$ to $f_8$) spanning dimensions $D = 2$ up to $D = 8$, bounded within a normalised unit hypercube $[0.0, 1.0]^D$.

* **Model Inputs ($x$):** A continuous coordinate vector $x = [x_1, x_2, \dots, x_D]$, where each $x_i \in [0.0, 1.0]$.
  * *Submission Format Example (5D):* `0.530854 - 0.325344 - 0.655830 - 0.970957 - 0.067609`
* **Model Outputs ($y$):** A continuous scalar response score $y = f(x) + \epsilon$, where $\epsilon$ denotes domain-specific observational noise.
* **Operational Constraints:**
  1. *Zero Domain Knowledge:* Functional topographies, convexity, multimodal landscapes, and gradient vectors are completely unknown a priori.
  2. *Strict Query Latency:* Sequential weekly query batches impose latency delays, demanding deliberate, sample-efficient planning rather than brute-force high-frequency evaluations.
  3. *Dimensional Sparsity:* As dimensionalities increase to 8D, search space volumes grow exponentially ($V = L^D$), rendering uniform grids and random search ineffective.
  4. *Finite Evaluation Horizon:* A hard cap of 13 rounds requires non-stationary balancing of exploration and exploitation, with exploration value collapsing to zero in the terminal round.

---

## 3. Executive Performance Summary: Final Results (Round 13)

Across the 13-round campaign, the optimisation framework secured all-time global project records across multiple functions and pinpointed summits across the remaining landscapes:

| Function | Dim | Problem Domain Analogue | Best Observed Score ($y^*$) | Best Round | Optimal Parameter Vector ($x^*$) |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Function 1** | 2D | Atmospheric Plume Source Localisation | **+2.834851e-09** | **Round 13** | `[0.539500, 0.560000]` |
| **Function 2** | 2D | Molecular Compound Synthesis Yield | **+0.682263** | Round 10 | `[0.696107, 0.950000]` |
| **Function 3** | 3D | Climate Simulator Calibration | **-0.008728** | Round 11 | `[0.850000, 0.515102, 0.393265]` |
| **Function 4** | 4D | Autonomous Rover Trajectory Control | **+0.609786** | Round 9 | `[0.403323, 0.417356, 0.355442, 0.430245]` |
| **Function 5** | 4D | Vaccine Bioreactor Yield Maximisation | **+3235.544585** | **Round 13** | `[0.205663, 0.939808, 0.974853, 0.984621]` |
| **Function 6** | 5D | Multi-Component Formulation Optimisation | **-0.412805** | Round 10 | `[0.538559, 0.314172, 0.688670, 0.969842, 0.062739]` |
| **Function 7** | 6D | Hyperparameter Tuning Benchmark | **+1.587444** | Initial / R13 | `[0.090550, 0.448455, 0.190005, 0.183119, 0.381195, 0.725980]` |
| **Function 8** | 8D | High-Dimensional Systems Control | **+9.968870** | **Round 12** | `[0.153176, 0.158316, 0.158207, 0.162015, 0.980000, 0.556544, 0.234606, 0.495983]` |

*Key Highlights:*
* **Function 1:** Bracketed the hypersensitive emission cross-section along $x_1$ ($0.537 < x_1^* \approx 0.5395 < 0.540$), establishing an all-time record score of $+2.834851 \times 10^{-9}$ in the final round.
* **Function 5:** Discovered an unprecedented bioreactor yield of $3235.54$ in Round 13, surpassing the initial course benchmark ($3048.66$) by more than 186 units by locking into the upper boundary subspace ($x_3 \ge 0.97, x_4 \ge 0.98$).
* **Function 8:** Maintained consistent plateau harvesting in 8D ($y > 9.968$), achieving an all-time global record ($9.968870$) in Round 12.

---

## 4. Technical Strategy & Algorithmic Evolution

The technical framework evolved through four distinct architectural phases across the 13 rounds:

### Phase 1: Unconstrained Global Exploration (Rounds 1–4)
* Initialised with isotropic Radial Basis Function (RBF) kernels and unconstrained global Latin Hypercube Sampling (LHS).
* *Identified Failure Mode:* Standard isotropic models suffered from the curse of dimensionality in $D \ge 4$. Furthermore, unconstrained optimisers pushed candidate queries to the outer boundaries ($x_i \in \{0.0, 1.0\}$), causing boundary-pooling artefacts.

### Phase 2: Structural Regularisation & Active Manifold Discovery (Rounds 5–8)
* Migrated to **Anisotropic RBF Kernels with Automatic Relevance Determination (ARD)**.
* Implemented **Interior Domain Clipping (`[0.02, 0.98]`)** to force proposals into active functional space.
* Added a numerical regularisation floor (`alpha = 1e-3`) and target standardisation (`normalize_y=True`) to stabilise the Cholesky decomposition of the kernel Gram matrix.
* *ARD Breakthrough:* Inverse length scales revealed that nominal high-dimensional spaces collapsed onto lower-dimensional active manifolds. Inactive parameters (e.g., $x_1$ in Function 3; $x_6, x_8$ in Function 8) expanded toward large length scales ($\ell_i \ge 1000$), allowing the algorithm to ignore orthogonal noise and focus query density along active ascent ridges.

### Phase 3: Contracted Trust Regions & Dynamic Distance Filtering (Rounds 9–11)
* Transitioned from global search to **Contracted Local Perturbation Shells** centered on historical summit centroids:
  $$x_{\text{candidate}} \sim \mathcal{N}(x_{\text{best}}, \sigma_{\text{perturb}}^2 I)$$
* Integrated a **Vectorised Euclidean Exclusion Filter** ($\Delta x \ge 0.005$ to $0.012$) to eliminate redundant sampling near past queries, forcing the surrogate to explore local ridges rather than resampling known points.

### Phase 4: Finite-Horizon Terminal Exploitation (Rounds 12–13)
* Exploitation schedules were treated as a non-stationary decaying function of remaining time ($T$).
* With $T \le 2$, epistemic uncertainty ($\sigma(x)$) held zero residual value because undiscovered regions could no longer be capitalised upon in subsequent queries.
* Collapsed acquisition exploration slack to zero: Upper Confidence Bound ($\kappa \to 0.0$), Expected Improvement ($\xi \to 0.0$), and evaluated directly at the **Predicted Posterior Mean Maximum ($\arg\max \mu(x)$)** to harvest final summits.

---

## 5. Repository Structure

```text
Imperial-ML-Capstone-Project/
│
├── README.md                           <- Executive project overview and results ledger
├── DATASHEET.md                        <- Module 21 Dataset specification and data provenance
├── MODEL_CARD.md                       <- Module 21 Surrogate architecture and diagnostic card
│
├── Data/
│   └── Raw/
│       ├── function_1/                 <- 2D inputs/outputs (.npy) across all 13 rounds
│       ├── function_2/                 <- 2D inputs/outputs (.npy) across all 13 rounds
│       ├── function_3/                 <- 3D inputs/outputs (.npy) across all 13 rounds
│       ├── function_4/                 <- 4D inputs/outputs (.npy) across all 13 rounds
│       ├── function_5/                 <- 4D inputs/outputs (.npy) across all 13 rounds
│       ├── function_6/                 <- 5D inputs/outputs (.npy) across all 13 rounds
│       ├── function_7/                 <- 6D inputs/outputs (.npy) across all 13 rounds
│       └── function_8/                 <- 8D inputs/outputs (.npy) across all 13 rounds
│
├── Notebooks/
│   ├── 01_function_1_2D.ipynb          <- Plume localisation GP surrogate pipeline
│   ├── 02_function_2_2D.ipynb          <- Chemical synthesis GP surrogate pipeline
│   ├── 03_function_3_3D.ipynb          <- Climate calibration GP surrogate pipeline
│   ├── 04_function_4_4D.ipynb          <- Autonomous rover GP surrogate pipeline
│   ├── 05_function_5_4D.ipynb          <- Vaccine bioreactor GP surrogate pipeline
│   ├── 06_function_6_5D.ipynb          <- Multi-component formulation GP pipeline
│   ├── 07_function_7_6D.ipynb          <- ML hyperparameter tuning GP pipeline
│   └── 08_function_8_8D.ipynb          <- 8D Complex systems control GP pipeline
│
└── src/
    ├── kernels.py                      <- Anisotropic ARD kernel specifications
    ├── acquisition.py                  <- UCB, EI, and Argmax Mu acquisition policies
    └── geometry.py                     <- Euclidean exclusion filters & clipping utils
---

## 6. Real-World Applications in Veterinary Science & Agriculture

The sample-efficient methodologies validated in this repository translate directly to research workflows in veterinary medicine, livestock epidemiology, and precision agriculture:

1. **Precision Dairy & Autonomous Sensor Networks:** Sensor array thresholding (e.g., rumination collars, computer-vision body condition scoring, and lameness detection) presents complex, multi-parameter tuning challenges. Bayesian optimisation provides an automated mechanism to tune detection sensitivities and false-alarm latencies in silico, avoiding animal disruption and alert fatigue.
2. **Veterinary Epidemiology & Commingling Pathogen Dynamics:** Testing intervention strategies (stocking densities, ventilation flow rates, commingling timings, biosecurity barriers) during cattle commingling or bovine respiratory disease outbreaks mirrors high-stakes black-box optimisation. ARD dimensional reduction enables researchers to isolate the 2 or 3 critical management levers driving infection resilience while filtering out background environmental noise.
3. **Sustainable Pasture & Grazing Optimization:** Managing rotational grazing and out-wintered suckler systems requires balancing forage regrowth, soil compaction, supplementation costs, and animal welfare across weather conditions. Sequential surrogate modelling provides a quantitative framework to identify optimal stocking cycles and rotational intervals within a minimal number of seasonal trials.
4. **AutoML for Precision Phenotyping:** Deep learning architectures trained on agricultural time-series data require fine hyperparameter tuning. Applying finite-horizon Bayesian optimisation ensures robust model convergence within constrained computational budgets.

---

## 7. Dependencies & Execution

The project is built on standard scientific Python libraries without external AutoML dependencies:

```bash
# Clone the repository
git clone [https://github.com/DrJNeary/Imperial-ML-Capstone-Project.git](https://github.com/DrJNeary/Imperial-ML-Capstone-Project.git)
cd Imperial-ML-Capstone-Project

# Recommended environment setup
conda create -n bbo-capstone python=3.11 -y
conda activate bbo-capstone

# Install core scientific dependencies
pip install numpy scipy scikit-learn pandas matplotlib jupyter
