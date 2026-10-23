# MODEL CARD FOR THE BBO OPTIMISATION APPROACH

## MODEL OVERVIEW

* **Approach Name:** Anisotropic Gaussian Process Bayesian Optimiser with ARD Kernel Tuning (AGP-BO)
* **Model Type:** Probabilistic Non-parametric Surrogate Regressor with Constrained Acquisition Search
* **Model Version:** Version 3.0 (Iteration Round 13 – Terminal Campaign Conclusion)
* **Core Framework:** Python 3.11 / 3.12, Scikit-Learn (`GaussianProcessRegressor`), SciPy, NumPy, Pandas
* **Author:** Dr Joseph Neary (University of Liverpool / Imperial College London Capstone)

---

## INTENDED USE

### Primary Intended Uses
Optimisation of expensive-to-evaluate, continuous black-box objective functions where gradient information is unavailable and evaluation budgets are strictly limited ($N \le 50$ per function).

### Relevant Practical Domains
Hyperparameter optimisation in deep learning, tuning industrial chemical formulations, automated robotics trajectory control, and calibrating complex biological/epidemiological simulations (e.g. livestock disease surveillance, bioprocess yield calibration, and precision pasture management).

### Out-of-Scope / Prohibited Uses
* High-throughput optimisation problems where evaluations are inexpensive (where evolutionary strategies, CMA-ES, or Nelder-Mead simplex methods are computationally superior).
* Non-continuous, combinatorial, discrete, or mixed-integer problems without appropriate categorical kernel encodings.
* Discontinuous landscapes characterised by sharp step functions or phase transitions where stationary RBF kernels fail to model boundary cliffs.

---

## OPTIMISATION DETAILS AND STRATEGY EVOLUTION

### Strategy Architecture
* **Surrogate Model:** Composite Constant Kernel multiplied by an Anisotropic Radial Basis Function (RBF) with Automatic Relevance Determination (ARD) dimension-specific length scales:
  $$k(x, x') = \sigma_f^2 \exp\left(-\frac{1}{2} \sum_{d=1}^D \frac{(x_d - x'_d)^2}{\ell_d^2}\right)$$
* **Hyperparameter Optimisation:** Fitted via Maximum Marginal Likelihood with 25 random restarts (`n_restarts_optimizer=25`) to prevent convergence to suboptimal local parameter minima.
* **Length-Scale Bounds:** Configured flexibly across `[1e-3, 1e5]` to accommodate both hypersensitive active dimensions ($\ell_d < 0.05$) and invariant background dimensions ($\ell_d \ge 10^3$).
* **Target Handling & Regularisation:** Target standardisation enabled (`normalize_y=True`) to maintain zero mean and unit variance across heterogeneous output distributions; diagonal regularisation noise floor (`alpha = 1e-3`) enforced to prevent Cholesky decomposition failures.

### Acquisition Mechanics & Terminal Policies
* **Expected Improvement (EI):** Used in early-to-mid rounds with exploration slack ($\xi = 0.05 \to 0.005$) to balance local exploitation with epistemic uncertainty resolution. In terminal rounds, exploration slack collapsed to zero ($\xi = 0.0$).
* **Upper Confidence Bound (UCB):** Employed with a time-decaying exploration schedule ($\kappa = 1.96 \to 0.5 \to 0.0$), transitioning from global optimism under uncertainty to pure posterior mean exploitation.
* **Predicted Posterior Mean Maximum ($\arg\max \mu$):** Primary terminal policy ($T \le 2$), evaluating strictly at the peak of the surrogate surface where the value of residual exploration drops to zero.
* **Geometric Guardrails:**
  * *Interior Domain Clipping:* Proposals restricted to $[0.02, 0.98]$ to eliminate boundary-pooling artefacts.
  * *Euclidean Separation Filters:* Vectorised distance exclusion ($\Delta x \ge 0.004 \to 0.012$) enforced to prevent redundant re-sampling of known coordinates and matrix singularity.
  * *Contracted Trust-Region Shells:* In late rounds, global candidate pools were replaced with local Gaussian perturbation shells ($x \sim \mathcal{N}(x^*, \sigma^2 I)$, $\sigma = 0.005 \to 0.006$) centred on proven empirical maxima.

### Thirteen-Round Strategy Evolution
* **Rounds 1–4 (Global Mapping):** Baseline exploratory initialisations; isotropic kernels and uniform Latin Hypercube Sampling to assess global scales.
* **Rounds 5–8 (ARD Divergence & Active Manifolds):** Transition to anisotropic kernels; divergence of length scales separated active topological drivers from uninformative background noise.
* **Rounds 9–11 (Trust-Region Bracketing):** Introduction of contracted perturbation shells and Euclidean distance filtering around confirmed peaks.
* **Rounds 12–13 (Terminal Harvest):** Complete collapse of exploration bonuses ($\kappa = 0.0, \xi = 0.0$); consensus harvesting via posterior mean maximum ($\arg\max \mu$) yielding multiple all-time records.

---

## PERFORMANCE AND QUANTITATIVE SUMMARY

Performance across all eight benchmark problems across the complete 13-round campaign:

| Function | Dim | Problem Domain Analogue | Baseline Score ($y_0$) | Peak Score ($y^*$) | Peak Round | Terminal ARD Length-Scale Dynamics ($\ell$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Function 1** | 2D | Atmospheric Plume Source Localisation | $-4.545 \times 10^{-3}$ | **$+2.834851 \times 10^{-9}$** | **Round 13** | $\ell_1 \approx 0.014$ (hypersensitive); $\ell_2 \approx 57,000$ (invariant) |
| **Function 2** | 2D | Molecular Compound Synthesis Yield | $-0.056647$ | **$+0.682263$** | Round 10 | $\ell_1 \approx 0.0107$ (active narrow ridge); $\ell_2 \approx 0.100$ (lower bound) |
| **Function 3** | 3D | Climate Simulator Calibration | $-0.134425$ | **$-0.008728$** | Round 11 | $\ell_1 \approx 12.0$ (invariant background); $\ell_2 \approx 0.845$; $\ell_3 \approx 0.0845$ (primary driver) |
| **Function 4** | 4D | Autonomous Rover Trajectory Control | $-16.920548$ | **$+0.609786$** | Round 9 | Isotropic manifold: $\ell \approx [0.75, 0.68, 0.73, 0.67]$ |
| **Function 5** | 4D | Vaccine Bioreactor Yield Maximisation | $+8.377400$ | **$+3235.544585$** | **Round 13** | $\ell_1 \approx 5.0$; $\ell_2 \approx 1.0$; active drivers $\ell_3 \approx 0.416, \ell_4 \approx 0.700$ |
| **Function 6** | 5D | Multi-Component Formulation Blending | $-1.699914$ | **$-0.412805$** | Round 10 | $\ell_1 \approx 3.51 \times 10^4$ (invariant); primary drivers $\ell_5 \approx 0.058, \ell_3 \approx 0.314$ |
| **Function 7** | 6D | Hyperparameter Tuning Benchmark | $+0.412950$ | **$+1.587444$** | Initial / R13 | Invariant $\ell_1 \approx 2880, \ell_2 \approx 14700$; primary driver $\ell_5 \approx 0.084$ |
| **Function 8** | 8D | High-Dimensional Systems Control | $+9.707293$ | **$+9.968870$** | **Round 12** | Plateau manifold; invariant $\ell_8 = 10^5$; smooth drivers $\ell_4 \approx 8.14, \ell_5 \approx 16.3$ |

---

## ASSUMPTIONS AND LIMITATIONS

* **Smoothness and Stationarity:** The model assumes the true underlying response surface is continuous and stationary. While the anisotropic ARD formulation resolves differing axis sensitivities, abrupt step changes or cliffs cause stationary RBF kernels to oversmooth local gradients.
* **Basin Commitment in Finite Horizons:** Late-stage trust-region exploitation operates under the empirical assumption that the highest summit identified during exploratory phases resides within the global optimum basin. In deceptive, highly multimodal landscapes, this risks premature convergence to a suboptimal local peak.
* **Dimensional Sparsity:** In 6D and 8D spaces ($N = 22$ to $47$), the volume of the search space remains overwhelmingly unsampled. The model relies heavily on ARD to collapse nominal dimensionality onto low-dimensional active sub-manifolds. Regions distant from historical observations revert to the prior mean.

---

## ETHICAL CONSIDERATIONS AND TRANSPARENCY

* **Reproducibility and Determinism:** All sampling routines, surrogate fits, and candidate proposals are strictly auditable through fixed pseudo-random seeds (`random_state=42`), explicit bounding constraints, and documented acquisition outputs.
* **Safety in Real-World Deployment:** When transferring Bayesian optimisation algorithms to physical biological systems (e.g. livestock biosecurity interventions, automated therapeutic dosing, or pasture grazing allocation), ungrounded model extrapolation or high uncertainty can cause operational harm or animal welfare risks. Transparent documentation ensures operators understand domain boundaries, active constraints, and the necessity of maintaining human-in-the-loop oversight.
* **Structural Adequacy:** The documentation provides full mathematical and procedural transparency without excessive narrative padding, preserving the technical diagnostic insights (ARD mechanics, boundary clipping, and Euclidean separation) that govern performance.
