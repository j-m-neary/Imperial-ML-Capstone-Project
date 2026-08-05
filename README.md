# Imperial-ML-Capstone-Project
Capstone project for Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence course[cite: 1]

Applying Machine Learning techniques to the project topic, with a focus on how these models could be adapted for infectious disease monitoring in livestock populations[cite: 1].

---
**Author:** Dr. Joseph Neary | [University of Liverpool Profile](https://www.liverpool.ac.uk/people/joseph-neary)[cite: 1]  
**Programme:** Imperial College London – Professional Certificate in Machine Learning & AI  
**Focus:** Black-Box Optimisation (BBO) Capstone  

---

## Section 1: Project Overview

This project focuses on optimizing eight continuous, expensive-to-evaluate "black-box" functions under conditions of incomplete knowledge. In real-world machine learning and applied sciences—such as hyperparameter tuning for deep learning, epidemiology model calibration, or bioprocess yield management—we rarely possess analytical formulas for our objective functions. Instead, we must query an expensive system, observe feedback, and iteratively guide the search towards optimal parameter configurations.

Coming from a veterinary infectious disease background, I approach this challenge by recognizing how Black-Box Optimisation mirrors complex biological sampling, where field evaluations (e.g., disease risk assessment or biosecurity intervention trials) are highly resource-intensive. Developing proficiency in BBO bridges theoretical machine learning and pragmatic experimental design, building core competencies in Gaussian Process (GP) regression, acquisition function design, and surrogate modelling.

## Section 2: Inputs and Outputs

The system optimizes eight distinct functions ($f_1$ to $f_8$) spanning varying dimensionalities ($2\text{D}$ up to $8\text{D}$) bounded within a normalized unit hypercube $[0, 1]^d$.

* **Model Inputs:** A continuous coordinate vector $\mathbf{x} = [x_1, x_2, \dots, x_d]$, where $x_i \in [0.0, 1.0]$.
  * *Submission Format Example ($5\text{D}$):* `0.497666 - 0.243930 - 0.614364 - 0.934116 - 0.165796`
* **Model Outputs:** A scalar evaluation response score $y = f(\mathbf{x}) + \epsilon$, where $\epsilon$ represents optional observational noise depending on the specific domain.
  * *Response Example:* `y = 2099.962628`

## Section 3: Challenge Objectives

The primary objective is to **maximise** the output score $y$ across all eight functions within a strict evaluation budget of weekly query rounds. 

### Key Constraints & Operational Limitations
1. **Zero Domain Knowledge:** Functional forms, convexity, and global landscape topographies are completely unknown a priori.
2. **Sequential Query Latency:** Batched feedback creates a response delay, requiring principled sampling strategies rather than high-frequency brute-force evaluation.
3. **Dimensional Sparsity:** As input dimensions scale from $2\text{D}$ to $8\text{D}$, the search space volume grows exponentially ($V = L^d$), rendering traditional grid search computationally infeasible.

## Section 4: Technical Approach

My technical framework evolves sequentially from exploratory baseline sampling to fully automated probabilistic surrogate modelling.

### 1. Model Architecture & Hyperparameter Optimisation
I deploy **Gaussian Process Regressors (GPR)** as primary surrogate models. Depending on surface roughness and noise profiles, I select either a Matérn kernel ($\nu = 2.5$) for noisy telemetry or Radial Basis Function (RBF) kernels with Automatic Relevance Determination (ARD) for smooth, high-dimensional spaces. 

Kernel hyperparameters (length scales and signal variances) are re-fitted each round via Maximum Marginal Likelihood. In higher dimensions ($d \ge 5$), length-scale upper bounds are expanded (`length_scale_bounds=(1e-3, 1e5)`) to accommodate uninformative or flat feature dimensions without optimizer truncation.

### 2. Exploration vs. Exploitation Balance
To maximize utility across $50,000$ Monte Carlo candidate vectors per function, I evaluate the Upper Confidence Bound (UCB) acquisition function:

$$\text{UCB}(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \cdot \sigma(\mathbf{x})$$

I set $\kappa = 2.0$ to maintain a disciplined equilibrium between exploiting predicted peaks ($\mu$) and sampling unobserved, high-variance regions ($\sigma$).

### 3. Human-in-the-Loop Interventions & Alternative Paradigms
My approach combines automated acquisition with targeted human supervision:
* **Low Dimensions ($2\text{D}$–$3\text{D}$):** I visually audit posterior mean and epistemic uncertainty landscapes, enforcing local bounding-box constraints to probe isolated high-yield anomalies that global acquisition functions temporarily ignore.
* **High Dimensions ($4\text{D}$–$8\text{D}$):** Standard 2D slice projections introduce visual distortion due to variable freezing. I pivot from visual inspection to diagnostic length-scale analysis, using ARD to identify critical driver features versus dormant noise dimensions.
* **Classification Alternative:** I also evaluate framing BBO via soft-margin Support Vector Classifiers (SVC). By thresholding top-performing $y$-values, kernel SVMs can map non-linear decision boundaries to act as pre-filters for candidate samplers.

---
**Technologies & Tools**[cite: 1]
* **Python** (Pandas, Numpy, Scikit-Learn)[cite: 1]
* **Jupyter Notebooks**[cite: 1]
* **Git/GitHub** for version control[cite: 1]
