# Imperial-ML-Capstone-Project
Capstone project for Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence course[cite: 3].

Applying Machine Learning techniques to the project topic, with a focus on how these models could be adapted for infectious disease monitoring in livestock populations[cite: 3].

---
**Author:** Dr Joseph Neary | [University of Liverpool Profile](https://www.liverpool.ac.uk/people/joseph-neary)[cite: 3]  
**Programme:** Imperial College London – Professional Certificate in Machine Learning & AI[cite: 3]  
**Focus:** Black-Box Optimisation (BBO) Capstone[cite: 3]  
**Repository:** [Imperial-ML-Capstone-Project](https://github.com/j-m-neary/Imperial-ML-Capstone-Project)[cite: 3]

---

## 📋 Module 21 Governance & Transparency Documentation
In accordance with Module 21 (Transparency and Interpretability) standards, full documentation artefacts accompany this project[cite: 22]:
* 📄 **[Datasheet for the BBO Dataset](DATASHEET.md):** Details data motivation, binary NumPy tensor composition, collection history across ten rounds, data standardisation, and structural coverage limitations[cite: 22].
* 📋 **[Model Card for the BBO Optimiser](MODEL_CARD.md):** Outlines the Anisotropic Gaussian Process (AGP-BO) architecture, ARD length-scale convergence dynamics, acquisition selection policies, quantitative benchmark performance, and deployment guardrails[cite: 22, 25].

---

## Section 1: Project Overview

This project focuses on optimizing eight continuous, expensive-to-evaluate "black-box" functions under conditions of incomplete knowledge[cite: 3]. In real-world machine learning and applied sciences—such as hyperparameter tuning for deep learning, epidemiology model calibration, or bioprocess yield management—we rarely possess analytical formulas for our objective functions[cite: 3, 25]. Instead, we must query an expensive system, observe feedback, and iteratively guide the search towards optimal parameter configurations[cite: 3].

Coming from a veterinary infectious disease background, I approach this challenge by recognizing how Black-Box Optimisation mirrors complex biological sampling, where field evaluations (e.g., disease risk assessment or biosecurity intervention trials) are highly resource-intensive[cite: 3]. Developing proficiency in BBO bridges theoretical machine learning and pragmatic experimental design, building core competencies in Gaussian Process (GP) regression, acquisition function design, and surrogate modelling[cite: 3, 25].

---

## Section 2: Inputs and Outputs

The system optimizes eight distinct functions ($f_1$ to $f_8$) spanning varying dimensionalities ($2\text{D}$ up to $8\text{D}$) bounded within a normalized unit hypercube $[0, 1]^d$[cite: 3].

* **Model Inputs:** A continuous coordinate vector $\mathbf{x} = [x_1, x_2, \dots, x_d]$, where $x_i \in [0.0, 1.0]$[cite: 3].
  * *Submission Format Example ($5\text{D}$):* `0.497666 - 0.243930 - 0.614364 - 0.934116 - 0.165796`[cite: 3]
* **Model Outputs:** A scalar evaluation response score $y = f(\mathbf{x}) + \epsilon$, where $\epsilon$ represents observational noise depending on the specific domain[cite: 3].
  * *Response Example:* `y = 2099.962628`[cite: 3]

---

## Section 3: Challenge Objectives & Operational Constraints

The primary objective is to **maximise** the output score $y$ across all eight functions within a strict evaluation budget of weekly query rounds[cite: 3]. 

### Key Constraints & Operational Limitations
1. **Zero Domain Knowledge:** Functional forms, convexity, and global landscape topographies are completely unknown a priori[cite: 3].
2. **Sequential Query Latency:** Batched feedback creates a response delay, requiring principled sampling strategies rather than high-frequency brute-force evaluation[cite: 3].
3. **Dimensional Sparsity:** As input dimensions scale from $2\text{D}$ to $8\text{D}$, the search space volume grows exponentially ($V = L^d$), rendering traditional grid search computationally infeasible[cite: 3].

---

## Section 4: Technical Approach & Strategy Evolution

My technical framework evolved sequentially from exploratory baseline sampling to fully automated probabilistic surrogate modelling across ten iterative rounds[cite: 3, 25]:
