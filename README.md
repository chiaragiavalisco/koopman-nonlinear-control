# Data-Driven Modeling, Prediction, and Control of Nonlinear Dynamical Systems via Koopman Operator Theory

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/framework-datafold%20%7C%20scikit--learn-orange.svg)](https://datafold-dev.gitlab.io/datafold/)

This repository contains the numerical implementations, benchmarks, and control frameworks developed as part of my **Master's Thesis**. The project investigates operator-theoretic techniques—specifically **Dynamic Mode Decomposition (DMD)**, **Extended Dynamic Mode Decomposition (EDMD)**, and **EDMD with Control (EDMDc)**—for the global data-driven linearization, forecasting, and feedback stabilization of nonlinear dynamical systems.

---

## 🔬 Theoretical Background

Traditional linearization techniques (such as Jacobian linearization) are strictly local around equilibrium points or nominal trajectories. In contrast, **Koopman Operator Theory** (Koopman, 1931; Koopman & von Neumann, 1932; Mezić, 2005) provides a global, linear, yet infinite-dimensional representation of nonlinear dynamical systems by shifting the perspective from state evolution to the evolution of **observable functions** $f \in \mathscr{F}$.

For an autonomous discrete-time system $\mathbf{x}_ {k+1} = \mathbf{S} (\mathbf{x}_ k)$, the Koopman operator $\mathcal{U}$ acts via composition:
$$\mathcal{U} f = f \circ \mathbf{S} \quad \implies \quad f(\mathbf{x}_{k+1}) = \mathcal{U} f(\mathbf{x}_k)$$

While $\mathbf{S}$ acts via function composition ($\mathbf{S}^k = \mathbf{S} \circ \dots \circ \mathbf{S}$), $\mathcal{U}$ evolves linearly via exponentiation ($\mathcal{U}^k f = \mathcal{U} \cdot \dots \cdot \mathcal{U} f$). This enables the deployment of linear estimation, spectral analysis, and optimal control techniques (e.g., LQR) on inherently nonlinear systems without local approximations.


```
   States:  x_k  ──────── Flow Map S (Nonlinear) ───────>  x_{k+1}
             │                                                │
Eigenfunctions │ Koopman Modes                   Eigenfunctions │ Koopman Modes
▼                                                ▼
Observables:  f(x_k) ───── Koopman Operator U (Linear) ────> f(x_{k+1})

```

---

## 📁 Repository Structure

```plaintext
├── hopf_bifurcation.py       # Example 1: Andronov-Hopf normal form (DMD vs. Polynomial EDMD vs. RBF EDMD)
├── vdp_control_lqr.py        # Example 2: Van der Pol oscillator feedback stabilization via EDMDc + LQR
├── vdp_koopman_benchmark.py  # Benchmark: Comprehensive VdP study (DMD, Polynomial, RBF, DMaps, EDMDc)
├── LICENSE                   # MIT License
└── README.md                 # Thesis and project documentation

```

---

## 🛠️ Case Studies & Scripts Overview

### 1. Normal Form of the Andronov–Hopf Bifurcation (`hopf_bifurcation.py`)

* **Dynamics:** Limit cycle attractor ($\mu = 1$) sampled outside the limit cycle over $t \in [0, 0.4]$ ($\Delta t = 0.02$).
* **Methods Evaluated:**
* **Standard DMD:** Fails to sustain oscillations; falsely characterizes the system as a continuous spiral sink ($\operatorname{Tr}(\mathbf{A}) \approx -2.81$, $\operatorname{det}(\mathbf{A}) \approx 2.92$).
* **EDMD (Degree-3 Polynomial):** Improves short-term reconstruction but diverges rapidly out-of-sample (OOS).
* **EDMD (Gaussian RBF Kernel, $\varepsilon = 0.17$):** Yields accurate phase-portrait reconstruction and stable long-term predictions by capturing localized nonlinearities.



### 2. Controlled Van der Pol Oscillator via EDMDc + LQR (`vdp_control_lqr.py`)

* **Dynamics:** Forced oscillator $\ddot{x} - \mu(1 - x^2)\dot{x} + x = u$ with actuation in $x_2$.
* **Methodology:**
* System identification via **EDMDc** using a targeted dictionary $\mathcal{D} = \{x_1, x_2, x_1^2, x_1^2 x_2\}$.
* Closed-loop stabilization using a **Linear Quadratic Regulator (LQR)** penalizing physical states $\mathbf{Q} = \operatorname{diag}(1, 1, 0, 0)$ with input penalty $R = 10^{-2}$.


* **Result:** Drives arbitrary initial conditions from the uncontrolled limit cycle directly to the unstable origin $\mathbf{x}^* = (0, 0)$ over $t \in [0, 5]$.

### 3. Comprehensive Koopman Benchmark on Van der Pol Dynamics (`vdp_koopman_benchmark.py`)

An exhaustive comparative study evaluating observable selection, spectral convergence, and control performance:

* **Standard DMD:** Decays towards the origin; structurally incapable of capturing nonlinear amplitude saturation.
* **Polynomial EDMD (Degrees 3 & 5 via `EDMDCV`):** Captures short-term geometric deformation, but OOS trajectories drift into unstable, outward-expanding spirals.
* **Gaussian RBF EDMD:** Achieves the lowest OOS prediction error among uncontrolled models, preserving the qualitative geometry of the limit cycle up to $t = 20$.
* **Diffusion Maps (DMs) as Data-Driven Dictionaries:**
* Spectral analysis of the graph Laplacian operator reveals that the first two non-trivial diffusion coordinates ($\zeta_1, \zeta_2$) carry almost all dynamical information (residual decay test).
* Embedding data into an intrinsic 2D manifold allows EDMD to reconstruct the nonlinear system using **only two diffusion coordinates**, avoiding manual dictionary engineering.


* **Controlled Comparison (EDMDc):** Demonstrates that while RBF dictionaries excel in autonomous forecasting, the **physics-informed VdP dictionary** is vastly superior for feedback control, achieving rapid convergence where RBF-based controllers exhibit slow or incomplete tracking.

---

## 📊 Summary of Results

| Setting | Method / Dictionary | Lifted Dim ($p$) | In-Sample Accuracy | Long-Term OOS Prediction | Control to Origin $(0, 0)$ |
| --- | --- | --- | --- | --- | --- |
| **Hopf** | Standard DMD | 2 | Poor (False Sink) | Diverges / Decays | N/A |
| **Hopf** | EDMD (Poly deg 3) | 9 | Moderate | Diverges ($t > 1.5$) | N/A |
| **Hopf** | **EDMD (Gaussian RBF)** | 66 | **High** | **Stable & Accurate** | N/A |
| **VdP** | Standard DMD | 2 | Poor | Decays to origin | N/A |
| **VdP** | EDMD (Poly deg 3 / 5) | 9 / 20 | Moderate | Unstable (Expanding spiral) | N/A |
| **VdP** | **EDMD (Gaussian RBF)** | 22 | **Very High** | **Preserves Limit Cycle** | Slow / Incomplete |
| **VdP** | **EDMD (Diffusion Maps)** | **2** | **Excellent** | **High (Intrinsic 2D Manifold)** | N/A |
| **VdP** | **EDMDc + LQR (VdP dict)** | 4 | High | N/A | **Fast & Exact ($t \le 5$)** |

---

## 🚀 Installation & Reproducibility

### Prerequisites

* Python $\ge 3.10$
* Virtual environment recommended (`venv` or `conda`)

### Setup

```bash
# Clone repository
git clone [https://github.com/](https://github.com/)chiaragiavalisco/koopman-nonlinear-control.git
cd koopman-nonlinear-control

```

### Running the Scripts

```bash
# 1. Run Andronov-Hopf bifurcation analysis
python hopf_bifurcation.py

# 2. Run Van der Pol EDMDc + LQR control experiment
python vdp_control_lqr.py

# 3. Run the full Van der Pol benchmark suite (DMD, EDMD, DMaps, EDMDc)
python vdp_koopman_benchmark.py

```

---

## 🔮 Future Extensions: Large-Scale Systems (PDEs)

As an ongoing research extension of this thesis, the methodology developed with **Diffusion Maps and Koopman Operator Theory** is being scaled to high-dimensional systems governed by partial differential equations (PDEs), with a primary focus on the **incompressible Navier–Stokes equations**:

1. High-dimensional flow snapshots are projected onto a low-dimensional latent manifold parameterised by Diffusion Maps eigenfunctions.
2. The Koopman operator is approximated directly on this intrinsic latent space, bypassing the curse of dimensionality while retaining global linear predictive capabilities for turbulent fluid flows.

---

## 📚 Key References

* **Koopman, B. O.** (1931). *Hamiltonian systems and transformation in Hilbert space*. PNAS, 17(5), 315–318.
* **Mezić, I.** (2005). *Spectral properties of dynamical systems, model reduction and decompositions*. Nonlinear Dynamics, 41(1–3), 309–325.
* **Williams, M. O., Kevrekidis, I. G., & Rowley, C. W.** (2015). *A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition*. Journal of Nonlinear Science, 25(6), 1307–1346.
* **Proctor, J. L., Brunton, S. L., & Kutz, J. N.** (2016). *Generalizing dynamic mode decomposition for inputs and control*. SIAM Journal on Applied Dynamical Systems, 15(1), 142–161.
* **Coifman, R. R., & Lafon, S.** (2006). *Diffusion maps*. Applied and Computational Harmonic Analysis, 21(1), 5–30.
* **Lehmberg, D. et al.** (2020). *datafold: data-driven models for point clouds and time series on manifolds*. JOSS, 5(51), 2283.

---

## 👤 Author

* **Chiara Giavalisco**
* Master's Thesis Project
* [LinkedIn Profile](https://www.linkedin.com/in/chiara-giavalisco-28b1b9268/) • [GitHub Profile](https://github.com/chiaragiavalisco) • [Email](mailto:chiara.giavalisco@gmail.com)
```

```
