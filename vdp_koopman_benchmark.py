# ==============================================================================
# Koopman Analysis of the Van der Pol (VdP) System
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------------------
import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import datafold.pfold as pfold
from datafold import (
    EDMD,
    EDMDCV,
    DMDStandard,
    GaussianKernel,
    InitialCondition,
    TSCDataFrame,
    TSCPolynomialFeatures,
    TSCRadialBasis,
    TSCTransformerMixin,
)
from datafold.appfold.mpc import LQR
from datafold.dynfold import DiffusionMaps
from datafold.dynfold.dmd import DMDControl
from datafold.pcfold.timeseries.metric import TSCKFoldTime
from datafold.utils._systems import VanDerPol
from datafold.utils.general import generate_2d_regular_mesh
from datafold.utils.plot import plot_eigenvalues

# ------------------------------------------------------------------------------
# 2. Configuration & Training Data Generation
# ------------------------------------------------------------------------------
n_timeseries = 20  # Number of trajectories
n_timesteps = 200  # Time points per trajectory
dt = 0.01          # Time step
time_values = np.arange(0, n_timesteps * dt, dt)  # Time grid (length = n_timesteps)
rng = np.random.default_rng(55)

vdp = VanDerPol()

# Initial conditions
X_ic = rng.uniform(-3.0, 3.0, size=(n_timeseries, 2))
idx = pd.MultiIndex.from_arrays(
    [np.arange(n_timeseries), np.zeros(n_timeseries)]
)
X_ic = TSCDataFrame(
    X_ic, index=idx, columns=vdp.feature_names_in_
)

# Simulate Van der Pol system
# The second object of vdp.predict gives the control variables 
# that in this case we don't have, so it is all 0
X_tsc, _ = vdp.predict(X_ic, time_values=time_values)

print("X_ic:", X_ic)
print("X_ic.shape", X_ic.shape)  # (n_timeseries, 2)
print("X_tsc:", X_tsc)
print("X_tsc.shape:", X_tsc.shape)  # (n_timeseries * n_timesteps, 2)
print(f"time delta: {X_tsc.delta_time}")
print(f"nr. time series: {X_tsc.n_timeseries}")
print(f"nr. timesteps per time series: {X_tsc.n_timesteps}")
print(f"(n_samples, n_features): {X_tsc.shape}")
print(f"time interval {X_tsc.time_interval()}")
print(f"Same time values: {X_tsc.is_same_time_values()}")
print("")
print("Data snippet fo training data:")
X_tsc

# ------------------------------------------------------------------------------
# 3. Helper Functions & Initial Trajectory Plotting
# ------------------------------------------------------------------------------
# Function to add a single arrow in the following time series plots
idx_arrow = np.array(
    [time_values.shape[0] // 2 - 1, time_values.shape[0] // 2]
)

def include_arrow(ax, df):
    arrow = df.iloc[idx_arrow, :]
    ax.arrow(
        arrow.iloc[0, 0],
        arrow.iloc[0, 1],
        dx=arrow.iloc[1, 0] - arrow.iloc[0, 0],
        dy=arrow.iloc[1, 1] - arrow.iloc[0, 1],
        color="black",
        head_width=0.05,
    )

fig, ax = plt.subplots(figsize=[5, 5])

for _id, df in X_tsc.itertimeseries():
    ax.plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax, df)

ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()

# ------------------------------------------------------------------------------
# 4. Method 1: Standard DMD
# ------------------------------------------------------------------------------
dmd = DMDStandard().fit(
    X=X_tsc, store_system_matrix=True
)  # X must be of type TSCDataFrame

dmd_values = dmd.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)
print("Data snipped for predicted time series training data", dmd_values)

# Perform out-of-sample prediction
n_timesteps_oos = 1000
time_values_oos = np.linspace(
    0, n_timesteps_oos * X_tsc.delta_time, n_timesteps_oos
)

x0 = [-2, 1]

idx = pd.MultiIndex.from_arrays(
    [[0], [0.0]],  # Single time series, time zero
    names=["ts_id", "time"],
)

X_ic_oos = TSCDataFrame(
    pd.DataFrame([x0], index=idx, columns=vdp.feature_names_in_)
)
print(X_ic_oos)

dmd_values_oos = dmd.predict(
    X_ic_oos, time_values=time_values_oos
)
print("Data snipped for out-of-sample prediction", dmd_values_oos)

# Plot training vs DMD in-sample & out-of-sample
f, ax = plt.subplots(1, 2, figsize=(14, 5))

for _id, df in X_tsc.itertimeseries():
    ax[0].plot(df["x1"], df["x2"], 0.1, c="black")
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

first = True
for _id, df in dmd_values.itertimeseries():
    ax[1].plot(
        df["x1"],
        df["x2"],
        0.1,
        c="black",
    )
    include_arrow(ax[1], df)
    first = False
include_arrow(ax[1], dmd_values_oos)

handles, labels = ax[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys())

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Generate true out-of-sample reference trajectory
true_values_oos, _ = vdp.predict(
    dmd_values_oos.initial_states(), time_values=time_values_oos
)

# Comparison between true trajectory and DMD prediction
f, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    true_values_oos.loc[:, "x1"].to_numpy(),
    true_values_oos.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax, true_values_oos)
ax.plot(
    dmd_values_oos.loc[:, "x1"].to_numpy(),
    dmd_values_oos.loc[:, "x2"].to_numpy(),
    c="orange",
    label="dmd",
)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()

# Error analysis for DMD
f, ax = plt.subplots(1, 2, figsize=(14, 5))

error = np.linalg.norm(
    dmd_values_oos[["x1", "x2"]].to_numpy()
    - true_values_oos[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(true_values_oos[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[1].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 5. Method 2: EDMD with Cross-Validated Polynomial Dictionary
# ------------------------------------------------------------------------------
# 1. Define EDMD pipeline with dictionary steps
edmd_model = EDMD(
    dict_steps=[("polynomial", TSCPolynomialFeatures())],
    include_id_state=True,
)

# 2. Parameter grid
param_grid = {"polynomial__degree": [2, 3, 4, 5, 6, 7]}

# 3. Cross-validation setup
grid_search = EDMDCV(
    estimator=edmd_model,
    param_grid=param_grid,
    cv=TSCKFoldTime(n_splits=3),
    n_jobs=-1,
    verbose=3,
)

# 4. Run grid search
grid_search.fit(X=X_tsc)

# 5. Best model and evaluation
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_edmd_model = grid_search.best_estimator_

edmd_poly_values = best_edmd_model.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)
print(edmd_poly_values.shape)

koopman_modes = best_edmd_model.koopman_modes
koopman_eigenvalues = best_edmd_model.koopman_eigenvalues

print("Eigenvalues of Koopman matrix:")
print(koopman_eigenvalues)

print("Eigenvectors of Koopman matrix:")
print(koopman_modes)

ax = plot_eigenvalues(koopman_eigenvalues, plot_unit_circle=True)
ax.grid()

magnitudes = np.abs(koopman_eigenvalues)

plt.figure(figsize=(12, 5))
plt.stem(magnitudes)
plt.title("Eigenvalue magnitudes")
plt.xlabel("Index")
plt.ylabel(r"$ |\lambda| $")
plt.xticks(np.arange(0, len(magnitudes) + 1))
plt.grid(True)

print(best_edmd_model.named_steps["polynomial"])
print("")
print("polynomial degrees for data (first column 'x1' and second 'x2'):")
print(best_edmd_model.named_steps["polynomial"].powers_)
print("")
print("Dictionary space values:")
print(best_edmd_model.transform(X_tsc))

len_koopman_matrix = len(
    best_edmd_model.named_steps["dmd"].eigenvectors_right_
)
print(f"shape of Koopman matrix: {len_koopman_matrix} x {len_koopman_matrix}")

# Plot in-sample polynomial EDMD comparisons
f, ax = plt.subplots(1, 2, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in edmd_poly_values.itertimeseries():
    ax[1].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Out-of-sample prediction
ground_truth, _ = vdp.predict(X_ic_oos, time_values=time_values_oos)
predicted = best_edmd_model.predict(X_ic_oos, time_values=time_values_oos)

f, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    ground_truth.loc[:, "x1"].to_numpy(),
    ground_truth.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax, ground_truth)
ax.plot(
    predicted.loc[:, "x1"].to_numpy(),
    predicted.loc[:, "x2"].to_numpy(),
    c="orange",
    label="edmd_poly",
)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()

# Error analysis (best polynomial model)
f, ax = plt.subplots(1, 2, figsize=(15, 5))
error = np.linalg.norm(
    predicted[["x1", "x2"]].to_numpy()
    - ground_truth[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(ground_truth[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[1].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 6. Method 3: EDMD with Fixed 3-Degree Polynomial Dictionary
# ------------------------------------------------------------------------------
edmd_model_3 = EDMD(
    dict_steps=[("polynomial", TSCPolynomialFeatures(3))],
    include_id_state=True,
).fit(X=X_tsc)

edmd_poly3_values = edmd_model_3.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

koopman_modes = edmd_model_3.koopman_modes
koopman_eigenvalues = edmd_model_3.koopman_eigenvalues

print("Eigenvalues of Koopman matrix:")
print(koopman_eigenvalues)

print("Eigenvectors of Koopman matrix:")
print(koopman_modes)

ax = plot_eigenvalues(koopman_eigenvalues, plot_unit_circle=True)
ax.grid()

magnitudes = np.abs(koopman_eigenvalues)

plt.figure(figsize=(5, 5))
plt.stem(magnitudes)
plt.title("Eigenvalue magnitudes")
plt.xlabel("Index")
plt.ylabel(r"$|\lambda|$")
plt.xticks(np.arange(0, len(magnitudes) + 1))
plt.grid(True)

print(edmd_model_3.named_steps["polynomial"])
print("")
print("polynomial degrees for data (first column 'x1' and second 'x2'):")
print(edmd_model_3.named_steps["polynomial"].powers_)
print("")
print("Dictionary space values:")
print(edmd_model_3.transform(X_tsc))

len_koopman_matrix = len(
    edmd_model_3.named_steps["dmd"].eigenvectors_right_
)
print(f"shape of Koopman matrix: {len_koopman_matrix} x {len_koopman_matrix}")

f, ax = plt.subplots(1, 2, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in edmd_poly3_values.itertimeseries():
    ax[1].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Out-of-sample prediction (degree 3)
predicted = edmd_model_3.predict(X_ic_oos, time_values=time_values_oos)

f, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    ground_truth.loc[:, "x1"].to_numpy(),
    ground_truth.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax, ground_truth)
ax.plot(
    predicted.loc[:, "x1"].to_numpy(),
    predicted.loc[:, "x2"].to_numpy(),
    c="orange",
    label="edmd_poly",
)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()

# Error analysis (degree 3)
f, ax = plt.subplots(1, 2, figsize=(15, 5))
error = np.linalg.norm(
    predicted[["x1", "x2"]].to_numpy()
    - ground_truth[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(ground_truth[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 7. Method 4: EDMD with Radial Basis Function (RBF) Dictionary
# ------------------------------------------------------------------------------
dict_step_rbf = [
    (
        "rbf",
        TSCRadialBasis(
            kernel=GaussianKernel(),
            center_type="initial_condition",
        ),
    )
]

edmd_model_rbf = EDMD(
    dict_steps=dict_step_rbf,
    include_id_state=True,
)

param_grid_rbf = {"rbf__kernel__epsilon": np.logspace(-2, 1, 10)}

edmd_cv = EDMDCV(
    estimator=edmd_model_rbf,
    param_grid=param_grid_rbf,
    cv=TSCKFoldTime(n_splits=5),
    verbose=2,
    n_jobs=-1,
)

edmd_cv.fit(X=X_tsc)

print(
    f"Best epsilon parameter: {edmd_cv.best_params_['rbf__kernel__epsilon']}"
)
print(f"Best score: {-edmd_cv.best_score_}")  # neg_mean_squared_error converted to positive

best_edmd_model_rbf = edmd_cv.best_estimator_

edmd_rbf_values = best_edmd_model_rbf.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

print("")
print("Dictionary space values:")
print(best_edmd_model_rbf.transform(X_tsc))

print("")
len_koopman_matrix = len(
    best_edmd_model_rbf.named_steps["dmd"].eigenvectors_right_
)
print(f"shape of Koopman matrix: {len_koopman_matrix} x {len_koopman_matrix}")

koopman_modes = best_edmd_model_rbf.koopman_modes
koopman_eigenvalues = best_edmd_model_rbf.koopman_eigenvalues

print("Eigenvalues of Koopman matrix:")
print(koopman_eigenvalues)

print("Eigenvectors of Koopman matrix:")
print(koopman_modes)

ax = plot_eigenvalues(koopman_eigenvalues, plot_unit_circle=True)
ax.grid()

magnitudes = np.abs(koopman_eigenvalues)

plt.figure(figsize=(7, 5))
plt.stem(magnitudes)
plt.title("Eigenvalue magnitudes")
plt.xlabel("Index")
plt.ylabel(r"$|\lambda|$")
plt.xticks(np.arange(0, len(magnitudes) + 1))
plt.grid(True)

f, ax = plt.subplots(1, 2, sharey=True, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in edmd_rbf_values.itertimeseries():
    ax[1].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Out-of-sample prediction (RBF)
predicted = best_edmd_model_rbf.predict(
    X_ic_oos, time_values=time_values_oos
)

f, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    ground_truth.loc[:, "x1"].to_numpy(),
    ground_truth.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax, ground_truth)
ax.plot(
    predicted.loc[:, "x1"].to_numpy(),
    predicted.loc[:, "x2"].to_numpy(),
    c="orange",
    label="edmd_poly",
)
ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()

# Error analysis (RBF)
f, ax = plt.subplots(1, 2, figsize=(15, 5))
error = np.linalg.norm(
    predicted[["x1", "x2"]].to_numpy()
    - ground_truth[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(ground_truth[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 8. Method 5: EDMD with Custom VdP Dictionary
# ------------------------------------------------------------------------------
# Out-of-sample setups
n_timesteps_oos = 1000
time_values_oos = np.linspace(
    0, n_timesteps_oos * X_tsc.delta_time, n_timesteps_oos
)

# First initial condition
x0 = [-2, 1]
idx = pd.MultiIndex.from_arrays(
    [[0], [0.0]], names=["ts_id", "time"]
)
X_ic_oos = TSCDataFrame(
    pd.DataFrame([x0], index=idx, columns=vdp.feature_names_in_)
)

# Second initial condition
x0 = [2, 1]
X_ic_oos1 = TSCDataFrame(
    pd.DataFrame([x0], index=idx, columns=vdp.feature_names_in_)
)

class VdPDictionary(BaseEstimator, TSCTransformerMixin):
    def _more_tags(self):
        return dict(tsc_contains_orig_states=True)

    def get_feature_names_out(self, input_features=None):
        return ["x1^2", "x1^2 * x2"]

    def fit(self, X, y=None):
        self._setup_feature_attrs_fit(X)
        return self

    def transform(self, X: TSCDataFrame):
        X = X.copy()
        X["x1^2"] = np.square(X.loc[:, "x1"].to_numpy())
        X["x1^2 * x2"] = X["x1^2"].to_numpy() * X["x2"].to_numpy()
        return X

edmd_vdp1 = EDMD(
    dict_steps=[("vdpdict", VdPDictionary())],
    include_id_state=True,
)
edmd_vdp1.fit(X=X_tsc)

edmd_vdp1_values = edmd_vdp1.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

print(edmd_vdp1.named_steps["vdpdict"])
print("")
print("Dictionary space values:")
print(edmd_vdp1.transform(X_tsc))

len_koopman_matrix = len(
    edmd_vdp1.named_steps["dmd"].eigenvectors_right_
)
print(f"shape of Koopman matrix: {len_koopman_matrix} x {len_koopman_matrix}")

koopman_modes = edmd_vdp1.koopman_modes
koopman_eigenvalues = edmd_vdp1.koopman_eigenvalues

print("Eigenvalues of Koopman matrix:")
print(koopman_eigenvalues)

print("Eigenvectors of Koopman matrix:")
print(koopman_modes)

ax = plot_eigenvalues(koopman_eigenvalues, plot_unit_circle=True)
ax.grid()

magnitudes = np.abs(koopman_eigenvalues)

plt.figure(figsize=(5, 5))
plt.stem(magnitudes)
plt.title("Eigenvalue magnitudes")
plt.xlabel("Index")
plt.ylabel(r"$|\lambda|$")
plt.xticks(np.arange(0, len(magnitudes) + 1))
plt.grid(True)

f, ax = plt.subplots(1, 2, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in edmd_vdp1_values.itertimeseries():
    ax[1].plot(
        df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black"
    )
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Out-of-sample predictions across two initial conditions
ground_truth, _ = vdp.predict(X_ic_oos, time_values=time_values_oos)
predicted = edmd_vdp1.predict(X_ic_oos, time_values=time_values_oos)
ground_truth1, _ = vdp.predict(X_ic_oos1, time_values=time_values_oos)
predicted1 = edmd_vdp1.predict(X_ic_oos1, time_values=time_values_oos)

f, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(
    ground_truth.loc[:, "x1"].to_numpy(),
    ground_truth.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax[0], ground_truth)
ax[0].plot(
    predicted.loc[:, "x1"].to_numpy(),
    predicted.loc[:, "x2"].to_numpy(),
    c="orange",
    label="edmd_vdp",
)
ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()
ax[0].legend()

ax[1].plot(
    ground_truth1.loc[:, "x1"].to_numpy(),
    ground_truth1.loc[:, "x2"].to_numpy(),
    label="true system",
)
include_arrow(ax[1], ground_truth)
ax[1].plot(
    predicted1.loc[:, "x1"].to_numpy(),
    predicted1.loc[:, "x2"].to_numpy(),
    c="orange",
    label="edmd_vdp",
)
ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()
ax[1].legend()

# Error analysis (Condition 1)
f, ax = plt.subplots(1, 2, figsize=(15, 5))
error = np.linalg.norm(
    predicted[["x1", "x2"]].to_numpy()
    - ground_truth[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(ground_truth[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Error analysis (Condition 2)
f, ax = plt.subplots(1, 2, figsize=(15, 5))
error = np.linalg.norm(
    predicted1[["x1", "x2"]].to_numpy()
    - ground_truth1[["x1", "x2"]].to_numpy(),
    axis=1,
)

ax[0].plot(error, color="darkred", linewidth=2)
ax[0].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[0].set_ylabel(r"$\|x_{pred} - x_{true}\|_2$", fontsize=14)
ax[0].grid(True)

error_normalized = (
    error
    / np.linalg.norm(ground_truth1[["x1", "x2"]].to_numpy(), axis=1)
    * 100
)

ax[1].plot(error_normalized, color="darkred", linewidth=2)
ax[1].axvline(x=200, color="blue", linestyle="--", linewidth=1)
ax[0].set_xlabel(r"$t_k$", fontsize=14)
ax[1].set_ylabel(
    r"$\|x_{pred} - x_{true}\|_2 / \|x_{true}\|_2\ \% 100$", fontsize=14
)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 9. Method 6: EDMD with Control (EDMDc) & LQR Regulation
# ------------------------------------------------------------------------------
# Configuration
n_timeseries = 20
n_timesteps = 200
dt = 0.01
time_values = np.arange(0, n_timesteps * dt, dt)
rng = np.random.default_rng(55)

vdp = VanDerPol(control_coord="y")

# Initial conditions & controls
X_ic = rng.uniform(-3.0, 3.0, size=(n_timeseries, 2))
idx = pd.MultiIndex.from_arrays(
    [np.arange(n_timeseries), np.zeros(n_timeseries)]
)
X_ic = TSCDataFrame(X_ic, index=idx, columns=vdp.feature_names_in_)

U_tsc = rng.uniform(-3.0, 3.0, size=(n_timeseries, 1, 1))
U_tsc = np.tile(U_tsc, (1, n_timesteps - 1, 1))
U_tsc = TSCDataFrame.from_tensor(
    U_tsc,
    time_series_ids=X_ic.ids,
    feature_names=vdp.control_names_in_,
    time_values=time_values[:-1],
)

X_tsc, U_tsc = vdp.predict(X_ic, U=U_tsc)

n_timesteps_oos = 4000
time_values_oos = np.linspace(
    0, n_timesteps_oos * X_tsc.delta_time, n_timesteps_oos
)

# Initial and target states
X_ic_oos = rng.uniform(-3, 3, size=(1, 2))
X_ic_oos = InitialCondition.from_array(
    X_ic_oos, feature_names=["x1", "x2"], time_value=0
)

target_state = InitialCondition.from_array(
    np.array([0, 0]), feature_names=["x1", "x2"], time_value=0
)

# EDMDc with RBF dictionary
dict_step_rbf = [
    (
        "rbf",
        TSCRadialBasis(
            kernel=GaussianKernel(1.2),
            center_type="initial_condition",
        ),
    )
]

edmd = EDMD(
    dict_steps=dict_step_rbf,
    dmd_model=DMDControl(),
    include_id_state=True,
).fit(X_tsc, U=U_tsc)

target_state = InitialCondition.from_array(
    np.array([0, 0]), feature_names=["x1", "x2"], time_value=0
)

cost_rbf = np.zeros(22)
cost_rbf[:2] = 1  # Set first two elements (x1, x2) to 1
lqr = LQR(edmd=edmd, cost_running=cost_rbf, cost_input=1e-2)
lqr.preset_target_state(target_state)

# Simulation loop
X_oos = TSCDataFrame.from_array(
    np.zeros((n_timesteps_oos, 2)),
    feature_names=vdp.feature_names_in_,
    time_values=time_values_oos,
)
U_oos = TSCDataFrame.from_array(
    np.zeros((n_timesteps_oos - 1, 1)),
    feature_names=vdp.control_names_in_,
    time_values=time_values_oos[:-1],
)
X_oos.iloc[0, :] = X_ic_oos.to_numpy()

for i in tqdm(range(1, n_timesteps_oos)):
    state = X_oos.iloc[[i - 1], :]
    U_oos.iloc[i - 1, :] = lqr.control_sequence(X=state)
    new_state, _ = vdp.predict(
        state,
        U=U_oos.iloc[[i - 1], :],
        time_values=time_values_oos[i - 1 : i + 1],
    )
    X_oos.iloc[i, :] = new_state.iloc[[1], :].to_numpy()

# Controlled trajectory phase plot
plt.figure(figsize=(10, 7))
plt.plot(
    X_oos.loc[:, "x1"].to_numpy(),
    X_oos.loc[:, "x2"].to_numpy(),
    c="red",
    label="controlled traj.",
)
plt.quiver(
    *X_oos.to_numpy()[:-1, :].T,
    *np.column_stack(
        [
            np.zeros_like(U_oos.to_numpy()),
            U_oos.to_numpy() / X_oos.delta_time,
        ]
    ).T,
    color="blue",
    label="control",
)
plt.plot(X_oos.iloc[0, 0], X_oos.iloc[0, 1], "o", c="red")
plt.plot(
    trajectory_uncontrolled.loc[:, "x1"].to_numpy(),
    trajectory_uncontrolled.loc[:, "x2"].to_numpy(),
    c="black",
    label="uncontrolled traj.",
)
plt.plot(
    trajectory_uncontrolled.iloc[0, 0],
    trajectory_uncontrolled.iloc[0, 1],
    "o",
    c="black",
    label="initial state",
)
plt.plot(
    target_state.iloc[0, 0],
    target_state.iloc[0, 1],
    "*",
    c="black",
    label="target state",
)
plt.xlabel(r"$x_1$", fontsize=16)
plt.ylabel(r"$x_2$", fontsize=16)
plt.legend()
plt.grid()

# Time series plot
plt.figure(figsize=(10, 7))
plt.plot(
    X_oos.time_values(),
    X_oos.loc[:, "x1"].to_numpy(),
    c="black",
    label=r"$x_1$",
)
plt.plot(
    X_oos.time_values(),
    X_oos.loc[:, "x2"].to_numpy(),
    c="blue",
    label=r"$x_2$ (controlled)",
)
plt.xlabel(r"$t$", fontsize=16)
plt.ylabel(r"$x_1, x_2$", fontsize=16)
plt.legend(fontsize=14)
plt.grid()

# Convergence norm plot
plt.figure(figsize=(10, 7))
plt.plot(
    np.linalg.norm(X_oos.to_numpy(), axis=1),
    c="blue",
    label=r"state $(x_1,x_2)$",
)
plt.axhline(
    np.linalg.norm(target_state),
    c="red",
    label=r"target $(0,0)$",
)
plt.xlabel(r"$t_k$", fontsize=16)
plt.ylabel(r"$\| \mathbf{x}_k \|$", fontsize=16)
plt.grid()
plt.legend(fontsize=14)

# ------------------------------------------------------------------------------
# 10. Method 7: EDMD with Diffusion Maps
# ------------------------------------------------------------------------------
# Compute epsilon as median pairwise distance
D = pairwise_distances(X_tsc, metric="euclidean")
dists = D[np.triu_indices(len(X_tsc), k=1)]
epsilon = np.median(dists)
print("epsilon =", epsilon)

X_pcm = pfold.PCManifold(X_tsc)

# Fit Diffusion Maps
t0 = time.time()
dmap = DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=epsilon),
    n_eigenpairs=10,
)
dmap = dmap.fit(X_pcm)

X_dmap = dmap.transform(X_pcm)

print("Embedding shape:", X_dmap.shape)
print(f"Diffusion map done in {time.time() - t0:0.3f} seconds.")

# TSCDataFrame for transformed diffusion map features
X_dmap = TSCDataFrame(
    X_dmap,
    index=X_tsc.index,
    columns=[f"psi{i+1}" for i in range(dmap.n_eigenpairs)],
)

evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

# Excluding the trivial eigenvalue and its respective eigenvector
evals_red = evals[1:]
evecs_red = evecs[:, 1:]

def custom_plot_eigenvalues(
    eigenvalues: np.ndarray,
    *,
    plot_unit_circle: bool = False,
    semilogy: bool = False,
    ax=None,
    subplot_kwargs: Optional[dict[str, object]] = None,
    plot_kwargs: Optional[dict[str, object]] = None,
):
    """Plots eigenvalue distribution."""
    if ax is None:
        _, ax = plt.subplots(
            {} if subplot_kwargs is None else subplot_kwargs
        )

    plot_kwargs = plot_kwargs or {}
    plot_kwargs.setdefault("marker", "*")
    plot_kwargs.setdefault("linewidth", 0)

    if eigenvalues.dtype == complex:
        ax.plot(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            **plot_kwargs,
        )

        if plot_unit_circle:
            circle_values = np.linspace(0, 2 * np.pi, 3000)
            ax.plot(
                np.cos(circle_values),
                np.sin(circle_values),
                "-",
                color="gray",
            )
            ax.set_aspect("equal")

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_xlabel(r"$\Re(\lambda)$")
            ax.set_ylabel(r"$\Im(\lambda)$")

    elif eigenvalues.dtype == float:
        if plot_unit_circle:
            warnings.warn(
                "eigenvalues are real-valued, 'plot_unit_circle=True' is ignored",
                stacklevel=2,
            )

        eigenvalues = np.sort(eigenvalues.copy())[::-1]

        _ylabel_text = r" $\lambda_k$"
        if semilogy:
            ax.semilogy(
                np.arange(len(eigenvalues)),
                eigenvalues,
                **plot_kwargs,
            )
            _ylabel_text = _ylabel_text + " (log scale)"
        else:
            ax.plot(
                np.arange(len(eigenvalues)),
                eigenvalues,
                **plot_kwargs,
            )

        with plt.rc_context(rc={"text.usetex": True}):
            ax.set_ylabel(_ylabel_text, fontsize=16)

        ax.set_xlabel(r"$k$", fontsize=16)

    return ax

num_evals = len(evals_red)
tick_positions = np.arange(num_evals)
tick_labels = np.arange(1, num_evals + 1)

ax = custom_plot_eigenvalues(evals_red, plot_unit_circle=True)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.grid()
plt.show()

# ------------------------------------------------------------------------------
# 11. Diffusion Coordinates Residual Analysis & Geometric Embeddings
# ------------------------------------------------------------------------------
def linearFit(evecs, k):
    """
    Python version of MATLAB linearFit(evecs, k).
    Compares the eigendirection given by an eigenvector to the directions 
    given by the previous eigenvectors.
    """
    m = evecs.shape[0]

    # PHI = [1, evecs[:, 0], evecs[:, 1], ... evecs[:, k-1]]
    PHI = np.hstack([np.ones((m, 1)), evecs[:, :k]])
    phi = evecs[:, k]

    pdists = pdist(PHI[:, 1:])
    ereg = np.median(pdists) / 3
    sqpdists = squareform(pdists)

    approx = np.zeros_like(phi)

    def kernel(d):
        return np.exp(-d**2 / (ereg**2))

    for i in range(m):
        curdists = sqpdists[i, :]
        dists = np.delete(curdists, i)
        W = np.diag(kernel(dists))

        curPHI = np.delete(PHI, i, axis=0)
        curphi = np.delete(phi, i, axis=0)

        curPHI_W = curPHI.T @ W

        approx[i] = PHI[i, :] @ (
            pinv(curPHI_W @ curPHI) @ (curPHI_W @ curphi)
        )

    res = np.linalg.norm(phi - approx) / np.linalg.norm(phi)
    return res

def compute_residuals(evecs, num_obs=1000):
    """Replicates the MATLAB loop computing r1(j)."""
    m, num_evecs = evecs.shape

    idx = np.random.choice(m, num_obs, replace=False)
    sub_evecs = evecs[idx, :]

    r = np.zeros(num_evecs)
    r[0] = 1

    for k in range(1, num_evecs):
        print(f"Computing residual for eigenvector {k+1}/{num_evecs}")
        r[k] = linearFit(sub_evecs, k)

    return r

# Plot residuals
r1 = compute_residuals(evecs_red)

num_evecs = len(r1)
tick_positions = np.arange(num_evecs)
tick_labels = np.arange(1, num_evecs + 1)

fig, ax = plt.subplots()
ax.scatter(range(len(r1)), r1)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlabel(r"$k$", fontsize=16)
ax.set_ylabel(r"$r_k$", fontsize=16)
ax.grid()
plt.show()

def plot_pairwise_eigenvector(
    eigenvectors: np.ndarray,
    n: int,
    idx_start=0,
    label=r"$\zeta_1$",
    scatter_params: Optional[dict] = None,
    fig_params: Optional[dict] = None,
):
    eigenvectors = np.asarray(eigenvectors)
    n_eigenvectors = eigenvectors.shape[1] - 1

    fig_params = {} if fig_params is None else fig_params
    ncols = fig_params.pop("ncols", 2)
    nrows = fig_params.pop("nrows", int(np.ceil(n_eigenvectors / 2)))
    sharex = fig_params.pop("sharex", True)
    sharey = fig_params.pop("sharey", True)

    f, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        **fig_params,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[ax]])
    elif nrows == 1:
        axes = np.array([ax])
    elif ncols == 1:
        axes = np.array([[a] for a in ax])
    else:
        axes = ax

    correct_one = 0
    dummy_scatter = None

    for i, idx_eigvec in enumerate(range(n_eigenvectors + 1)):
        if i == n:
            correct_one = 1
            continue
        else:
            i = i - correct_one

        current_row = i // ncols
        current_col = i - current_row * ncols
        _ax = axes[current_row, current_col]

        sparams = {} if scatter_params is None else scatter_params

        sc = _ax.scatter(
            eigenvectors[:, n],
            eigenvectors[:, idx_eigvec],
            **sparams,
        )
        dummy_scatter = sc

        _ax.set_xlabel(
            rf"$\zeta_{{{(n + idx_start)}}}$",
            fontsize=16,
        )
        _ax.set_ylabel(
            rf"$\zeta_{{{(idx_eigvec + idx_start)}}}$",
            fontsize=16,
        )

    if dummy_scatter is not None:
        cbar = f.colorbar(
            dummy_scatter,
            ax=ax,
            location="right",
            fraction=0.05,
            pad=0.02,
        )
        cbar.set_label(label, fontsize=14)

    return f, ax

plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_[:, 1:6],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=X_tsc["x1"].to_numpy()),
)

# ------------------------------------------------------------------------------
# 12. Koopman Operator Approximation via Diffusion Coordinates & Reconstruction
# ------------------------------------------------------------------------------
X_dmap_np = X_dmap.values
X_dmap_np = X_dmap_np[:, 1:3]
X_dmap_np_T = X_dmap_np.T  # (n_eigenpairs, n_timesteps * n_timeseries)

Psi_minus = X_dmap_np_T[:, :-1]  # (n_eigenpairs, n_timesteps * n_timeseries - 1)
Psi_plus = X_dmap_np_T[:, 1:]    # (n_eigenpairs, n_timesteps * n_timeseries - 1)

K = Psi_plus @ np.linalg.pinv(Psi_minus)

# Eigen-decomposition of K
D, Phi = np.linalg.eig(K)
Omega = np.diag(D)

X_tsc_values_transposed = X_tsc.values.T
X_minus = X_tsc_values_transposed[:, :-1]

Psi_evals = Phi.T @ Psi_minus

# Compute Koopman modes C for original data X
C = X_minus @ np.linalg.pinv(Psi_evals)

# Prediction routine
predictions = []
current_psi = X_dmap_np_T[:, 0].reshape(-1, 1)

for i in range(X_dmap_np_T.shape[1]):
    x_predicted = C @ Omega @ (Phi.T @ current_psi)
    predictions.append(x_predicted)

    if i + 1 < X_dmap_np_T.shape[1]:
        current_psi = X_dmap_np_T[:, i + 1].reshape(-1, 1)

predictions = np.hstack(predictions)
predictions = predictions.T

predictions = TSCDataFrame(
    predictions, index=X_tsc.index, columns=X_tsc.columns
)

# Plot training data vs DMAP-Koopman predicted trajectories
f, ax = plt.subplots(1, 2, figsize=(14, 5))

for _id, df in X_tsc.itertimeseries():
    ax[0].plot(df["x1"], df["x2"], c="black")
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

predictions_real = TSCDataFrame(
    np.real(predictions.values),
    index=predictions.index,
    columns=predictions.columns,
)

for _id, df in predictions_real.itertimeseries():
    ax[1].plot(df["x1"], df["x2"], c="black")
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

plt.show()
