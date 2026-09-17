# ==============================================================================
# Van der Pol (VdP) System: EDMD Control and LQR Trajectory Tracking
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from tqdm import tqdm

from datafold import (
    EDMD,
    InitialCondition,
    TSCDataFrame,
    TSCTransformerMixin,
)
from datafold.appfold.mpc import LQR
from datafold.dynfold.dmd import DMDControl
from datafold.utils._systems import VanDerPol

# ------------------------------------------------------------------------------
# 2. System Configuration & Training Data Generation
# ------------------------------------------------------------------------------
# Note that it may not converge when choosing another seed (defaults to 55)
rng = np.random.default_rng(55)

vdp = VanDerPol(control_coord="y")

n_timeseries = 20  # Number of timeseries in training set
n_timesteps = 200   # How many timesteps for every time series
dt = 0.01          # Delta time
time_values = np.arange(0, n_timesteps * dt, dt)

# Generate initial conditions
X_ic = rng.uniform(-3.0, 3.0, size=(n_timeseries, 2))
idx = pd.MultiIndex.from_arrays(
    [np.arange(n_timeseries), np.zeros(n_timeseries)]
)
X_ic = TSCDataFrame(X_ic, index=idx, columns=vdp.feature_names_in_)

# Generate constant control input profiles
U_tsc = rng.uniform(-3.0, 3.0, size=(n_timeseries, 1, 1))
U_tsc = np.tile(U_tsc, (1, n_timesteps - 1, 1))
U_tsc = TSCDataFrame.from_tensor(
    U_tsc,
    time_series_ids=X_ic.ids,
    feature_names=vdp.control_names_in_,
    time_values=time_values[:-1],
)

# Simulate system response
X_tsc, U_tsc = vdp.predict(X_ic, U=U_tsc)
print(X_tsc)
print(U_tsc)

# ------------------------------------------------------------------------------
# 3. Plot Training Trajectories
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 7))

for i in X_tsc.ids:
    idx = pd.IndexSlice[i, :]
    plt.plot(
        X_tsc.loc[idx, "x1"].to_numpy(),
        X_tsc.loc[idx, "x2"].to_numpy(),
        label=f"{i}",
    )

plt.xlabel(r"$x_1$", fontsize=12)
plt.ylabel(r"$x_2$", fontsize=12)
plt.grid()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
plt.show()

# ------------------------------------------------------------------------------
# 4. Custom Dictionary & EDMD Model Fitting
# ------------------------------------------------------------------------------
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

edmd = EDMD(
    dict_steps=[("vdpdict", VdPDictionary())],
    dmd_model=DMDControl(),
    include_id_state=False,
)
edmd.fit(X_tsc, U=U_tsc)

# ------------------------------------------------------------------------------
# 5. Out-of-Sample Setup & LQR Controller Configuration
# ------------------------------------------------------------------------------
# Number of time steps and time values for controlled time series
n_timesteps_oos = 500
time_values_oos = np.linspace(
    0, n_timesteps_oos * X_tsc.delta_time, n_timesteps_oos
)

# Random initial state
X_ic_oos = rng.uniform(-3, 3, size=(1, 2))
X_ic_oos = InitialCondition.from_array(
    X_ic_oos, feature_names=edmd.feature_names_in_, time_value=0
)

# Target state (origin)
target_state = InitialCondition.from_array(
    np.array([0, 0]), feature_names=edmd.feature_names_in_, time_value=0
)

# Setup Linear Quadratic Regulator
lqr = LQR(
    edmd=edmd,
    cost_running=np.array([1, 1, 0, 0]),
    cost_input=1e-2,
)
lqr.preset_target_state(target_state)

# ------------------------------------------------------------------------------
# 6. Closed-Loop Simulation
# ------------------------------------------------------------------------------
# Allocate data structures and fill in the following system loop
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

# Simulate uncontrolled reference trajectory from identical initial state
trajectory_uncontrolled, _ = vdp.predict(
    X_ic_oos,
    U=np.zeros((n_timesteps_oos - 1)),
    time_values=time_values_oos,
)

# ------------------------------------------------------------------------------
# 7. Results Visualization
# ------------------------------------------------------------------------------
# Phase-space comparison: Controlled vs. Uncontrolled
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
plt.xlabel(r"$x_1$", fontsize=14)
plt.ylabel(r"$x_2$", fontsize=14)
plt.legend(fontsize=14)
plt.grid()

# Time series of state variables
plt.figure(figsize=(10, 7))
plt.plot(
    X_oos.time_values(),
    X_oos.loc[:, "x1"].to_numpy(),
    c="black",
    label="x1",
)
plt.plot(
    X_oos.time_values(),
    X_oos.loc[:, "x2"].to_numpy(),
    c="blue",
    label="x2 (controlled)",
)
plt.ylabel(r"$x_1, x_2$", fontsize=14)
plt.xlabel(r"$t$", fontsize=14)
plt.legend(fontsize=14)
plt.grid()

# State norm convergence over time
plt.figure(figsize=(10, 7))
plt.plot(
    np.linalg.norm(X_oos.to_numpy(), axis=1),
    c="blue",
    label="state",
)
plt.axhline(
    np.linalg.norm(target_state),
    c="red",
    label="target",
)
plt.xlabel(r"$t_k$", fontsize=14)
plt.ylabel(r"$\|\mathbf{x}_k\|$", fontsize=14)
plt.legend(fontsize=14)
plt.grid()
