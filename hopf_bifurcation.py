# ==============================================================================
# Normal form of the Andronov-Hopf bifurcation
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from datafold import (
    EDMD,
    DMDStandard,
    GaussianKernel,
    TSCPolynomialFeatures,
    TSCRadialBasis,
)
from datafold.utils._systems import Hopf
from datafold.utils.general import generate_2d_regular_mesh

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------
# Function to add a single arrow in the following time series plots
idx_arrow = np.array([time_values.shape[0] // 2 - 1, time_values.shape[0] // 2]) if 'time_values' in locals() else None

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

# ------------------------------------------------------------------------------
# 3. System Initialization & Training Data Generation
# ------------------------------------------------------------------------------
system = Hopf()

# Set up training data by sampling original Hopf system
n_timesteps = 21
time_values = np.linspace(0, 0.4, n_timesteps)

# Update arrow index now that time_values is defined
idx_arrow = np.array([time_values.shape[0] // 2 - 1, time_values.shape[0] // 2])

X_ic = generate_2d_regular_mesh(
    low=(-2, -2),
    high=(2, 2),
    n_xvalues=8,
    n_yvalues=8,
    feature_names=system.feature_names_in_,
)
X_tsc = system.predict(X_ic, time_values=time_values)

print(f"time delta : {X_tsc.delta_time}")
print(f"nr. time series : {X_tsc.n_timeseries}")
print(f"nr. timesteps per time series : {X_tsc.n_timesteps}")
print(f"(n_samples , n_features ): {X_tsc.shape}")
print(f"time interval {X_tsc.time_interval()}")
print(f"Same time values : {X_tsc.is_same_time_values()}")
print("")
print("Data snippet for training data :")
X_tsc

# ------------------------------------------------------------------------------
# 4. Plot Training Trajectories
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=[5, 5])

for _id, df in X_tsc.itertimeseries():
    ax.plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax, df)

ax.set_xlabel(r"$x_1$", fontsize=12)
ax.set_ylabel(r"$x_2$", fontsize=12)
ax.axis("equal")
ax.grid()

# ------------------------------------------------------------------------------
# 5. Standard DMD
# ------------------------------------------------------------------------------
# X must be of type TSCDataFrame
dmd = DMDStandard().fit(X=X_tsc, store_system_matrix=True)

dmd_values = dmd.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

# Perform out-of-sample prediction (not contained in training data)
# Note that the time sampling is independent of the original samples
dmd_values_oos = dmd.predict(
    np.array([-1.8, 2]),
    time_values=np.linspace(0, 100, 1000),
)

print("Data snippet for predicted time series training data and out-of-sample prediction")
dmd_values
dmd_values_oos

# Plot DMD results
f, ax = plt.subplots(1, 2, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in dmd_values.itertimeseries():
    ax[1].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Generate red "out-of-sample" prediction, for extra analysis below
ax[1].plot(
    dmd_values_oos["x1"].to_numpy(),
    dmd_values_oos["x2"].to_numpy(),
    0.1,
    c="red",
    linewidth=3,
)
include_arrow(ax[1], dmd_values_oos)

# Stability analysis
generator_A = (dmd.system_matrix_ - np.eye(2)) / dmd.dt_
det = np.linalg.det(generator_A)
trace = np.trace(generator_A)

print("Relevant values for the stability analysis : \n")
print(f"determinant of A: {det}")
print(f"trace of A: {trace}")
print(f"Delta {1/4. * trace ** 2}")

# ------------------------------------------------------------------------------
# 6. EDMD with Polynomial Dictionary (Degree 3)
# ------------------------------------------------------------------------------
dict_step = [
    (
        "polynomial",
        TSCPolynomialFeatures(degree=3),
    )
]

edmd_poly = EDMD(
    dict_steps=dict_step,
    include_id_state=True,
).fit(X=X_tsc)

edmd_poly_values = edmd_poly.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

# Access models in the dictionary (name defined in dict_step above)
print(edmd_poly.named_steps["polynomial"])
print("")
print("polynomial degrees for data (first column 'x1' and second 'x2'):")
print(edmd_poly.named_steps["polynomial"].powers_)
print("")
print("Dictionary space values :")
edmd_poly.transform(X_tsc)

# Plot EDMD polynomial comparisons
f, ax = plt.subplots(1, 2, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")

for _id, df in edmd_poly_values.itertimeseries():
    ax[1].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")

# Make out-of-sample prediction (Polynomial)
X_ic_oos = np.array([[2, 1]])
time_values_oos = np.linspace(0, 7, 400)

ground_truth = system.predict(X_ic_oos, time_values=time_values_oos)
predicted = edmd_poly.predict(X_ic_oos, time_values=time_values_oos)

f, ax = plt.subplots(figsize=(5, 5))
ax.plot(
    ground_truth.loc[:, "x1"].to_numpy(),
    ground_truth.loc[:, "x2"].to_numpy(),
    label="True trajectory",
)
include_arrow(ax, ground_truth)

ax.plot(
    predicted.loc[:, "x1"].to_numpy(),
    predicted.loc[:, "x2"].to_numpy(),
    c="orange",
    label="EDMD prediction",
)

ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()

# ------------------------------------------------------------------------------
# 7. EDMD with Radial Basis Function (RBF) Dictionary
# ------------------------------------------------------------------------------
dict_step = [
    (
        "rbf",
        TSCRadialBasis(
            kernel=GaussianKernel(epsilon=0.17),
            center_type="initial_condition",
        ),
    )
]

# Note that the "extended" part is in the transformations
edmd_rbf = EDMD(
    dict_steps=dict_step,
    include_id_state=True,
).fit(X=X_tsc)

edmd_rbf_values = edmd_rbf.predict(
    X_tsc.initial_states(), time_values=X_tsc.time_values()
)

len_koopman_matrix = len(edmd_rbf.named_steps["dmd"].eigenvectors_right_)
print(f"shape of Koopman matrix : {len_koopman_matrix} x {len_koopman_matrix}")

edmd_rbf.transform(X_tsc)

# Plot EDMD RBF comparisons
f, ax = plt.subplots(1, 2, sharey=True, figsize=(14, 5))
for _id, df in X_tsc.itertimeseries():
    ax[0].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[0], df)

ax[0].set_xlabel(r"$x_1$", fontsize=14)
ax[0].set_ylabel(r"$x_2$", fontsize=14)
ax[0].axis("equal")
ax[0].grid()

for _id, df in edmd_rbf_values.itertimeseries():
    ax[1].plot(df["x1"].to_numpy(), df["x2"].to_numpy(), 0.1, c="black")
    include_arrow(ax[1], df)

ax[1].set_xlabel(r"$x_1$", fontsize=14)
ax[1].set_ylabel(r"$x_2$", fontsize=14)
ax[1].axis("equal")
ax[1].grid()

# Make out-of-sample prediction (RBF)
X_ic_oos = np.array([[2, 1]])
time_values_oos = np.linspace(0, 7, 400)

ground_truth = system.predict(X_ic_oos, time_values=time_values_oos)
predicted = edmd_rbf.predict(X_ic_oos, time_values=time_values_oos)

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
    label="edmd_rbf",
)

ax.set_xlabel(r"$x_1$", fontsize=14)
ax.set_ylabel(r"$x_2$", fontsize=14)
ax.axis("equal")
ax.grid()
ax.legend()
