import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ============================
# Define data
# ============================

plane_L = {
    "points": [
        (93.0, 0.24, 40.7, 0.525302),
        (109.15, 0.24, 48.1, 0.843377)
    ]
}

plane_h = {
    "points": [
        (105.25, 0.4, 10, 0.416133),
        (105.25, 0.22, 59.8, 0.680405)
    ]
}

plane_w = {
    "points": [
        (101.35, 0.24, 45, 1.06032),
        (102.9, 0.24, 45, 0.54639)
    ]
}

reference_point = (105.25, 0.24, 45.0, -0.254007)
other_points = [
    (104, 0.24, 45.0, 0.17171),
    (103.3, 0.24, 58.5, 8.59333),
    (107.8, 0.24, 36.7, -9.49018),
    (103.1, 0.24, 59.7, 9.18728)
]

# ============================
# Combine all points
# ============================

all_points = (
    plane_L["points"]
    + plane_h["points"]
    + plane_w["points"]
    + [reference_point]
    + other_points
)

L_vals = np.array([p[0] for p in all_points])
h_vals = np.array([p[1] for p in all_points])
w_vals = np.array([p[2] for p in all_points])
disp_vals = np.array([p[3] for p in all_points])

# ============================
# Fit plane z = aL + bh + cw + d
# ============================

A = np.c_[L_vals, h_vals, w_vals, np.ones_like(L_vals)]
coeffs, _, _, _ = np.linalg.lstsq(A, disp_vals, rcond=None)
a, b, c, d = coeffs
print(f"Fitted plane: z = {a:.5f}*L + {b:.5f}*h + {c:.5f}*w + {d:.5f}")

# ============================
# 3D visualization using colored prediction cloud
# ============================

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot original points
sc1 = ax.scatter(L_vals, h_vals, w_vals, c=disp_vals, cmap='plasma', s=80, edgecolors='black', label='Measured points')

# Create dense 3D grid of predicted values
L_lin = np.linspace(min(L_vals), max(L_vals), 20)
h_lin = np.linspace(min(h_vals), max(h_vals), 20)
w_lin = np.linspace(min(w_vals), max(w_vals), 20)
L_grid, h_grid, w_grid = np.meshgrid(L_lin, h_lin, w_lin)

# Flatten and compute predictions
Lf = L_grid.ravel()
hf = h_grid.ravel()
wf = w_grid.ravel()
disp_pred = a * Lf + b * hf + c * wf + d

# Plot predicted displacement field as 3D colored cloud
sc2 = ax.scatter(Lf, hf, wf, c=disp_pred, cmap='coolwarm', alpha=0.3, s=10, label='Fitted surface')

# Labels and colorbar
ax.set_xlabel("L (mm)")
ax.set_ylabel("h (mm)")
ax.set_zlabel("w (mm)")
ax.set_title("3D Displacement Prediction from Fitted Model")
cbar = fig.colorbar(sc2, ax=ax, shrink=0.6)
cbar.set_label("Predicted Displacement (mm)")

plt.legend()
plt.tight_layout()
plt.show()
