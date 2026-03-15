import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- BSM Gamma Logic ---
def bsm_gamma(S, K, T, r, sigma):
    """
    Calculates the BSM Gamma (Convexity).
    Gamma = n'(d1) / (S * sigma * sqrt(T))
    """
    if T <= 0.001: # Avoid division by zero at expiration
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    gamma = nd1 / (S * sigma * np.sqrt(T))
    return gamma

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2

# Grids: Spot Price and Time to Maturity
S_range = np.linspace(70, 130, 80)
T_range = np.linspace(0.05, 2.0, 80) # Start slightly above 0 to avoid Gamma explosion
S_grid, T_grid = np.meshgrid(S_range, T_range)

# Calculate Gamma values for the grid
Z_gamma = np.vectorize(bsm_gamma)(S_grid, K, T_grid, r, sigma)

# --- Plotting ---
plt.style.use('default')
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 1. Gamma Surface
surf = ax.plot_surface(S_grid, T_grid, Z_gamma, cmap=cm.coolwarm,
                       linewidth=0.1, antialiased=True, alpha=0.8, edgecolors='gray')

# 2. Lignes de niveau (Contour Lines) projected on the bottom plane
cset = ax.contour(S_grid, T_grid, Z_gamma, zdir='z', offset=-0.01, cmap=cm.coolwarm, linewidths=2)

# 3. Optional: Contour lines on the surface itself for "Value Zone" effect
cset2 = ax.contour(S_grid, T_grid, Z_gamma, levels=10, cmap=cm.coolwarm, linewidths=0.5, alpha=0.5)

# Styling
ax.set_title(r"Option Gamma ($\Gamma$) - Convexity Zone Analysis", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Spot Price (S)', fontsize=12, labelpad=15)
ax.set_ylabel('Time to Maturity (T)', fontsize=12, labelpad=15)
ax.set_zlabel('Gamma Value', fontsize=12, labelpad=15)

# Ticks configuration
ax.set_xticks([70, 85, 100, 115, 130])
ax.set_xticklabels(['70', '85', 'ATM (100)', '115', '130'])

# Set Z limit to focus on the convexity area (avoiding extreme peaks if T is too low)
ax.set_zlim(-0.01, Z_gamma.max() * 1.1)

# Panes and Grid
ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
ax.grid(True, linestyle=':', alpha=0.4)

# View Angle - Best to see the "mountain" of Gamma around ATM
ax.view_init(elev=30, azim=-135)

# Add Colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1, label='Gamma Intensity')

plt.tight_layout()

print("Graphique de la Zone de Convexité (Gamma) terminé. Affichage...")
plt.show()
