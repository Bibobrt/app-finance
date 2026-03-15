import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- BSM Density Logic ---
def get_d1_d2(S, K, T, r, sigma):
    if T <= 0:
        return 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def normal_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2

# Grids: Spot and Maturity
S_range = np.linspace(50, 150, 60)
T_range = np.linspace(0.01, 2.0, 60)
S_grid, T_grid = np.meshgrid(S_range, T_range)

# Vectors for densities
def calc_n_d1(s, t):
    d1, _ = get_d1_d2(s, K, t, r, sigma)
    return normal_pdf(d1)

def calc_n_d2(s, t):
    _, d2 = get_d1_d2(s, K, t, r, sigma)
    return normal_pdf(d2)

Z_nd1 = np.vectorize(calc_n_d1)(S_grid, T_grid)
Z_nd2 = np.vectorize(calc_n_d2)(S_grid, T_grid)

# --- Plotting ---
plt.style.use('default') # Fond blanc comme demandé
fig = plt.figure(figsize=(16, 10))

# Subplot 1: n'(d1) Surface
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(S_grid, T_grid, Z_nd1, cmap=cm.viridis,
                         linewidth=0, antialiased=True, alpha=0.9)
ax1.set_title(r"Probability Density $n'(d_1)$", fontsize=14, fontweight='bold')
ax1.set_xlabel('Spot Price (S)', labelpad=10)
ax1.set_ylabel('Maturity (T)', labelpad=10)
ax1.set_zlabel('Density Value', labelpad=10)
ax1.view_init(elev=25, azim=-125)

# Subplot 2: n'(d2) Surface (for comparison)
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(S_grid, T_grid, Z_nd2, cmap=cm.magma,
                         linewidth=0, antialiased=True, alpha=0.9)
ax2.set_title(r"Probability Density $n'(d_2)$", fontsize=14, fontweight='bold')
ax2.set_xlabel('Spot Price (S)', labelpad=10)
ax2.set_ylabel('Maturity (T)', labelpad=10)
ax2.set_zlabel('Density Value', labelpad=10)
ax2.view_init(elev=25, azim=-125)

# Common Styling
for ax in [ax1, ax2]:
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.grid(True, linestyle=':', alpha=0.4)

plt.suptitle(f"BSM Probability Densities Analysis\nStrike K={K}, r={r*100}%, sigma={sigma*100}%", 
             fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

print("Génération des surfaces de densité n'(d1) et n'(d2) terminée. Affichage...")
plt.show()
