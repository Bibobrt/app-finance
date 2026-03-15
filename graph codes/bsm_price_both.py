import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

# --- BSM Logic ---
def bsm_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return np.maximum(0, S - K) if option_type == 'call' else np.maximum(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# --- Configuration ---
K = 100
r = 0.05
sigma = 0.2
S_range = np.linspace(70, 130, 40)
T_range = np.linspace(0.01, 1.0, 40)
S_grid, T_grid = np.meshgrid(S_range, T_range)

# Calcul des prix
Z_call = np.vectorize(lambda s, t: bsm_price(s, K, t, r, sigma, 'call'))(S_grid, T_grid)
Z_put = np.vectorize(lambda s, t: bsm_price(s, K, t, r, sigma, 'put'))(S_grid, T_grid)

# --- Plotting ---
fig = plt.figure(figsize=(16, 8))

# Subplot 1: Call
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(S_grid, T_grid, Z_call, cmap=cm.viridis, alpha=0.9, antialiased=True)
ax1.set_title('BSM Call Price Surface', fontsize=14)
ax1.set_xlabel('Spot (St)')
ax1.set_ylabel('Time (T)')
ax1.set_zlabel('Premium')
ax1.view_init(elev=25, azim=-135)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# Subplot 2: Put
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(S_grid, T_grid, Z_put, cmap=cm.plasma, alpha=0.9, antialiased=True)
ax2.set_title('BSM Put Price Surface', fontsize=14)
ax2.set_xlabel('Spot (St)')
ax2.set_ylabel('Time (T)')
ax2.set_zlabel('Premium')
ax2.view_init(elev=25, azim=45)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.suptitle(f'Black-Scholes Option Surfaces (Strike K={K}, r={r*100}%, sigma={sigma*100}%)', fontsize=16)
plt.tight_layout()

print("Génération du graphique comparatif terminé. Affichage en cours...")
plt.show()
