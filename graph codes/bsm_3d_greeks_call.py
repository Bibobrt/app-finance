import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

# --- BSM Formulas ---
def d1_d2(S, K, T, r, sigma):
    # Protection against T=0 to avoid division by zero
    T_safe = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    return d1, d2, T_safe

def bsm_call_price(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    # If T is effectively 0, intrinsic value
    intrinsic = np.where(S > K, S - K, 0.0)
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, intrinsic, price)

def bsm_call_delta(S, K, T, r, sigma):
    d1, _, _ = d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    # At T=0, Delta is a step function (1 if ITM, 0 if OTM, 0.5 if ATM)
    delta_expi = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
    return np.where(T <= 1e-8, delta_expi, delta)

def bsm_gamma(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    gamma = nd1 / (S * sigma * np.sqrt(T_safe))
    return np.where(T <= 1e-8, 0.0, gamma)

def bsm_call_theta(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T_safe))
    term2 = r * K * np.exp(-r * T_safe) * norm.cdf(d2)
    # Return Theta normalized per 1 day (divided by 365)
    return np.where(T <= 1e-8, 0.0, (term1 - term2) / 365.0)

def bsm_vega(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    # Return Vega per 1% points of volatility
    return np.where(T <= 1e-8, 0.0, S * nd1 * np.sqrt(T_safe) / 100.0)

def bsm_call_rho(S, K, T, r, sigma):
    _, d2, T_safe = d1_d2(S, K, T, r, sigma)
    # Return Rho per 1% point of interest rate
    return np.where(T <= 1e-8, 0.0, K * T_safe * np.exp(-r * T_safe) * norm.cdf(d2) / 100.0)

# --- Configuration ---
K = 100.0
r = 0.05
sigma = 0.20

# Meshgrid (Spot and Time)
S_arr = np.linspace(60, 140, 60)
T_arr = np.linspace(0.001, 1.0, 60) # Évite exactement 0 pour les graphes 3D
S, T_grid = np.meshgrid(S_arr, T_arr)

# Calculate all surfaces
Price = bsm_call_price(S, K, T_grid, r, sigma)
Delta = bsm_call_delta(S, K, T_grid, r, sigma)
Gamma = bsm_gamma(S, K, T_grid, r, sigma)
Theta = bsm_call_theta(S, K, T_grid, r, sigma)
Vega  = bsm_vega(S, K, T_grid, r, sigma)
Rho   = bsm_call_rho(S, K, T_grid, r, sigma)

# --- 3D Plotting Matrix ---
plt.style.use('default')
fig = plt.figure(figsize=(18, 12))

# Titles and color maps for each metric
metrics = [
    (Price, "Prix (Call)", cm.viridis),
    (Delta, r"Delta ($\Delta$)", cm.coolwarm),
    (Gamma, r"Gamma ($\Gamma$)", cm.magma),
    (Theta, r"Theta 1D ($\Theta$)", cm.inferno),
    (Vega, r"Vega ($\nu$)", cm.plasma),
    (Rho, r"Rho ($\rho$)", cm.cividis)
]

for idx, (Z_data, title, cmap) in enumerate(metrics, start=1):
    ax = fig.add_subplot(2, 3, idx, projection='3d')
    
    # Plot Surface
    surf = ax.plot_surface(T_grid, S, Z_data, cmap=cmap, linewidth=0.2, antialiased=True, edgecolor='black', alpha=0.9)
    
    # Axes Labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Maturité (T)', fontsize=10)
    ax.set_ylabel('Spot (S)', fontsize=10)
    
    # Custom View Angles for better readability
    if "Gamma" in title or "Vega" in title:
        ax.view_init(elev=25, azim=130)  # Show the peak from the front
    elif "Theta" in title:
        ax.view_init(elev=25, azim=45)   # Show the deep ditch
    else:
        ax.view_init(elev=25, azim=230)  # Classic view for Price/Delta/Rho
        
    ax.invert_xaxis() # Pour que T=0 soit au fond ou devant selon l'angle, plus logique temporellement

# Super Title
fig.suptitle(f"Matrice 3D des Sensibilités Black-Scholes (Option Call)\nStrike K={K}, Taux r={r*100}%, Volatilité $\sigma$={sigma*100}%",
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95]) # Leave space for suptitle
plt.subplots_adjust(wspace=0.1, hspace=0.1)

print("Graphique Matrice 3D des Grecques terminé. Affichage...")
plt.show()
