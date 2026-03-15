import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm

# --- BSM Formulas ---
def d1_d2(S, K, T, r, sigma):
    T_safe = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)
    return d1, d2, T_safe

def bsm_call_price(S, K, T, r, sigma):
    d1, d2, T_safe = d1_d2(S, K, T, r, sigma)
    intrinsic = np.where(S > K, S - K, 0.0)
    price = S * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    return np.where(T <= 1e-8, intrinsic, price)

def bsm_call_delta(S, K, T, r, sigma):
    d1, _, _ = d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
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
    return np.where(T <= 1e-8, 0.0, (term1 - term2) / 365.0)

def bsm_vega(S, K, T, r, sigma):
    d1, _, T_safe = d1_d2(S, K, T, r, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, S * nd1 * np.sqrt(T_safe) / 100.0)

def bsm_call_rho(S, K, T, r, sigma):
    _, d2, T_safe = d1_d2(S, K, T, r, sigma)
    return np.where(T <= 1e-8, 0.0, K * T_safe * np.exp(-r * T_safe) * norm.cdf(d2) / 100.0)

# --- Configuration ---
K = 100.0
r = 0.05
sigma = 0.20
S_range = np.linspace(60, 140, 200)

maturities_days = np.array([365, 180, 90, 30, 7, 1])
colors = cm.coolwarm(np.linspace(0.1, 0.9, len(maturities_days)))

# --- 2D Plotting Matrix ---
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

titles_and_funcs = [
    ("Prix (Call)", bsm_call_price),
    (r"Delta ($\Delta$)", bsm_call_delta),
    (r"Gamma ($\Gamma$)", bsm_gamma),
    (r"Theta 1D ($\Theta$)", bsm_call_theta),
    (r"Vega ($\nu$)", bsm_vega),
    (r"Rho ($\rho$)", bsm_call_rho)
]

for idx, (title, func) in enumerate(titles_and_funcs):
    ax = axes[idx]
    
    for i, days in enumerate(maturities_days):
        T = days / 365.0
        Z_data = func(S_range, K, T, r, sigma)
        
        label = f'{days}j' if days < 365 else '1 an'
        ax.plot(S_range, Z_data, color=colors[i], label=label, linewidth=2)
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Spot (S)', fontsize=10)
    ax.axvline(K, color='black', linestyle=':', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if idx == 0: # Add legend only to the first subplot to save space
        ax.legend(title="Maturité", fontsize=9, loc='upper left')

# Super Title
fig.suptitle(r"Matrice 2D des Sensibilités Black-Scholes (Option CALL)" + f"\nStrike K={K}, Taux r={r*100}%, Volatilité $\\sigma$={sigma*100}%",
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.94])
print("Graphique Matrice 2D des Grecques (CALL) terminé. Affichage...")
plt.show()
