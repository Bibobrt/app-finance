import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_call_price(S, K, T, r, sigma):
    if T <= 0: return np.maximum(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bsm_call_delta(S, K, T, r, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bsm_call_gamma(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return nd1 / (S * sigma * np.sqrt(T))

# --- Configuration ---
S0 = 100
K = 100
T = 0.5
r = 0.02
sigma = 0.2

# Range of spot moves (from -20% to +20%)
moves = np.linspace(-30, 30, 200) # dS
S_new = S0 + moves

# 1. Actual P&L
price0 = bsm_call_price(S0, K, T, r, sigma)
pnl_actual = np.array([bsm_call_price(s, K, T, r, sigma) - price0 for s in S_new])

# 2. Delta Component (dPrice ~ Delta * dS)
delta0 = bsm_call_delta(S0, K, T, r, sigma)
pnl_delta = delta0 * moves

# 3. Gamma Component (dPrice ~ 0.5 * Gamma * dS^2)
gamma0 = bsm_call_gamma(S0, K, T, r, sigma)
pnl_gamma = 0.5 * gamma0 * (moves**2)

# 4. Total Approximation (Delta + Gamma)
pnl_approx = pnl_delta + pnl_gamma

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

plt.plot(moves, pnl_actual, label='Actual P&L (Full BSM)', color='black', linewidth=3, zorder=5)
plt.plot(moves, pnl_delta, label=r'Delta Component ($\Delta \cdot dS$)', color='blue', linestyle='--', alpha=0.7)
plt.plot(moves, pnl_gamma, label=r'Gamma Component ($\frac{1}{2} \Gamma \cdot dS^2$)', color='orange', linestyle='--', alpha=0.7)
plt.plot(moves, pnl_approx, label='Delta + Gamma Approx', color='red', linestyle=':', linewidth=2)

# Styling
plt.title(f'Instantaneous P&L Attribution: Delta vs Gamma\nATM Call (S={S0}, K={K}, T={T}Y, $\sigma$={sigma*100}%)', 
          fontsize=14, pad=20)
plt.xlabel('Spot Price Change ($dS$)', fontsize=12)
plt.ylabel('P&L ($d\Pi$)', fontsize=12)

# Horizontal/Vertical reference lines
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)

# Annotations
plt.annotate('Convexity (Gamma) always positive for Long Option', xy=(20, 5), xytext=(5, 15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()

print("Graphique P&L Attribution (Delta/Gamma) terminé. Affichage...")
plt.show()
