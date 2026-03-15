import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- BSM Logic ---
def bsm_gamma(S, K, T, r, sigma):
    if T <= 0: return np.zeros_like(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return nd1 / (S * sigma * np.sqrt(T))

def bsm_theta(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return np.zeros_like(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

# --- Configuration ---
K = 100.0
r = 0.05
sigma = 0.20
S_range = np.linspace(70, 130, 200)

# Maturities in days decreasing
maturities_days = np.array([180, 90, 60, 30, 15, 7, 3, 1])

# Color Gradient Setup: Coolwarm (Cool/Blue = Long Maturity, Warm/Red = Short Maturity)
# We map index 0 to max index to get a progression of colors
colors = cm.coolwarm(np.linspace(0.1, 0.9, len(maturities_days)))

# --- Plotting ---
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

for i, days in enumerate(maturities_days):
    T = days / 365.0
    
    # Calculate Greeks
    gamma_vals = bsm_gamma(S_range, K, T, r, sigma)
    # Annualized Theta scaled down to Daily for better reading
    theta_vals = bsm_theta(S_range, K, T, r, sigma, 'call') / 365.0 
    
    label = f'T = {days} jours'
    if days == 1:
        label = f'T = {days} jour (Expi)'
        
    ax1.plot(S_range, gamma_vals, color=colors[i], label=label, linewidth=2.5)
    ax2.plot(S_range, theta_vals, color=colors[i], label=label, linewidth=2.5)

# --- Styling Subplot 1 (Gamma) ---
ax1.set_title(r"Évolution du Profil de Gamma ($\Gamma$) selon la Maturité", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel(r'Gamma ($\Gamma$)', fontsize=12)
ax1.axvline(K, color='black', linestyle=':', alpha=0.5, label='Strike (K)')
ax1.grid(True, linestyle=':', alpha=0.5)
# Show legend on the first subplot
ax1.legend(title="Maturités (Froid=Loin, Chaud=Proche)", fontsize=10, loc='upper right')

# --- Styling Subplot 2 (Theta) ---
ax2.set_title(r"Évolution du Profil de Theta Quotidien ($\Theta_{1j}$) selon la Maturité", fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Spot Price (S)', fontsize=12)
ax2.set_ylabel('Theta Quotidien (en $)', fontsize=12)
ax2.axvline(K, color='black', linestyle=':', alpha=0.5)
ax2.grid(True, linestyle=':', alpha=0.5)

# Add annotations to explain the behavior
ax1.annotate(r"Le Gamma 'explose' ATM" + "\n" + r"et s'écrase ITM/OTM", 
             xy=(102, ax1.get_ylim()[1]*0.8), xytext=(110, ax1.get_ylim()[1]*0.6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=11)

# Theta is negative for a long call
ax2.annotate(r"Le Theta (coût du temps)" + "\n" + r"plonge drastiquement ATM", 
             xy=(98, ax2.get_ylim()[0]*0.9), xytext=(75, ax2.get_ylim()[0]*0.6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=11)

plt.suptitle(f"Impact du Temps (Time Decay) sur la Forme des Sensibilités\nOption Call: Strike K={K}, r={r*100}%, $\sigma$={sigma*100}%", 
             fontsize=16, y=0.98, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

print("Graphique de l'évolution de la forme Gamma/Theta terminé. Affichage...")
plt.show()
