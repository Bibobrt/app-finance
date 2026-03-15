import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- BSM Logic ---
def bsm_gamma(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return nd1 / (S * sigma * np.sqrt(T))

def bsm_theta(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    
    term1 = - (S * nd1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

# --- Configuration ---
S0 = 100.0
K = 100.0  # ATM straddle focus
r = 0.0    # Put at 0 to focus strictly on Volatility Theta
sigma = 0.20
dt = 1 / 365.0 # 1 Day time step

# X-axis: Spot moves (dS)
moves = np.linspace(-3, 3, 200) # De -3$ à +3$ en une journée

# Different Maturities to compare
# (Days, Label, Color)
maturities = [
    (90, '90 Jours (3 mois)', '#1a9850'),    # Green (Flat gamma, small theta)
    (30, '30 Jours (1 mois)', '#fee08b'),    # Yellow
    (7,  '7 Jours (1 semaine)', '#fdae61'),  # Orange
    (1,  '1 Jour (Veille de l\'expi)', '#d73027') # Red (Spike gamma, huge theta)
]

# --- Plotting ---
plt.style.use('default')
plt.figure(figsize=(12, 8))

for days, label, color in maturities:
    T = days / 365.0
    
    # Calculate Greeks for a Straddle (Call + Put)
    gamma_straddle = bsm_gamma(S0, K, T, r, sigma) * 2
    theta_straddle = bsm_theta(S0, K, T, r, sigma, 'call') + bsm_theta(S0, K, T, r, sigma, 'put')
    
    # P&L Approximation: dPi = Theta*dt + 0.5*Gamma*dS^2
    daily_theta = theta_straddle * dt
    gamma_pnl = 0.5 * gamma_straddle * (moves**2)
    
    total_pnl = daily_theta + gamma_pnl
    
    # Trace de la parabole
    plt.plot(moves, total_pnl, label=label, color=color, linewidth=2.5)
    
    # Mark the Theta at dS=0 with a dot
    plt.scatter([0], [daily_theta], color=color, s=50, zorder=5)

# Styling
plt.title(f'Effet du Temps sur le P&L Quotidien (Gamma vs Theta)\nStraddle ATM ($S_0={S0}, K={K}, \sigma={sigma*100}\%$)', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Mouvement Quotidien du Spot ($dS$)', fontsize=12)
plt.ylabel(r'P&L Journalier ($d\Pi \approx \Theta dt + \frac{1}{2}\Gamma dS^2$)', fontsize=12)

# Reference lines
plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Spot Inchangé ($dS=0$)')
plt.axhline(0, color='black', linewidth=1, label='Seuil de Rentabilité (Breakeven)')

# Annotations
plt.annotate("Theta décroît (coût augmente)\nà l'approche de la maturité", 
             xy=(0, -0.2), xytext=(1.5, -0.15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.annotate("Gamma explose\n(Parabole plus serrée)", 
             xy=(-2, 0.4), xytext=(-2.8, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=10)

plt.grid(True, linestyle=':', alpha=0.4)
plt.legend(title="Maturités Restantes", fontsize=10, loc='upper right')

# Add Y limits to focus on the action
plt.ylim(-0.25, 1.0)

plt.tight_layout()

print("Graphique Effet du Temps (Gamma/Theta P&L) terminé. Affichage...")
plt.show()
