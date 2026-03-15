import numpy as np
import pandas as pd
from scipy.stats import norm

# --- BSM Logic ---
def bsm_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return np.maximum(0, S - K) if option_type == 'call' else np.maximum(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bsm_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1.0

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
K = 100.0
T0 = 30 / 365.0  # 30 days to maturity
r = 0.0          # Simplified to focus purely on Gamma vs Theta
sigma = 0.20
dt = 1 / 365.0   # 1 day step

# Simulated daily spot moves (dS)
# Incorporating both small moves (Theta wins) and big moves (Gamma wins)
spot_moves = [0.0, 1.2, -0.4, -2.5, 0.5, 3.0, -0.2, -1.5, 1.0, 0.0]

results = []
S = S0
T = T0

# Straddle Gamma is the sum of Call Gamma and Put Gamma (which are equal in BSM)
# Straddle Theta is the sum of Call Theta and Put Theta

for day in range(len(spot_moves)):
    move = spot_moves[day]
    S_new = S + move
    
    # 1. Calculate Greeks at the START of the day
    gamma_call = bsm_gamma(S, K, T, r, sigma)
    straddle_gamma = gamma_call * 2
    
    theta_call = bsm_theta(S, K, T, r, sigma, 'call')
    theta_put = bsm_theta(S, K, T, r, sigma, 'put')
    straddle_theta = theta_call + theta_put
    
    # "Daily" Theta (Theta in BSM formula is annualized, so we multiply by dt)
    daily_theta = straddle_theta * dt
    
    # 2. Gamma / Theta Taylor Approximations
    gamma_pnl_approx = 0.5 * straddle_gamma * (move ** 2)
    net_approx_pnl = gamma_pnl_approx + daily_theta
    
    # 3. Exact BSM Hedged P&L Calculation (For comparison)
    V_old = bsm_price(S, K, T, r, sigma, 'call') + bsm_price(S, K, T, r, sigma, 'put')
    delta_old = bsm_delta(S, K, T, r, sigma, 'call') + bsm_delta(S, K, T, r, sigma, 'put')
    
    T_new = T - dt
    V_new = bsm_price(S_new, K, T_new, r, sigma, 'call') + bsm_price(S_new, K, T_new, r, sigma, 'put')
    
    # P&L of the Option + Hedging portfolio
    exact_pnl = (V_new - V_old) - delta_old * move
    
    # 4. Store results
    results.append({
        'Jour': day + 1,
        'Spot Final': round(S_new, 2),
        'Move (dS)': round(move, 2),
        'Gamma P&L': round(gamma_pnl_approx, 4),
        'Theta Decay': round(daily_theta, 4),
        'Net P&L (Approx)': round(net_approx_pnl, 4),
        'Net P&L (Exact)': round(exact_pnl, 4)
    })
    
    # Prepare for next day
    S = S_new
    T = T_new

# Format as DataFrame
df = pd.DataFrame(results)
df['Cumul P&L'] = df['Net P&L (Exact)'].cumsum().round(4)

# Output
print("\n" + "="*85)
print(" "*20 + "EXEMPLE CHIFFRÉ : GAMMA SCALPING (STRADDLE)")
print("="*85)
print(f"Position Initiale : Long Straddle ATM (Call + Put)")
print(f"Paramètres        : S0={S0}, K={K}, Maturité={int(T0*365)} jours, Vol={sigma*100}%, R={r*100}%")
print(f"Hedge             : Le Delta est neutralisé à la fin de chaque journée.")
print("Règle d'or        : Gamma gagne avec les gros mouvements (dS²), Theta perd tous les jours (dt).")
print("-"*85 + "\n")

print(df.to_string(index=False))

print("\n" + "-"*85)
print("OBSERVATIONS :")
print("1. Jours 4 et 6 : Les gros 'Move (dS)' génèrent un Gamma P&L énorme qui écrase le Theta Decay.")
print("2. Jours 1, 3, 7 et 10 : Les petits mouvements ne suffisent pas, le Theta Decay fait perdre de l'argent.")
print("3. Net P&L (Approx) correspond très fidèlement au P&L exact via Black-Scholes.")
print("="*85 + "\n")
