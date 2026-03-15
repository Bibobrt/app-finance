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

def bsm_vega(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return S * nd1 * np.sqrt(T)

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

# --- Simulation Portfolio ---
S0 = 100.0
r = 0.02
sigma = 0.20

# Create a dummy portfolio of options
np.random.seed(42)
num_positions = 500

# Random strikes between 60 and 140
strikes = np.random.uniform(60, 140, num_positions)
# Random maturities between 30 days and 1 year
maturities = np.random.uniform(30/365, 1.0, num_positions)
# Random types (Calls and Puts)
types = np.random.choice(['call', 'put'], num_positions)
# Random quantity (Contracts of 100 multiplier)
quantities = np.random.randint(-50, 50, num_positions) * 100

port_data = []
for i in range(num_positions):
    K = strikes[i]
    T = maturities[i]
    opt = types[i]
    qty = quantities[i]
    
    # Calculate risks per option * quantity
    mv = bsm_price(S0, K, T, r, sigma, opt) * qty
    delta = bsm_delta(S0, K, T, r, sigma, opt) * qty
    vega = bsm_vega(S0, K, T, r, sigma) / 100.0 * qty  # Vega per 1% vol move
    theta = bsm_theta(S0, K, T, r, sigma, opt) / 365.0 * qty # 1 Day Theta
    
    # Assign Strike Bucket
    if K < 80: bucket = '< 80 (Deep ITM/OTM)'
    elif K < 95: bucket = '80-95'
    elif K <= 105: bucket = '95-105 (ATM)'
    elif K <= 120: bucket = '105-120'
    else: bucket = '> 120 (Deep OTM/ITM)'
    
    port_data.append({
        'Bucket': bucket,
        'MV': mv,
        'Delta': delta,
        'Vega': vega,
        'Theta 1D': theta
    })

df_port = pd.DataFrame(port_data)

# --- Risk Aggregation ---
# Group by Strike Bucket and sum the risks
pivot_risk = df_port.groupby('Bucket')[['Delta', 'Theta 1D', 'Vega', 'MV']].sum().T

# Add a Total column
pivot_risk['Total Portfolio'] = pivot_risk.sum(axis=1)

# Formatting numbers for display (rounding to integers for cleaner view in finance)
for col in pivot_risk.columns:
    pivot_risk[col] = pivot_risk[col].apply(lambda x: f"{int(x):,}".replace(',', ' '))

# Sort columns for logical display (Buckets then Total)
ordered_cols = ['< 80 (Deep ITM/OTM)', '80-95', '95-105 (ATM)', '105-120', '> 120 (Deep OTM/ITM)', 'Total Portfolio']
pivot_risk = pivot_risk[ordered_cols]

# Order rows as requested by user
ordered_rows = ['Delta', 'Theta 1D', 'Vega', 'MV']
pivot_risk = pivot_risk.loc[ordered_rows]

# --- Output ---
print("\n" + "="*95)
print(" "*25 + "RISK MATRIX : GREEKS BY STRIKE BUCKETS")
print("="*95)
print(f"Sous-jacent : S0 = {S0} | Volatilité = {sigma*100}% | Taux = {r*100}%")
print("Lecture :")
print("  - Delta    : Exposition directionnelle (en équivalent action)")
print("  - Theta 1D : P&L estimé si le temps passe d'un jour (en $)")
print("  - Vega     : P&L estimé pour une hausse de 1 point de volatilité (+1%)")
print("  - MV       : Market Value (Prime totale des options)")
print("-"*95 + "\n")

print(pivot_risk.to_string())

print("\n" + "="*95)
print("Note: Ce tableau simule un book de 500 positions options générées aléatoirement.")
print("="*95 + "\n")
