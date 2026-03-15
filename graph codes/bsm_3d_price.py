import numpy as np
import plotly.graph_objects as go
import os
from scipy.stats import norm

# --- BSM Pricing Logic ---
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
S_range = np.linspace(50, 150, 50)
T_range = np.linspace(0.01, 2.0, 50)

S_grid, T_grid = np.meshgrid(S_range, T_range)
Z_price = np.array([bsm_price(s, K, t, r, sigma) for s, t in zip(np.ravel(S_grid), np.ravel(T_grid))])
Z_price = Z_price.reshape(S_grid.shape)

# --- Plotting ---
fig = go.Figure(data=[go.Surface(z=Z_price, x=S_grid, y=T_grid, colorscale='Viridis')])

fig.update_layout(
    title='Black-Scholes Option Price Surface (Call)',
    scene=dict(
        xaxis_title='Spot Price (S)',
        yaxis_title='Time to Maturity (T)',
        zaxis_title='Option Price'
    ),
    template='plotly_dark',
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Save ---
output_dir = '/Users/thibaultberton/Desktop/pyhton/graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'bsm_price_3d.html')
fig.write_html(output_path)
print(f"Graph saved to {output_path}")
