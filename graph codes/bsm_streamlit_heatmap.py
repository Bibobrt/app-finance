import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="BSM Heatmap Studio", page_icon="📊", layout="wide")

# --- Header with LinkedIn ---
col_title, col_li = st.columns([8, 1])
with col_title:
    st.title("📊 Black-Scholes Heatmap Studio")
    st.markdown("Choisissez les axes, les métriques, et les paramètres fixes pour générer vos 4 heatmaps.")
with col_li:
    st.markdown("""
<a href="https://www.linkedin.com/in/thibault-berton/" target="_blank" style="display:flex; align-items:center; gap:8px; text-decoration:none; color:inherit;">
    <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="36">
    <span style="font-size:13px; font-weight:600;">Thibault Berton<br>ESCP</span>
</a>
""", unsafe_allow_html=True)
st.markdown("---")

# --- BSM Formulas (Merton, with continuous dividend yield q) ---
def d1_d2(S, K, T, r, q, sigma):
    T_safe = np.maximum(T, 1e-8)
    s_safe = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * s_safe**2) * T_safe) / (s_safe * np.sqrt(T_safe))
    d2 = d1 - s_safe * np.sqrt(T_safe)
    return d1, d2, T_safe, s_safe

def call_price(S, K, T, r, q, sigma):
    d1, d2, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    intrinsic = np.where(S > K, S - K, 0.0)
    p = S * np.exp(-q * T_s) * norm.cdf(d1) - K * np.exp(-r * T_s) * norm.cdf(d2)
    return np.where(T <= 1e-8, intrinsic, p)

def put_price(S, K, T, r, q, sigma):
    d1, d2, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    intrinsic = np.where(K > S, K - S, 0.0)
    p = K * np.exp(-r * T_s) * norm.cdf(-d2) - S * np.exp(-q * T_s) * norm.cdf(-d1)
    return np.where(T <= 1e-8, intrinsic, p)

def call_delta(S, K, T, r, q, sigma):
    d1, _, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, np.where(S > K, 1.0, 0.0), np.exp(-q * T_s) * norm.cdf(d1))

def put_delta(S, K, T, r, q, sigma):
    d1, _, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, np.where(K > S, -1.0, 0.0), np.exp(-q * T_s) * (norm.cdf(d1) - 1.0))

def gamma(S, K, T, r, q, sigma):
    d1, _, T_s, s_s = d1_d2(S, K, T, r, q, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, np.exp(-q * T_s) * nd1 / (S * s_s * np.sqrt(T_s)))

def vega(S, K, T, r, q, sigma):
    d1, _, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.where(T <= 1e-8, 0.0, S * np.exp(-q * T_s) * nd1 * np.sqrt(T_s) / 100.0)

def call_theta(S, K, T, r, q, sigma):
    d1, d2, T_s, s_s = d1_d2(S, K, T, r, q, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    t1 = -(S * np.exp(-q * T_s) * nd1 * s_s) / (2 * np.sqrt(T_s))
    t2 = r * K * np.exp(-r * T_s) * norm.cdf(d2)
    t3 = q * S * np.exp(-q * T_s) * norm.cdf(d1)
    return np.where(T <= 1e-8, 0.0, (t1 - t2 + t3) / 365.0)

def put_theta(S, K, T, r, q, sigma):
    d1, d2, T_s, s_s = d1_d2(S, K, T, r, q, sigma)
    nd1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)
    t1 = -(S * np.exp(-q * T_s) * nd1 * s_s) / (2 * np.sqrt(T_s))
    t2 = r * K * np.exp(-r * T_s) * norm.cdf(-d2)
    t3 = q * S * np.exp(-q * T_s) * norm.cdf(-d1)
    return np.where(T <= 1e-8, 0.0, (t1 + t2 - t3) / 365.0)

def call_rho(S, K, T, r, q, sigma):
    _, d2, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, 0.0, K * T_s * np.exp(-r * T_s) * norm.cdf(d2) / 100.0)

def put_rho(S, K, T, r, q, sigma):
    _, d2, T_s, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, 0.0, -K * T_s * np.exp(-r * T_s) * norm.cdf(-d2) / 100.0)

# --- Helper for probabilities ---
def call_prob_itm(S, K, T, r, q, sigma):
    """Risk-neutral probability that Call finishes ITM = N(d2)"""
    _, d2, _, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, np.where(S >= K, 1.0, 0.0), norm.cdf(d2))

def put_prob_itm(S, K, T, r, q, sigma):
    """Risk-neutral probability that Put finishes ITM = N(-d2)"""
    _, d2, _, _ = d1_d2(S, K, T, r, q, sigma)
    return np.where(T <= 1e-8, np.where(K >= S, 1.0, 0.0), norm.cdf(-d2))

# --- Available Metrics ---
METRICS = {
    "Prime Call (% Spot)":      (lambda S, K, T, r, q, sg: call_price(S, K, T, r, q, sg) / S, ":.0%"),
    "Prime Put (% Spot)":       (lambda S, K, T, r, q, sg: put_price(S, K, T, r, q, sg) / S, ":.0%"),
    "Proba ITM Call [N(d2)]": (call_prob_itm, ":.1%"),
    "Proba ITM Put [N(-d2)]": (put_prob_itm, ":.1%"),
    "Delta (Call)":             (call_delta,  ":.3f"),
    "Delta (Put)":              (put_delta,   ":.3f"),
    "Gamma":                    (gamma,       ":.5f"),
    "Vega":                     (vega,        ":.4f"),
    "Theta Call (1J)":          (call_theta,  ":.4f"),
    "Theta Put (1J)":           (put_theta,   ":.4f"),
    "Rho (Call)":               (call_rho,    ":.4f"),
    "Rho (Put)":                (put_rho,     ":.4f"),
}

# --- Parameter Configs ---
PARAMS_META = {
    "Moneyness (S/K)":    {"default": 1.0,    "min": 0.5,   "max": 2.0,   "step": 0.01,  "range": (0.70, 1.30, 13), "is_moneyness": True},
    "Spot (S)":           {"default": 100.0,  "min": 50.0,  "max": 200.0, "step": 1.0,   "range": (70.0, 130.0, 13), "is_moneyness": False},
    "Strike (K)":         {"default": 100.0,  "min": 50.0,  "max": 200.0, "step": 1.0,   "range": (80.0, 120.0, 11), "is_moneyness": False},
    "Maturité (jours)":   {"default": 30.0,   "min": 1.0,   "max": 365.0, "step": 1.0,   "range": (1.0,  180.0, 10), "is_moneyness": False},
    "Volatilité (σ)":     {"default": 0.20,   "min": 0.01,  "max": 2.0,   "step": 0.01,  "range": (0.05, 0.80, 16), "is_moneyness": False},
    "Taux sans risque (r)": {"default": 0.05, "min": -0.02, "max": 0.20,  "step": 0.005, "range": (0.0,  0.12,  7),  "is_moneyness": False},
    "Dividendes (q)":     {"default": 0.0,    "min": 0.0,   "max": 0.20,  "step": 0.005, "range": (0.0,  0.08,  5),  "is_moneyness": False},
}
PARAM_NAMES = list(PARAMS_META.keys())

COLORMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Blues", "Reds", "Greens", "Purples",
    "RdBu", "RdYlGn", "Spectral",
    "hot", "cool", "turbo", "rainbow",
]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Configuration Globale")

# --- LinkedIn at the TOP ---
st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/thibault-berton/" target="_blank" style="display:flex; align-items:center; gap:10px; text-decoration:none; color:inherit;">
    <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="32">
    <span style="font-size:13px;"><b>Thibault Berton</b><br>ESCP</span>
</a>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("📐 Axes Variables")
x_param = st.sidebar.selectbox("Axe X (colonne)", PARAM_NAMES, index=1)   # Strike
y_param = st.sidebar.selectbox("Axe Y (ligne)",   PARAM_NAMES, index=3)   # Vol

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Paramètres Fixes")

# Moneyness is AXIS-ONLY: never show it as a fixed slider.
# When Moneyness IS on an axis, both Spot AND Strike are hidden:
# → Spot is derived (S = moneyness × 100), and Strike is normalized to 100
#   because Price(% Spot) is scale-invariant when S/K ratio is fixed.
AXIS_ONLY_PARAMS    = {"Moneyness (S/K)"}
MONEYNESS_ACTIVE    = "Moneyness (S/K)" in [x_param, y_param]
HIDE_WHEN_MONEYNESS = {"Spot (S)", "Strike (K)"}

if MONEYNESS_ACTIVE:
    st.sidebar.info("💡 Mode Moneyness actif — Spot = Moneyness \u00d7 100, Strike normalisé \u00e0 100")

fixed_params = {}
for p_name, meta in PARAMS_META.items():
    # Skip params that are on an axis
    if p_name in [x_param, y_param]:
        continue
    # Skip axis-only params (Moneyness must never be a fixed slider)
    if p_name in AXIS_ONLY_PARAMS:
        continue
    # Hide Spot AND Strike when Moneyness is active
    if MONEYNESS_ACTIVE and p_name in HIDE_WHEN_MONEYNESS:
        continue
    fixed_params[p_name] = st.sidebar.slider(
        p_name,
        min_value=float(meta["min"]),
        max_value=float(meta["max"]),
        value=float(meta["default"]),
        step=float(meta["step"]),
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Apparence")
n_cols       = st.sidebar.selectbox("Heatmap — résolution (nb colonnes)", [7, 9, 11, 13, 15, 17], index=2)
n_rows       = st.sidebar.selectbox("Heatmap — résolution (nb lignes)",   [7, 9, 11, 13, 15, 17], index=2)
cell_height  = st.sidebar.slider("Hauteur d'une heatmap (px)", 200, 800, 350, step=50)
top_cmap     = st.sidebar.selectbox("Couleur heatmaps HAUT (Prix)",  COLORMAPS, index=0)
bot_cmap     = st.sidebar.selectbox("Couleur heatmaps BAS (Choix)", COLORMAPS, index=3)
show_text    = st.sidebar.checkbox("Afficher les valeurs dans les cellules", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 Format des annotations")
fmt_mode_top = st.sidebar.radio("Format — Haut (Prix)",     ["% (pourcentage)", "Décimal"], index=0, horizontal=True)
fmt_mode_bot = st.sidebar.radio("Format — Bas (Métriques)", ["% (pourcentage)", "Décimal"], index=1, horizontal=True)
decimals_top = st.sidebar.number_input("Décimales — Haut", min_value=0, max_value=6, value=1, step=1)
decimals_bot = st.sidebar.number_input("Décimales — Bas",  min_value=0, max_value=6, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Métriques heatmaps BAS")
metric_bl_name = st.sidebar.selectbox("Métrique — Bas Gauche", list(METRICS.keys()), index=2)
metric_br_name = st.sidebar.selectbox("Métrique — Bas Droite", list(METRICS.keys()), index=4)

# =========================================================
# ERROR CHECK
# =========================================================
if x_param == y_param:
    st.error("⚠️ Veuillez choisir deux paramètres **différents** pour les axes X et Y.")
    st.stop()

# =========================================================
# DATA COMPUTATION
# =========================================================
meta_x = PARAMS_META[x_param]
meta_y = PARAMS_META[y_param]
x_arr = np.linspace(meta_x["range"][0], meta_x["range"][1], n_cols)
y_arr = np.linspace(meta_y["range"][0], meta_y["range"][1], n_rows)

def fmt_arr(arr, p_name):
    if "σ" in p_name or p_name in ["Taux sans risque (r)", "Dividendes (q)"]:
        return [f"{v:.0%}" for v in arr]
    if "Moneyness" in p_name:
        return [f"{v:.2f}x" for v in arr]
    return [f"{v:.0f}" for v in arr]

x_labels = fmt_arr(x_arr, x_param)
y_labels = fmt_arr(y_arr, y_param)

def resolve_bsm(x_val, y_val):
    p = {**fixed_params, x_param: x_val, y_param: y_val}
    if "Moneyness" in x_param or "Moneyness" in y_param:
        # Both S and K are normalized: K=100 always, S = moneyness * 100
        # This makes the heatmap a pure display of BSM sensitivity to  
        # the S/K ratio — scale-invariant and Strike-independent.
        K = 100.0
        mon = p.get("Moneyness (S/K)", 1.0)
        S = mon * K
    else:
        S = p.get("Spot (S)", 100.0)
        K = p.get("Strike (K)", 100.0)
    T  = p.get("Maturité (jours)", 30.0) / 365.0
    r  = p.get("Taux sans risque (r)", 0.05)
    q  = p.get("Dividendes (q)", 0.0)
    sg = p.get("Volatilité (σ)", 0.20)
    return S, K, T, r, q, sg

def compute_matrix(metric_func):
    Z = np.zeros((len(y_arr), len(x_arr)))
    for i, y_val in enumerate(y_arr):
        for j, x_val in enumerate(x_arr):
            S, K, T, r, q, sg = resolve_bsm(x_val, y_val)
            Z[i, j] = metric_func(S, K, T, r, q, sg)
    return Z

# Four matrices
Z_call  = compute_matrix(METRICS["Prime Call (% Spot)"][0])
Z_put   = compute_matrix(METRICS["Prime Put (% Spot)"][0])
Z_bl    = compute_matrix(METRICS[metric_bl_name][0])
Z_br    = compute_matrix(METRICS[metric_br_name][0])

def build_fmt_spec(mode, decimals):
    """Build a Python format() spec from user choices."""
    if "pourcentage" in mode:
        return f".{decimals}%"
    else:
        return f".{decimals}f"

def make_annotation(Z, spec):
    rows = []
    for i in range(Z.shape[0]):
        row = []
        for j in range(Z.shape[1]):
            row.append(format(Z[i, j], spec))
        rows.append(row)
    return rows

spec_top = build_fmt_spec(fmt_mode_top, int(decimals_top))
spec_bot = build_fmt_spec(fmt_mode_bot, int(decimals_bot))

ann_call = make_annotation(Z_call, spec_top)
ann_put  = make_annotation(Z_put,  spec_top)
ann_bl   = make_annotation(Z_bl,   spec_bot)
ann_br   = make_annotation(Z_br,   spec_bot)

# =========================================================
# PLOTTING
# =========================================================
def make_heatmap(Z, ann, title, cmap, show_text, height):
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=x_labels,
        y=y_labels,
        text=ann if show_text else None,
        texttemplate="%{text}" if show_text else None,
        textfont={"size": 10, "color": "white"},
        colorscale=cmap,
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16)),
        xaxis_title=x_param, yaxis_title=y_param,
        yaxis=dict(autorange='reversed'),
        height=height,
        margin=dict(t=60, b=60, l=60, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# =========================================================
# HELP SECTION — Collapsible expander (above heatmaps)
# =========================================================
with st.expander("📖 À quoi sert cet outil ? — Guide d'utilisation", expanded=False):
    st.markdown("""
### À quoi sert cet outil ?

Cette application permet de visualiser l'impact des paramètres du modèle de **Black-Scholes (Merton)** sur le prix des options et sur leurs sensibilités (les *«Greeks»*).

Plutôt que de regarder une seule valeur pour une combinaison de paramètres donnée, l'outil permet d'explorer une large zone de scénarios grâce à des **heatmaps**. Deux paramètres peuvent varier simultanément sur les axes X et Y, tandis que les autres restent fixés.

L'objectif est de donner une **intuition visuelle** du pricing des options et de leurs sensibilités.

---

### Ce que montrent les graphiques

L'application affiche **4 heatmaps** :

| Position | Contenu |
|---|---|
| 🔵 Haut Gauche | **Prime Call** (% du Spot) — fixe |
| 🔵 Haut Droit | **Prime Put** (% du Spot) — fixe |
| 🟠 Bas Gauche | Métrique au **choix** (sidebar) |
| 🟠 Bas Droit | Métrique au **choix** (sidebar) |

Les métriques disponibles incluent :

- **Proba ITM Call `N(d2)`** : probabilité risque-neutre que le Call finisse In-The-Money à expiration.
- **Proba ITM Put `N(−d2)`** : idem pour le Put.
- **Prix Call / Put** : prime de l'option en % du Spot.
- **Delta** : sensibilité du prix de l'option aux variations du sous-jacent.
- **Gamma** : variation du delta lorsque le prix du sous-jacent change.
- **Vega** : sensibilité du prix à la volatilité implicite.
- **Theta** : perte de valeur liée au passage du temps (par jour).
- **Rho** : sensibilité du prix aux taux d'intérêt.

---

### Comment utiliser l'outil

1. **Choisir les axes variables**
   Dans la sidebar, sélectionnez deux paramètres qui varieront dans les heatmaps. Conseil : **Moneyness (S/K)** en X et **Volatilité** en Y.

2. **Définir les paramètres fixes**
   Tous les autres paramètres (Strike, Maturité, Taux, Dividendes) sont ajustables avec les sliders. Si Moneyness est actif, le Spot est calculé automatiquement.

3. **Observer les heatmaps**
   Les couleurs indiquent les zones de valeurs hautes ou basses. Les chiffres dans les cellules donnent la valeur exacte.

4. **Choisir le format**
   Affichez les valeurs en `%` ou en décimal, et choisissez le nombre de décimales.

5. **Comparer des métriques**
   Sélectionnez deux métriques différentes pour les analyser côte-à-côte.

---

### Note sur le modèle

```
d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T
Call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

Les probabilités ITM sont « risque-neutres » (mesure Q), pas des probabilités réelles (monde P).
    """)

st.markdown("---")
fixed_info = "  |  ".join([f"**{k.split('(')[0].strip()}** = {v}" for k, v in fixed_params.items()])
st.markdown(f"🔒 Paramètres fixes : {fixed_info}")
st.markdown(f"**Axes :** X = *{x_param}* | Y = *{y_param}*")

col_tl, col_tr = st.columns(2)
with col_tl:
    st.plotly_chart(make_heatmap(Z_call, ann_call, "Prime Call (% Spot)", top_cmap, show_text, cell_height), width='stretch')
with col_tr:
    st.plotly_chart(make_heatmap(Z_put,  ann_put,  "Prime Put (% Spot)",  top_cmap, show_text, cell_height), width='stretch')

col_bl, col_br = st.columns(2)
with col_bl:
    st.plotly_chart(make_heatmap(Z_bl, ann_bl, metric_bl_name, bot_cmap, show_text, cell_height), width='stretch')
with col_br:
    st.plotly_chart(make_heatmap(Z_br, ann_br, metric_br_name, bot_cmap, show_text, cell_height), width='stretch')


