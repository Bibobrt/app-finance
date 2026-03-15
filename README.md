# Black-Scholes Heatmap Studio 📊

Une application interactive complète pour visualiser et analyser le modèle de Black-Scholes-Merton à travers des heatmaps dynamiques.

## 🚀 Fonctionnalités

- **4 Heatmaps en simultané** : Visualisez les primes Call/Put et deux autres métriques de votre choix (Grecques, Probabilités ITM).
- **Flexibilité totale des axes** : Choisissez n'importe quel paramètre (Moneyness, Spot, Strike, Volatilité, Temps, Taux, Dividendes) pour les axes X et Y.
- **Support des dividendes** : Implémentation du modèle de Black-Scholes-Merton incluant le taux de dividende continu *q*.
- **Probabilités ITM** : Visualisation des probabilités de finir dans la monnaie ($N(d_2)$).
- **Interface Premium** : Design sombre, palettes de couleurs personnalisables, et interactivité fluide via Streamlit et Plotly.

## 🛠 Installation

1. Assurez-vous d'avoir Python installé.
2. Installez les dépendances nécessaires :

```bash
pip install streamlit plotly scipy numpy
```

## 🏃 Comment lancer l'application

Lancez la commande suivante à la racine du projet :

```bash
streamlit run "graph codes/bsm_streamlit_heatmap.py"
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement sur `http://localhost:8501`).

## 📁 Structure du projet

- `graph codes/` : Contient le code source de l'application Streamlit ainsi que d'autres scripts de visualisation 2D et 3D des Grecques.
- `README.md` : Ce fichier.

---
Développé par [Thibault Berton](https://www.linkedin.com/in/thibault-berton/)
