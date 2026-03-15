#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:12:34 2026

@author: thibaultberton
"""

import numpy as np

def replicating_option_price(S0, K, r, T, sigma=None, option_type="call", S_up=None, S_down=None):
    """
    One-step binomial option pricing using the replicating portfolio method.
    Calculates u and d from volatility (sigma) OR uses explicit S_up/S_down.
    """
    if S_up is not None and S_down is not None:
        # Use explicit prices
        Su = S_up
        Sd = S_down
        u = Su / S0
        d = Sd / S0
    elif sigma is not None:
        # Use volatility
        u = np.exp(sigma * np.sqrt(T))
        d = 1 / u
        Su = S0 * u
        Sd = S0 * d
    else:
        raise ValueError("Either sigma or (S_up and S_down) must be provided.")

    # Payoffs
    if option_type == "call":
        fu = max(Su - K, 0)
        fd = max(Sd - K, 0)
    else:  # put
        fu = max(K - Su, 0)
        fd = max(K - Sd, 0)

    # Replicating portfolio
    # Delta (shares to hold)
    delta = (fu - fd) / (Su - Sd) # Note: Denominator is Su - Sd = S0 * (u - d)
    
    # Beta (amount to hold in risk-free asset/bonds)
    # B0 represents the value of the bond part TODAY. 
    # Usually Beta is amount of bonds to hold, but sometimes defined as amount of CASH.
    # Let's verify standard formula: V0 = delta * S0 + B
    # where B is the amount of money in the bank account.
    # The value of the bank account at T must be (fu - delta * Su).
    # So B = exp(-rT) * (fu - delta * Su)
    
    # Let's use the explicit Psi calculation for clarity.
    
    amount_in_bonds = np.exp(-r * T) * (u * fd - d * fu) / (u - d)

    # Option price today
    V0 = delta * S0 + amount_in_bonds

    return V0, delta, amount_in_bonds, u, d

if __name__ == "__main__":
    try:
        print("Pricing Option par Portefeuille de Réplication (1 Période)")
        print("Veuillez entrer les paramètres :")
        
        S0 = float(input("Prix actuel du sous-jacent (S0) : "))
        K = float(input("Prix d'exercice (K) : "))
        r = float(input("Taux sans risque (r, ex: 0.05) : "))
        T = float(input("Temps jusqu'à maturité (T, en années) : "))
        
        mode = input("Choisir méthode : [1] Volatilité, [2] Prix Explicites (Haut/Bas) : ")
        
        sigma = None
        S_up = None
        S_down = None
        
        if mode == "2":
            S_up = float(input("Prix Haut (S_up) en T1 : "))
            S_down = float(input("Prix Bas (S_down) en T1 : "))
        else:
            sigma = float(input("Volatilité (sigma, ex: 0.2) : "))
        
        type_input = input("Type d'option (c/p) [defaut: c] : ").lower()
        if type_input.startswith('p'):
            option_type = "put"
        else:
            option_type = "call"

        price, delta, bond_part, u_calc, d_calc = replicating_option_price(S0, K, r, T, sigma=sigma, option_type=option_type, S_up=S_up, S_down=S_down)

        print("\n" + "="*40)
        print(f"Résultats pour un {option_type.upper()} Européen")
        print("="*40)
        print(f"Paramètres utilisés/calculés :")
        if S_up is not None:
             print(f"  S_up (Haut)     : {S_up:.4f}")
             print(f"  S_down (Bas)    : {S_down:.4f}")
        print(f"  u (up factor)   : {u_calc:.4f}")
        print(f"  d (down factor) : {d_calc:.4f}")
        print("-"*40)
        print(f"Prix de l'option (V0) : {price:.4f}")
        print(f"Delta (Actions)       : {delta:.4f}")
        print(f"Partie Obligataire    : {bond_part:.4f}")
        print("-"*40)
        
        # Advice
        action = "ACHETER" if delta > 0 else "VENDRE"
        print(f"Stratégie de Réplication :")
        print(f"1. {action} {abs(delta):.4f} actions.")
        print(f"2. PLACER {bond_part:.4f} au taux sans risque." if bond_part > 0 else f"2. EMPRUNTER {abs(bond_part):.4f} au taux sans risque.")
        
    except ValueError:
        print("Erreur : Veuillez entrer des valeurs numériques valides.")
    except KeyboardInterrupt:
        print("\nAnnulé par l'utilisateur.")