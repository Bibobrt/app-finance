import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import scipy.stats as stats
import datetime

# --- CONFIGURATION & INPUTS ---

def get_user_inputs():
    print("\n--- STOCK DASHBOARD & MONTE CARLO ANALYSIS ---")
    print("----------------------------------------------")
    print("TIP: Press ENTER to accept [Default] values.\n")
    
    # 1. Ticker
    ticker = input("Enter Stock Ticker [AAPL]: ").strip().upper()
    if not ticker:
        ticker = "AAPL"
        print(f"Using Default: {ticker}")
       
    # 2. Benchmarks 
    print("\n--- BENCHMARKS ---")
    
    print("\nCommon Risk-Free Rate Tickers:")
    print("  ^IRX         (US T-Bill 13 Weeks) [Default]")
    print("  ^DE10YT=RR   (German Bund 10Y - Euro Risk-Free)")
    
    rfr_in = input("Enter Risk-Free Rate Ticker [^IRX]: ").strip().upper()
    rfr_ticker = rfr_in if rfr_in else "^IRX"
    if not rfr_in: print(f"Using Default: {rfr_ticker}")
    
    print("\nCommon Market Portfolios:")
    print("  ^GSPC        (S&P 500 - US) [Default]")
    print("  ^STOXX50E    (Euro STOXX 50 - EU)")
    print("  ^FCHI        (CAC 40 - France)")
    
    mkt_in = input("Enter Benchmark Market Ticker [^GSPC]: ").strip().upper()
    market_ticker = mkt_in if mkt_in else "^GSPC"
    if not mkt_in: print(f"Using Default: {market_ticker}")

    # 3. Monte Carlo Params
    print("\n--- MONTE CARLO PARAMETERS ---")
    try:
        years_in = input("Enter Simulation Horizon in years (e.g. 0.5, 1) [1.0]: ")
        years = float(years_in) if years_in.strip() else 1.0
    except ValueError:
        years = 1.0
        print(f"Using Default: {years} years")
    if not years_in.strip(): print(f"Using Default: {years} years")
        
    try:
        sims_in = input("Enter number of Monte Carlo paths [1000]: ")
        num_simulations = int(sims_in) if sims_in.strip() else 1000
    except ValueError:
        num_simulations = 1000
        print(f"Using Default: {num_simulations} simulations")
    if not sims_in.strip(): print(f"Using Default: {num_simulations} simulations")

    return ticker, rfr_ticker, market_ticker, years, num_simulations

# --- DATA FETCHING (SIMPLIFIED) ---

def fetch_data_simple(ticker, period="5y"):
    """
    Fetches data directly from Yahoo Finance.
    NO error handling, NO retry logic, NO local files.
    """
    print(f"Fetching data for {ticker} from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    # Cleanup
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.tz_localize(None)
        
    # Standardize Price Column
    if 'Adj Close' in data.columns:
        data['Price'] = data['Adj Close']
    elif 'Close' in data.columns:
        data['Price'] = data['Close']
    else:
        # Fallback
        if not data.empty:
            data['Price'] = data.iloc[:, 0]
        
    return data

# --- CALCULATIONS: DASHBOARD ---

def compute_indicators(df):
    df = df.copy()
    
    # Price Trends
    df['SMA_50'] = df['Price'].rolling(window=50).mean()
    df['SMA_200'] = df['Price'].rolling(window=200).mean()
    
    # Returns
    df['Returns'] = df['Price'].pct_change()
    df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
    df['Cum_Returns'] = (1 + df['Returns']).cumprod() - 1
    
    # Volatility & Drawdown
    df['Vol_20d'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Vol_60d'] = df['Log_Returns'].rolling(window=60).std() * np.sqrt(252)
    
    df['Running_Max'] = df['Price'].cummax()
    df['Drawdown'] = (df['Price'] / df['Running_Max']) - 1
    
    # RSI (14)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    if 'Volume' in df.columns:
        df['Vol_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    return df.dropna()

def calculate_stats(df, benchmark_data=None, risk_free_rate=0.04):
    rets = df['Returns']
    stats_dict = {}
    
    # Core Stats
    stats_dict['Last Price'] = df['Price'].iloc[-1]
    stats_dict['Total Return'] = (df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1
    stats_dict['Ann. Return'] = rets.mean() * 252
    stats_dict['Ann. Volatility'] = rets.std() * np.sqrt(252)
    stats_dict['Max Drawdown'] = df['Drawdown'].min()
    stats_dict['Sharpe Ratio'] = (stats_dict['Ann. Return'] - risk_free_rate) / stats_dict['Ann. Volatility']
    
    # Extremes & Dist
    stats_dict['Best Day'] = rets.max()
    stats_dict['Worst Day'] = rets.min()
    stats_dict['Skewness'] = rets.skew()
    stats_dict['Kurtosis'] = rets.kurtosis()
    
    # VaR 95%
    stats_dict['VaR 95%'] = np.percentile(rets, 5)
    cvar_rets = rets[rets <= stats_dict['VaR 95%']]
    stats_dict['CVaR 95%'] = cvar_rets.mean() if len(cvar_rets) > 0 else stats_dict['VaR 95%']
    
    # Beta / Alpha (if benchmark exists)
    beta, alpha = np.nan, np.nan
    market_ann_ret = np.nan
    
    if benchmark_data is not None and not benchmark_data.empty:
        # Align
        mkt_rets = benchmark_data['Price'].pct_change().dropna()
        combined = pd.concat([df['Returns'], mkt_rets], axis=1, join='inner').dropna()
        combined.columns = ['Stock', 'Market']
        
        if len(combined) > 50:
            cov_mat = np.cov(combined['Stock'], combined['Market'])
            beta = cov_mat[0, 1] / cov_mat[1, 1]
            market_ann_ret = mkt_rets.mean() * 252
            alpha = stats_dict['Ann. Return'] - (risk_free_rate + beta * (market_ann_ret - risk_free_rate))
            
    stats_dict['Beta'] = beta
    stats_dict['Alpha'] = alpha
    stats_dict['Market Return'] = market_ann_ret
    
    return stats_dict

# --- CALCULATIONS: MONTE CARLO ---

def run_monte_carlo(S0, mu, sigma, years, num_simulations):
    TRADING_DAYS = 252
    num_steps = int(years * TRADING_DAYS)
    dt = 1 / TRADING_DAYS
    
    # Parameter Estimation Output
    print("\n--- PARAMETER ESTIMATION ---")
    print("Source: Yahoo Finance")
    print(f"Last Price (S0): ${S0:.2f}")
    print(f"Annualized Volatility (sigma): {sigma:.2%}")
    print(f"Annualized Drift (mu): {mu:.2%}")
    
    print("\n--- SIMULATION ---")
    print(f"Simulating {num_steps} time steps over {years} years...")
    
    # GBM Model
    prices = np.zeros((num_steps + 1, num_simulations))
    prices[0] = S0
    
    Z = np.random.normal(0, 1, (num_steps, num_simulations))
    
    drift_term = (mu - 0.5 * sigma**2) * dt
    shock_term = sigma * np.sqrt(dt) * Z
    
    daily_returns = np.exp(drift_term + shock_term)
    price_paths = S0 * np.cumprod(daily_returns, axis=0)
    
    prices[1:] = price_paths
    return prices

def calculate_mc_risk_metrics(prices_mc):
    S0 = prices_mc[0, 0]
    final_prices = prices_mc[-1, :]
    
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    prob_loss = np.mean(final_prices < S0)
    
    price_05 = np.percentile(final_prices, 5) # VaR 95% level
    cvar_prices = final_prices[final_prices <= price_05]
    cvar_price_level = np.mean(cvar_prices) if len(cvar_prices) > 0 else price_05
    
    expected_return = (mean_price - S0) / S0
    
    metrics = {
        'Initial Price': S0,
        'Expected Price': mean_price,
        'Expected Return': expected_return,
        'Median Price': median_price,
        'Probability of Loss': prob_loss,
        'VaR 95%': price_05,
        'CVaR 95%': cvar_price_level
    }
    
    print("\n--- RISK METRICS (TERMINAL) ---")
    print(f"Initial Price:       ${S0:.2f}")
    print(f"Expected Price:      ${mean_price:.2f}")
    print(f"Expected Return:     {expected_return:+.2%}")
    print(f"Median Price:        ${median_price:.2f}")
    print(f"Probability of Loss: {prob_loss:.2%}")
    print(f"VaR 95% (Price):     ${price_05:.2f} (Worst 5% cut-off)")
    print(f"CVaR 95% (Price):    ${cvar_price_level:.2f} (Avg of worst 5%)")
    print("-------------------------------")
    
    return metrics

# --- VISUALIZATION ---

def plot_dashboard_overview(df, ticker, stats_dict, rf_rate, mkt_ticker):
    """ FIGURE 1: Price, SMA, RSI, Stats text """
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title(f"{ticker} - Overview")
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1.2])
    
    plt.suptitle(f"{ticker}: Price & Trend Analysis", fontsize=18, fontweight='bold')
    
    # 1. Price
    ax_price = plt.subplot(gs[0, :])
    ax_price.plot(df.index, df['Price'], label='Price', color='black', linewidth=1.5)
    ax_price.plot(df.index, df['SMA_50'], label='SMA 50', color='blue', linestyle='--', linewidth=1)
    ax_price.plot(df.index, df['SMA_200'], label='SMA 200', color='red', linestyle='--', linewidth=1)
    ax_price.set_title("Price Action & Moving Averages", loc='left', fontweight='bold')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    ax_price.set_ylabel("Price ($)")
    
    # Annotate Last Price
    last_date = df.index[-1]
    last_price = df['Price'].iloc[-1]
    ax_price.annotate(f"{last_price:.2f}", xy=(last_date, last_price), xytext=(5, 0), 
                      textcoords='offset points', fontweight='bold', va='center')
    ax_price.plot(last_date, last_price, 'o', color='black')
    
    # 2. RSI
    ax_rsi = plt.subplot(gs[1, 0])
    ax_rsi.plot(df.index, df['RSI'], color='purple', linewidth=1)
    ax_rsi.axhline(70, color='red', linestyle=':', linewidth=1)
    ax_rsi.axhline(30, color='green', linestyle=':', linewidth=1)
    ax_rsi.fill_between(df.index, 70, 30, color='gray', alpha=0.1)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI (14) - Momentum", loc='left', fontweight='bold')
    ax_rsi.grid(True, alpha=0.3)
    
    # 3. KPI Text
    ax_stats = plt.subplot(gs[1, 1])
    ax_stats.axis('off')
    ax_stats.set_title("Key Performance Indicators", fontweight='bold')
    
    beta_txt = f"{stats_dict['Beta']:.2f}" if not np.isnan(stats_dict['Beta']) else "N/A"
    alpha_txt = f"{stats_dict['Alpha']:+.4f}" if not np.isnan(stats_dict['Alpha']) else "N/A"
    
    text_str = f"""
    PERIOD: {df.index.min().date()} - {df.index.max().date()}
    
    RET:  Total {stats_dict['Total Return']:+.1%} | Ann {stats_dict['Ann. Return']:+.1%}
    RISK: Vol {stats_dict['Ann. Volatility']:.1%} | DD {stats_dict['Max Drawdown']:.1%}
    
    SHARPE (Rf={rf_rate:.0%}): {stats_dict['Sharpe Ratio']:.2f}
    BETA ({mkt_ticker}): {beta_txt}
    ALPHA: {alpha_txt}
    
    EXTREMES
    --------
    Best Day:  {stats_dict['Best Day']:+.2%}
    Worst Day: {stats_dict['Worst Day']:+.2%}
    
    DISTRIBUTION
    ------------
    VaR 95%:   {stats_dict['VaR 95%']:.2%}
    Skew:      {stats_dict['Skewness']:.2f}
    """
    ax_stats.text(0.05, 0.96, text_str, fontsize=10, fontfamily='monospace', va='top',
                  bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_risk_profile(df, ticker, stats_dict):
    """ FIGURE 2: Risk Analysis (Updated Layout - No Dail Log Returns) """
    fig = plt.figure(figsize=(14, 12)) # Slightly taller for better spacing
    fig.canvas.manager.set_window_title(f"{ticker} - Risk Profile")
    
    # Layout: 3 Rows
    # Row 1: Cum Ret | Drawdown
    # Row 2: Rolling Vol (Full Width)
    # Row 3: Hist | QQ
    
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.8, 1])
    plt.suptitle(f"{ticker}: Risk & Distribution Profile", fontsize=18, fontweight='bold')
    
    # 1. Cum Returns
    ax_ret = plt.subplot(gs[0, 0])
    ax_ret.fill_between(df.index, df['Cum_Returns'], color='green', alpha=0.1)
    ax_ret.plot(df.index, df['Cum_Returns'], color='green', linewidth=1)
    ax_ret.axhline(0, color='black', linewidth=0.5)
    ax_ret.set_title("Cumulative Returns", loc='left', fontweight='bold')
    
    # 2. Drawdown
    ax_dd = plt.subplot(gs[0, 1])
    ax_dd.fill_between(df.index, df['Drawdown'], 0, color='red', alpha=0.3)
    ax_dd.plot(df.index, df['Drawdown'], color='red', linewidth=0.8)
    ax_dd.set_title("Drawdown (Underwater)", loc='left', fontweight='bold')
    
    # 3. Rolling Vol (Full Width now)
    ax_vol = plt.subplot(gs[1, :])
    ax_vol.plot(df.index, df['Vol_20d'], label='20d Vol', color='purple', linewidth=1.2)
    ax_vol.plot(df.index, df['Vol_60d'], label='60d Vol', color='orange', linewidth=1)
    ax_vol.set_title("Rolling Volatility (Ann.)", loc='left', fontweight='bold')
    ax_vol.legend(loc='upper left', fontsize='small')
    
    # 4. Hist vs Normal
    ax_hist = plt.subplot(gs[2, 0])
    ax_hist.hist(df['Returns'], bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.6)
    mu, std = stats.norm.fit(df['Returns'])
    xmin, xmax = ax_hist.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax_hist.plot(x, p, 'k', linewidth=1.5, label='Normal')
    ax_hist.axvline(stats_dict['VaR 95%'], color='red', linestyle='--', label='VaR 95%')
    ax_hist.set_title("Return Dist. vs Normal", loc='left', fontweight='bold')
    ax_hist.legend(fontsize='small')
    
    # 5. QQ Plot
    ax_qq = plt.subplot(gs[2, 1])
    stats.probplot(df['Returns'], dist="norm", plot=ax_qq)
    ax_qq.set_title("QQ Plot (Fat Tails Check)", loc='left', fontweight='bold')
    ax_qq.get_lines()[0].set_markerfacecolor('blue')
    ax_qq.get_lines()[0].set_markersize(2.0)
    
    # Improve Spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.25) 

def plot_monte_carlo(prices_mc, years, ticker, mc_metrics):
    """ FIGURE 3: Monte Carlo Analysis (Improved Spacing) """
    fig = plt.figure(figsize=(16, 12))
    fig.canvas.manager.set_window_title(f"{ticker} - Monte Carlo")
    plt.suptitle(f"Monte Carlo Simulation Analysis: {ticker}", fontsize=16)
    
    num_steps = prices_mc.shape[0]
    time_axis = np.linspace(0, years, num_steps)
    
    # 1. Spaghetti (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    display_paths = min(prices_mc.shape[1], 200)
    ax1.plot(time_axis, prices_mc[:, :display_paths], alpha=0.1, linewidth=1)
    ax1.set_title(f"1. Future MC Paths ({display_paths} samples)")
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True)
    
    # 2. Fan Chart (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    percentiles = np.percentile(prices_mc, [5, 25, 50, 75, 95], axis=1)
    ax2.fill_between(time_axis, percentiles[0], percentiles[4], color='gray', alpha=0.15, label="5th-95th Pctl")
    ax2.fill_between(time_axis, percentiles[1], percentiles[3], color='blue', alpha=0.2, label="25th-75th Pctl")
    ax2.plot(time_axis, percentiles[2], color='red', linewidth=2, label="Median Forecast")
    ax2.set_title("2. Fan Chart (Risk Cone)")
    ax2.set_xlabel("Time (Years)")
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # 3. Terminal Distribution (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    final_prices = prices_mc[-1, :]
    ax3.hist(final_prices, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    kde = stats.gaussian_kde(final_prices)
    x_grid = np.linspace(min(final_prices), max(final_prices), 200)
    ax3.plot(x_grid, kde(x_grid), color='darkblue', linewidth=2, label='Density')
    ax3.axvline(prices_mc[0,0], color='green', linestyle='--', linewidth=2, label='Start Price')
    ax3.set_title("3. Terminal Price Distribution")
    ax3.set_xlabel("Price ($)")
    ax3.legend()
    
    # 4. Metrics Table (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    # Removed title "4. Risk Metrics Summary" to avoid overlap.
    # It is now part of the text box header.
    
    table_text = f"""
    4. RISK METRICS SUMMARY
    =====================================
    Simulations:       {prices_mc.shape[1]}
    Time Horizon:      {years} Years
    
    Initial Price:     ${mc_metrics['Initial Price']:.2f}
    Expected Price:    ${mc_metrics['Expected Price']:.2f}
    Median Price:      ${mc_metrics['Median Price']:.2f}
    Expected Return:   {mc_metrics['Expected Return']:+.2%}
    
    KEY RISKS
    -------------------------------------
    Probability of Loss: {mc_metrics['Probability of Loss']:.2%}
    
    VaR 95% (Price):     ${mc_metrics['VaR 95%']:.2f}
    (Worst 5% Outcome)
    
    CVaR 95% (Price):    ${mc_metrics['CVaR 95%']:.2f}
    (Average of Worst 5%)
    """
    
    ax4.text(0.5, 0.5, table_text, fontsize=12, fontfamily='monospace', va='center', ha='center',
             bbox=dict(boxstyle="round,pad=1", fc="white", ec="black", alpha=0.8))

    # Improve Spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.25)


def plot_summary_window(stats_dict, mc_metrics, ticker, years, mkt_ticker, rf_scalar, mu, sigma, num_sims):
    """ FIGURE 4: Summary Text Window (2 Columns) """
    fig = plt.figure(figsize=(14, 8)) # Wider
    fig.canvas.manager.set_window_title(f"{ticker} - Executive Summary")
    
    beta_txt = f"{stats_dict['Beta']:.2f}" if not np.isnan(stats_dict['Beta']) else "N/A"
    alpha_txt = f"{stats_dict['Alpha']:+.4f}" if not np.isnan(stats_dict['Alpha']) else "N/A"
    mkt_ret_txt = f"{stats_dict['Market Return']:.2%}" if not np.isnan(stats_dict['Market Return']) else "N/A"
    
    # --- LEFT COLUMN: DASHBOARD ---
    ax_left = plt.subplot(1, 2, 1)
    ax_left.axis('off')
    
    left_text = f"""
    FINANCIAL DASHBOARD: {ticker}
    ========================================
    
    PERFORMANCE
    ----------------------------------------
    Current Price:      ${stats_dict['Last Price']:.2f}
    Ann. Return:        {stats_dict['Ann. Return']:+.2%}
    Ann. Volatility:    {stats_dict['Ann. Volatility']:.2%}
    Max Drawdown:       {stats_dict['Max Drawdown']:.2%}
    Sharpe Ratio:       {stats_dict['Sharpe Ratio']:.2f} 
                        (Rf={rf_scalar:.1%})
    
    BENCHMARKS
    ----------------------------------------
    Market ({mkt_ticker}) Ret: {mkt_ret_txt}
    Beta:               {beta_txt}
    Alpha (Jensen):     {alpha_txt}
    
    DISTRIBUTION & EXTREMES
    ----------------------------------------
    Skewness:           {stats_dict['Skewness']:.2f}
    Kurtosis:           {stats_dict['Kurtosis']:.2f}
    VaR 95% (1-day):    {stats_dict['VaR 95%']:.2%}
    Best Day:           {stats_dict['Best Day']:+.2%}
    Worst Day:          {stats_dict['Worst Day']:+.2%}
    """
    
    ax_left.text(0.05, 0.95, left_text, fontsize=11, fontfamily='monospace', va='top', linespacing=1.4)

    # --- RIGHT COLUMN: MONTE CARLO ---
    ax_right = plt.subplot(1, 2, 2)
    ax_right.axis('off')
    
    right_text = f"""
    MONTE CARLO SIMULATION
    ========================================
    
    PARAMETERS (Est. from History)
    ----------------------------------------
    - Annualized Volatility: {sigma:.2%}
    - Annualized Drift:      {mu:.2%}
    - Horizon:               {years} Years
    - Simulations:           {num_sims}
    - Start Price:           ${mc_metrics['Initial Price']:.2f}
    
    OUTCOMES (Terminal)
    ----------------------------------------
    Expected Price:      ${mc_metrics['Expected Price']:.2f}
    Median Price:        ${mc_metrics['Median Price']:.2f}
    Expected Return:     {mc_metrics['Expected Return']:+.2%}
    
    RISK METRICS
    ----------------------------------------
    Probability of Loss: {mc_metrics['Probability of Loss']:.2%}
    
    VaR 95% (Price):     ${mc_metrics['VaR 95%']:.2f} 
                         (Worst 5%)
    CVaR 95% (Price):    ${mc_metrics['CVaR 95%']:.2f} 
                         (Avg Worst 5%)
    """
    
    ax_right.text(0.05, 0.95, right_text, fontsize=11, fontfamily='monospace', va='top', linespacing=1.4)
    
    plt.tight_layout()

# --- MAIN EXECUTION ---

def main():
    # 1. Inputs
    ticker, rfr_ticker, mkt_ticker, years, num_sims = get_user_inputs()
    
    # 2. Benchmarks (Before Dashboard)
    print(f"\n[Benchmarks] Fetching {rfr_ticker} from Yahoo Finance...")
    rfr_data = fetch_data_simple(rfr_ticker)
    
    print(f"[Benchmarks] Fetching {mkt_ticker} from Yahoo Finance...")
    mkt_data = fetch_data_simple(mkt_ticker)
    
    # Determine Risk Free Rate
    rf_scalar = 0.04 # Default
    if not rfr_data.empty:
        # Heuristic: If > 0.2 it's probably %. ^IRX is %, so divide by 100
        val = rfr_data['Price'].iloc[-1]
        rf_scalar = val / 100.0
        print(f"\n[Benchmarks] Using Risk-Free Rate from {rfr_ticker}: {rf_scalar:.2%}")
    else:
        print(f"\n[Benchmarks] Using Default Risk-Free Rate: {rf_scalar:.2%}")

    # 3. Main Stock Data
    print("\n--- CALCULATING METRICS ---")
    df = fetch_data_simple(ticker)
    if df.empty:
        print(f"Error: No data found for {ticker}")
        return

    # 4. Dashboard Stats
    df_calc = compute_indicators(df)
    stats_dict = calculate_stats(df_calc, mkt_data, rf_scalar)
    
    if not np.isnan(stats_dict.get('Market Return', np.nan)):
         print(f"[Benchmarks] Market ({mkt_ticker}) Ann. Return: {stats_dict['Market Return']:.2%}")

    # Console Output Frame
    print("\n" + "="*40)
    print(f" FINANCIAL DASHBOARD: {ticker}")
    print("="*40)
    print(f"Period: {df.index.min().date()} -> {df.index.max().date()} ({len(df)} days)")
    print("-" * 40)
    print(f"Current Price:      ${df['Price'].iloc[-1]:.2f}")
    if not df_calc['SMA_50'].isnull().all():
        print(f"SMA 50:             ${df_calc['SMA_50'].iloc[-1]:.2f}")
    if not df_calc['SMA_200'].isnull().all():
        print(f"SMA 200:            ${df_calc['SMA_200'].iloc[-1]:.2f}")
    print("-" * 40)
    print(f"Ann. Return:        {stats_dict['Ann. Return']:+.2%}")
    print(f"Ann. Volatility:    {stats_dict['Ann. Volatility']:.2%}")
    print(f"Max Drawdown:       {stats_dict['Max Drawdown']:.2%}")
    print(f"Sharpe Ratio:       {stats_dict['Sharpe Ratio']:.2f} (Rf={rf_scalar:.1%})")
    print("-" * 40)
    beta_txt = f"{stats_dict['Beta']:.2f}" if not np.isnan(stats_dict['Beta']) else "N/A"
    alpha_txt = f"{stats_dict['Alpha']:+.4f}" if not np.isnan(stats_dict['Alpha']) else "N/A"
    print(f"Beta vs {mkt_ticker}:    {beta_txt}")
    print(f"Alpha ( Jensen):    {alpha_txt}")
    print("-" * 40)
    print(f"Skewness:           {stats_dict['Skewness']:.2f} (0 = Normal)")
    print(f"Kurtosis:           {stats_dict['Kurtosis']:.2f} (3 = Normal)")
    print(f"VaR 95% (1-day):    {stats_dict['VaR 95%']:.2%}")
    print("="*40)

    print("\nGenerating Dashboard...")
    
    # 5. Monte Carlo
    # Estimate params using full history
    log_rets = df_calc['Log_Returns']
    sigma = log_rets.std() * np.sqrt(252)
    mu = log_rets.mean() * 252 + 0.5 * sigma**2
    
    current_price = df['Price'].iloc[-1]
    
    prices_mc = run_monte_carlo(current_price, mu, sigma, years, num_sims)
    mc_metrics = calculate_mc_risk_metrics(prices_mc)
    
    # 6. Plotting
    plot_dashboard_overview(df_calc, ticker, stats_dict, rf_scalar, mkt_ticker)
    plot_risk_profile(df_calc, ticker, stats_dict)
    plot_monte_carlo(prices_mc, years, ticker, mc_metrics)
    plot_summary_window(stats_dict, mc_metrics, ticker, years, mkt_ticker, rf_scalar, mu, sigma, num_sims)
    
    plt.show()

if __name__ == "__main__":
    main()
