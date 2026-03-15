import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define the watchlist with all requested tickers
WATCHLIST = {
    "Indices": {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^RUT": "Russell 2000",
        "^STOXX50E": "Euro Stoxx 50",
        "^FCHI": "CAC 40",
        "^GDAXI": "DAX",
        "^FTSE": "FTSE 100",
        "^IBEX": "IBEX 35",      # Spain
        "^N225": "Nikkei 225",   # Japan
        "^HSI": "Hang Seng",     # Hong Kong
        "^VIX": "VIX"
    },
    "Rates (US & EU)": {
        "^IRX": "13 Week T-Bill",
        "^FVX": "5 Year T-Note",
        "^TNX": "10 Year T-Note",
        "^TYX": "30 Year T-Bond"
        # European yields, Euribor, ESTR, SOFR will be added dynamically
    },
    "FX": {
        "EURUSD=X": "EUR/USD",
        "EURGBP=X": "EUR/GBP",
        "USDJPY=X": "USD/JPY",
        "GBPUSD=X": "GBP/USD",
        "USDCHF=X": "USD/CHF",
        "DX-Y.NYB": "Dollar Index"
    },
    "Commodities": {
        "BZ=F": "Brent Crude",
        "CL=F": "WTI Crude",
        "GC=F": "Gold",
        "SI=F": "Silver",
        "HG=F": "Copper",
        "NG=F": "Natural Gas"
    },
    "Crypto": {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum"
    },
    "Tech & Mega Caps": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "AMZN": "Amazon",
        "GOOGL": "Google",
        "META": "Meta",
        "TSLA": "Tesla",
        "TSM": "TSMC"
    },
    "Financials (US)": {
        "JPM": "JPMorgan Chase",
        "GS": "Goldman Sachs",
        "MS": "Morgan Stanley",
        "BAC": "Bank of America",
        "V": "Visa"
    },
    "Pharma & Healthcare": {
        "LLY": "Eli Lilly",
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth",
        "NOVO-B.CO": "Novo Nordisk"
    },
    "Energy & Industrials": {
        "XOM": "Exxon Mobil",
        "CVX": "Chevron",
        "CAT": "Caterpillar",
        "GE": "General Electric"
    },
    "Retail & Consumer": {
        "WMT": "Walmart",
        "COST": "Costco",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola"
    },
    "European Key Players": {
        "ASML.AS": "ASML (Tech)",
        "MC.PA": "LVMH (Luxury)",
        "OR.PA": "L'Oréal (Consumer)",
        "RMS.PA": "Hermès (Luxury)",
        "SAP.DE": "SAP (Tech)",
        "SIE.DE": "Siemens (Indus)",
        "AIR.PA": "Airbus (Aero)",
        "TTE.PA": "TotalEnergies (Energy)",
        "NESN.SW": "Nestlé (Food)",
        "ROG.SW": "Roche (Pharma)"
    }
}

def fetch_fred_rate(series_id, name):
    """
    Fetches the latest rate from FRED (Federal Reserve Economic Data) via CSV.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            # Iterate backwards to find the latest valid numerical value (FRED uses '.' for missing data)
            for i in range(len(lines) - 1, 0, -1):
                date_str, value_str = lines[i].split(',')
                if value_str.strip() and value_str.strip() != '.':
                    return {
                        "Ticker": series_id,
                        "Name": name,
                        "Price": float(value_str),
                        "Change": 0.0,
                        "% Change": 0.0,
                        "Source": "FRED"
                    }
    except Exception as e:
        print(f"    Error fetching {name} from FRED: {e}")
    return None

def fetch_euribor_3m():
    """
    Scrapes the 3-month Euribor rate from euribor-rates.eu.
    """
    url = "https://www.euribor-rates.eu/en/current-euribor-rates/2/euribor-rate-3-months/"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        # Simple regex to find the rate in the table
        # Structure often looks like: <tr><td>Current rate (07/02/2026)</td><td class="text-right">3.123 %</td></tr>
        # We look for the first occurrence of a percentage after "Current rate" or similar context
        
        # Find the main rate table
        match = re.search(r'Current euribor 3 months.*?<td[^>]*>([0-9]+\.[0-9]+)\s*%', response.text, re.IGNORECASE | re.DOTALL)
        if not match:
             # Fallback: look for just a number followed by % in a table data cell
             match = re.search(r'<td[^>]*class="text-right"[^>]*>\s*([0-9]+\.[0-9]+)\s*%\s*</td>', response.text)

        if match:
            rate = float(match.group(1))
            return {
                "Ticker": "EURIBOR3M",
                "Name": "Euribor 3 Month",
                "Price": rate,
                "Change": 0.0, # parsing change needs more complex logic
                "% Change": 0.0,
                "Source": "Web"
            }
    except Exception as e:
        print(f"    Error fetching Euribor: {e}")
    return None

def get_market_data(watchlist):
    """
    Fetches market data for all tickers in the watchlist.
    Returns a dictionary structured by category.
    """
    print(f"Fetching market data... ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    results = {}
    
    # 1. Fetch Standard Yahoo Data
    for category, tickers_dict in watchlist.items():
        results[category] = []
        
        # Prepare Yahoo tickers
        tickers_list = list(tickers_dict.keys())
        if not tickers_list:
            continue
            
        print(f"  - Fetching {category} (Yahoo)...")
        try:
            data = yf.download(tickers_list, period="5d", progress=False, auto_adjust=True)
            
            if data.empty:
                print(f"    Warning: No data found for {category}")
                continue

            # Handle returned data structure (MultiIndex vs Single Index)
            # If we only requested one ticker, columns might not be MultiIndex depending on yfinance version
            # But usually if we pass a list, even of 1, it tries to be consistent. 
            # However, safer to check.
            
            is_multi = isinstance(data.columns, pd.MultiIndex)
            
            # Get latest valid data points
            # We want the last row that isn't all NaN, but 'Close' usually has data
            try:
                # Sometimes the last row is missing data for some tickers if they update at different times
                # We'll take the simple approach: last row.
                current_prices = data['Close'].iloc[-1]
                prev_prices = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[-1]
            except IndexError:
                # Not enough data
                continue

            for ticker, name in tickers_dict.items():
                try:
                    price = 0.0
                    prev_close = 0.0
                    
                    if is_multi:
                        if ticker in current_prices.index:
                            price = current_prices[ticker]
                            prev_close = prev_prices[ticker]
                    else:
                        # If simple index, check if the column name matches or if it's the only column
                        if ticker in data['Close']:
                             price = data['Close'][ticker].iloc[-1]
                             prev_close = data['Close'][ticker].iloc[-2]
                        elif len(tickers_list) == 1:
                             # If we asked for 1 ticker, the series IS the ticker data
                             price = data['Close'].iloc[-1]
                             prev_close = data['Close'].iloc[-2]

                    if pd.isna(price):
                         # Try to find last valid value
                         # This is expensive to do for all, so we skip for now or set to 0
                         price = 0.0

                    if price != 0.0:
                        change = price - prev_close
                        pct_change = (change / prev_close) * 100
                    else:
                        change = 0.0
                        pct_change = 0.0

                    results[category].append({
                        "Ticker": ticker,
                        "Name": name,
                        "Price": price,
                        "Change": change,
                        "% Change": pct_change
                    })
                except Exception as e:
                    results[category].append({
                        "Ticker": ticker,
                        "Name": name,
                        "Price": 0.0,
                        "Change": 0.0,
                        "% Change": 0.0,
                        "Error": "N/A"
                    })

        except Exception as e:
            print(f"    Error fetching {category}: {e}")

    # 2. Fetch Special Rates (FRED / Web)
    print("  - Fetching Official Rates (FRED/Web)...")
    special_rates = []
    
    # SOFR from FRED
    sofr = fetch_fred_rate("SOFR", "SOFR (Overnight)")
    if sofr: special_rates.append(sofr)
    
    # ESTR from FRED
    estr = fetch_fred_rate("ECBESTRVOLWGTTRMDMNRT", "€STR (Overnight)")
    if estr: special_rates.append(estr)

    # European Bond Yields from FRED
    uk_10y = fetch_fred_rate("IRLTLT01GBM156N", "UK 10Y Yield")
    if uk_10y: special_rates.append(uk_10y)
    uk_5y = fetch_fred_rate("IR3TIB01GBM156N", "UK 3M Libor (Proxy)") 
    if uk_5y: special_rates.append(uk_5y)
    
    ger_10y = fetch_fred_rate("IRLTLT01DEM156N", "Germany 10Y Yield")
    if ger_10y: special_rates.append(ger_10y)
    ger_5y = fetch_fred_rate("IR3TIB01DEM156N", "Germany 3M Libor (Proxy)")
    if ger_5y: special_rates.append(ger_5y)
    
    fra_10y = fetch_fred_rate("IRLTLT01FRM156N", "OAT 10Y (France)")
    if fra_10y: special_rates.append(fra_10y)
    fra_5y = fetch_fred_rate("IR3TIB01FRM156N", "France 3M (Proxy)")
    if fra_5y: special_rates.append(fra_5y)

    # 3. Inflation Data (FRED)
    print("  - Fetching Inflation (FRED)...")
    inflation_data = []
    us_inf = fetch_fred_rate("CPALTT01USM659N", "US Inflation (YoY)")
    if us_inf: inflation_data.append(us_inf)
    fr_inf = fetch_fred_rate("CPALTT01FRM659N", "France Inflation (YoY)")
    if fr_inf: inflation_data.append(fr_inf)
    de_inf = fetch_fred_rate("CPALTT01DEM659N", "Germany Inflation (YoY)")
    if de_inf: inflation_data.append(de_inf)
    uk_inf = fetch_fred_rate("CPALTT01GBM659N", "UK Inflation (YoY)")
    if uk_inf: inflation_data.append(uk_inf)
    results["Inflation"] = inflation_data

    # 4. Key Financial Leaders
    results["Key Financial Leaders"] = [
        {"Ticker": "FED", "Name": "Jerome Powell (Fed Chair)", "Price": 0, "Change": 0, "% Change": 0, "Source": "Info"},
        {"Ticker": "ECB", "Name": "Christine Lagarde (ECB Pres)", "Price": 0, "Change": 0, "% Change": 0, "Source": "Info"},
        {"Ticker": "BIS", "Name": "Agustín Carstens (BIS GM)", "Price": 0, "Change": 0, "% Change": 0, "Source": "Info"}
    ]

    # Euribor 3M from Web
    euribor = fetch_euribor_3m()
    if euribor: special_rates.append(euribor)
    
    # Add to "Rates (US & EU)" category
    if "Rates (US & EU)" in results:
        results["Rates (US & EU)"].extend(special_rates)
    else:
        results["Rates (US & EU)"] = special_rates

    return results

def export_to_pdf(results, filename="market_snapshot.pdf"):
    """
    Exports the market snapshot data to a PDF file using matplotlib.
    """
    print(f"\nExporting to PDF: {filename} ...")
    
    with PdfPages(filename) as pdf:
        # Create a new page for each set of data or fit multiple on one
        # For simplicity, let's try to fit everything on pages
        
        # Prepare data for all tables
        # We'll create a figure for each page as needed
        
        # Configuration
        items_per_page = 30 # Approximation
        current_items = 0
        
        fig, ax = plt.subplots(figsize=(8.5, 11)) # Letter size
        ax.axis('off')
        
        y_pos = 0.95
        ax.text(0.5, 0.98, f"Market Snapshot - {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
        
        col_labels = ["Name", "Price", "Change", "% Chg"]
        col_widths = [0.45, 0.20, 0.175, 0.175]
        
        for category, items in results.items():
            if not items: continue
            
            # Check if we need a new page
            if y_pos < 0.1:
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                y_pos = 0.95
                ax.text(0.5, 0.98, f"Market Snapshot (Cont.)", 
                        transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold')
            
            # Category Header
            ax.text(0.05, y_pos, category.upper(), transform=ax.transAxes, 
                    fontsize=12, fontweight='bold', color='darkblue')
            y_pos -= 0.03
            
            # Prepare table data
            cell_text = []
            cell_colors = []
            
            for item in items:
                price = item.get('Price', 0.0)
                change = item.get('Change', 0.0)
                pct = item.get('% Change', 0.0)
                
                # Colors
                bg_color = 'white'
                if change > 0: text_color = 'green'
                elif change < 0: text_color = 'red'
                else: text_color = 'black'
                
                # Formatting
                if item.get("Source") in ["FRED", "Web"] and change == 0:
                     change_str = "--"
                     pct_str = "--"
                     text_color = 'black'
                else:
                    change_str = f"{change:+.2f}"
                    pct_str = f"{pct:+.2f}%"

                p_str = f"{price:,.2f}"
                if 'Error' in item: p_str = "N/A"
                
                row = [item['Name'], p_str, change_str, pct_str]
                cell_text.append(row)
                cell_colors.append(['white', 'white', 'white', 'white']) # Can customize
            
            # Draw Table
            # Table height estimation
            row_height = 0.035
            table_height = row_height * (len(cell_text) + 1)
            
            # Check if table fits
            if y_pos - table_height < 0.05:
                 # If table is too big for remaining space, push to next page
                 # (Handling split tables is complex, assume categories fit on a page for now)
                pdf.savefig(fig)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                y_pos = 0.95
                ax.text(0.05, y_pos, category.upper() + " (Cont.)", transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', color='darkblue')
                y_pos -= 0.03

            the_table = ax.table(cellText=cell_text, colLabels=col_labels, 
                                 loc='top', bbox=[0.05, y_pos - table_height, 0.9, table_height])
            
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            
            # Optional: Color text in table cells based on value (Change column)
            # This is tricky with plt.table as we need to access cells
            for i, row_data in enumerate(cell_text):
                # Columns 3 and 4 are Change and % Change
                # We need to access (row_idx + 1, col_idx) because 0 is header
                change_val = items[i].get('Change', 0.0)
                color = 'black'
                if change_val > 0: color = 'green'
                elif change_val < 0: color = 'red'
                
                the_table[(i+1, 2)].get_text().set_color(color)
                the_table[(i+1, 3)].get_text().set_color(color)

            y_pos -= (table_height + 0.05) # Space for next category

        pdf.savefig(fig)
        plt.close(fig)
        
    print(f"Done. Saved to {filename}")

def display_snapshot(results):
    """
    Displays the market snapshot in a clean tabular format.
    """
    print("\n" + "="*80)
    print(f"MARKET SNAPSHOT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    for category, items in results.items():
        print(f"--- {category.upper()} ---")
        # Header
        print(f"{'Name':<40} {'Price':>12} {'Change':>10} {'% Change':>10}")
        print("-" * 80)
        
        for item in items:
            name = item['Name']
            price = item['Price']
            change = item['Change']
            pct = item['% Change']
            
            # Formatting
            GREEN = '\033[92m'
            RED = '\033[91m'
            RESET = '\033[0m'
            
            if 'Error' in item:
                line = f"{name:<40} {'N/A':>12} {'N/A':>10} {'N/A':>10}"
            else:
                price_fmt = f"{price:,.2f}"
                
                # If source is FRED/Web, we might not have change data (0.0)
                # We can visually indicate this or just show 0.00
                if change == 0.0 and pct == 0.0 and item.get("Source") in ["FRED", "Web"]:
                     change_fmt = " -- "
                     pct_fmt = " -- "
                     color = RESET
                else:
                    change_fmt = f"{change:+.2f}"
                    pct_fmt = f"{pct:+.2f}%"
                    color = GREEN if change >= 0 else RED
                
                if category == "Key Financial Leaders" or item.get("Source") == "Info":
                    line = f"{name:<40} {'--':>12} {'--':>10} {'--':>10}"
                else:
                    line = f"{name:<40} {price_fmt:>12} {color}{change_fmt:>10} {pct_fmt:>10}{RESET}"
            
            print(line)
        print("\n")

if __name__ == "__main__":
    try:
        snapshot_data = get_market_data(WATCHLIST)
        display_snapshot(snapshot_data)
        
        print("Note: 'Rates' are typically in yield points (e.g. 4.05 is 4.05%).")
        print("Disclaimer: Yahoo data is delayed. FRED/Euribor data is daily/historical close.")
        
        # Ask for PDF Export
        choice = input("\nExport to PDF? (y/n) [n]: ").strip().lower()
        if choice == 'y':
            filename = f"market_snapshot_{datetime.now().strftime('%Y%m%d')}.pdf"
            export_to_pdf(snapshot_data, filename)
            
    except KeyboardInterrupt:
        print("\nAborted.")
