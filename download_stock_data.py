import yfinance as yf
import pandas as pd
import time
import os

def download_data(tickers=None, period="2y", is_excel=False):
    print("\n--- STOCK DATA DOWNLOADER ---")
    print("-----------------------------")
    
    if tickers is None:
        user_input = input("Enter Stock Ticker(s) (e.g., AAPL, MSFT): ").strip().upper()
        if not user_input:
            print("Ticker cannot be empty.")
            return
        tickers = [t.strip() for t in user_input.split(',')]

    # Period selection if not provided and interactive
    if period == "ask":
        print("\nSelect Period:")
        print("1. Last 2 Years (Recommended for Monte Carlo)")
        print("2. Last 5 Years")
        print("3. Max History")
        
        p_choice = input("Choice (1-3): ").strip()
        
        period = "2y"
        if p_choice == '2':
            period = "5y"
        elif p_choice == '3':
            period = "max"
    
    # Format selection if not provided and interactive
    if is_excel == "ask":
        print("\nSelect Output Format:")
        print("1. CSV (.csv)")
        print("2. Excel (.xlsx)")
        
        f_choice = input("Choice (1/2): ").strip()
        is_excel = (f_choice == '2')
    
    print(f"\nProcessing {len(tickers)} tickers: {', '.join(tickers)}")
    
    for ticker in tickers:
        print(f"\nDownloading data for {ticker}...")
        
        # Download with retry logic
        max_retries = 3
        retry_delay = 2
        data = pd.DataFrame()
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # history() handles start/end parameters if period is None, but here we simplify to period only for batch
                # If sophisticated date range needed, logic can be expanded.
                data = stock.history(period=period)
                
                if not data.empty:
                    break
                    
                print(f"Attempt {attempt+1} failed. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2

        if data.empty:
            print(f"Failed to retrieve data for {ticker}. Skipping.")
            continue

        # Clean up data
        # Standardize index (remove timezone if present)
        data.index = data.index.tz_localize(None)
        
        # Filename
        t_str = str(int(time.time()))
        ext = ".xlsx" if is_excel else ".csv"
        filename = f"{ticker}_history_{t_str}{ext}"
        
        # Create data directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        full_path = os.path.join(data_dir, filename)
        
        try:
            if is_excel:
                data.to_excel(full_path)
            else:
                data.to_csv(full_path)
                
            print(f"SUCCESS! Saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving file: {e}")

if __name__ == "__main__":
    download_data(period="ask", is_excel="ask")
