import requests
from yahooquery import Ticker
#import yfinance as yf
import pandas as pd
from datetime import date, datetime
import sqlite3
import time
import warnings
import os
from tqdm import tqdm  # For progress bars
from pathlib import Path
import numpy as np  # For np.isnan
import json

api_key = os.environ.get('API_KEY')

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Function to initialize SQLite database with indexes
def init_db():
    """Create the SQLite DB (and file) if it does not exist yet."""
    db_path = Path('stocks.db')
    conn = sqlite3.connect(db_path)          # creates the file automatically
    c = conn.cursor()

    # NOTE: **no trailing comma** after the last column
    c.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            sector TEXT, 
            industry TEXT,
            company_name TEXT,
            category TEXT,
            type TEXT,
            last_updated TEXT,
            current_price REAL,
            yesterday_close REAL,
            week_high_52 REAL,
            week_low_52 REAL, 
            Turnover REAL, 
            averageTurnover REAL, 
            averageTurnover10Day REAL 
        )
    ''')

    c.execute('CREATE INDEX IF NOT EXISTS idx_symbol   ON stocks(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_category ON stocks(category)')

    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY
        )
    ''')

    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] SQLite database 'stocks.db' ready (created if missing)")

# Function to clear stocks table
def clear_db_table(warning_log):
    try:
        conn = sqlite3.connect('stocks.db')
        c = conn.cursor()
        c.execute('DELETE FROM stocks')
        conn.commit()
        print(f"[{datetime.now()}] Cleared stocks table")
    except Exception as e:
        warning_log.append(f"Error clearing table: {e}")
        print(f"[{datetime.now()}] Error clearing table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def get_last_updated():
    db_path = Path("stocks.db")
    # 1. Check if file exists
    if not db_path.exists():
        print("[INFO] Database 'stocks.db' not found. Returning None.")
        return None
    conn = None
    try:
        # 2. Connect with timeout
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()

        # 3. Query the first record
        cursor.execute("SELECT last_updated FROM Stocks ORDER BY Symbol ASC LIMIT 1")
        row = cursor.fetchone()

        if row and row[0]:
            # Parse ISO format: '2025-04-05 12:34:56' → date
            dt = datetime.fromisoformat(row[0])
            return dt.date()
        else:
            print("[INFO] No records found in Stocks table.")
            return None
    except sqlite3.Error as e:
        print(f"[ERROR] SQLite error: {e}")
        return None
    except ValueError as e:
        # Handles invalid ISO format in last_updated
        print(f"[ERROR] Invalid date format in last_updated: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return None
    finally:
        if conn:
            conn.close()


# Function to fetch stock data from Nasdaq API
def fetch_stocks(exchange, warning_log):
    url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=0&exchange={exchange}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        print(f"[{datetime.now()}] Fetching stocks from {exchange}...")
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])
            print(f"[{datetime.now()}] Fetched {len(rows)} stocks from {exchange}")
            return rows
        else:
            warning_log.append(f"Failed to fetch data for {exchange}: {response.status_code}")
            print(f"[{datetime.now()}] Failed to fetch data for {exchange}: {response.status_code}")
            return []
    except Exception as e:
        warning_log.append(f"Error fetching data for {exchange}: {e}")
        print(f"[{datetime.now()}] Error fetching data for {exchange}: {e}")
        return []

# Function to convert market cap
def convert_market_cap(market_cap):
    if not market_cap or market_cap in [0, "n/a", "unknown", "-", ""]:
        return None
    try:
        return float(market_cap)
    except (ValueError, TypeError):
        return None

def categorize_market_cap(market_cap):
    if market_cap is None:
        return "Unknown"
    if market_cap >= 200000000000:
        return "Mega-Cap"
    elif market_cap >= 10000000000:
        return "Large-Cap"
    elif market_cap >= 2000000000:
        return "Mid-Cap"
    else:
        return "Small-Cap or Below"

def retry_request(func, max_attempts=3, backoff_factor=2):
    """Wrapper to retry API requests with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            sleep_time = backoff_factor * (2 ** attempt)
            print(f"[{datetime.now()}] Retry {attempt + 1}/{max_attempts} failed: {str(e)}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
    return None

# Assuming categorize_market_cap is defined elsewhere

def safe_float(value, default=None, strip_dollar=False):
    """
    Safely convert value to float, handling 'NA', empty, None, or NaN.
    - default: Fallback value (e.g., None for market_cap, 0.0 for prices).
    - strip_dollar: If True, remove '$' before conversion.
    """
    if value is None or value == '' or value == 'NA' or (isinstance(value, float) and np.isnan(value)):
        return default
    
    # Convert to string for cleaning
    str_val = str(value)
    if strip_dollar:
        str_val = str_val.replace('$', '')
    str_val = str_val.replace(',', '')
    
    try:
        result = float(str_val)
        if np.isnan(result):  # Extra NaN check post-conversion
            return default
        return result
    except ValueError:
        return default
    
def fetch_price_data(symbols, all_stocks,warning_log, new_record):
    desired_categories = ["Mega-Cap", "Large-Cap", "Mid-Cap", "Small-Cap or Below"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stock_data = []  # Store all data in memory
    skipped_count = 0
    error_count = 0

    # Split symbols into smaller batches
    batch_size = 500
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    print(f"[{datetime.now()}] Processing {len(symbols)} symbols in {len(symbol_batches)} batches...")
    for batch_idx, batch in enumerate(symbol_batches):
        print(f"[{datetime.now()}] Fetching data for batch {batch_idx + 1}/{len(symbol_batches)} ({len(batch)} symbols)...")
        try:
            tickers = Ticker(batch, asynchronous=True)
            quote = tickers.price           
            summary_detail = tickers.summary_detail
            profile_data = tickers.asset_profile
            # Process each symbol in the batch
            for symbol in tqdm(batch, desc=f"Processing batch {batch_idx + 1}", unit="stock"):
                try:
                    # Get metadata              
                    if isinstance(profile_data[symbol], str): 
                        sector = ''
                        industry = ''
                    else:
                        sector = profile_data[symbol].get('sector','')
                        industry = profile_data[symbol].get('industry','')
                    #market cap
                    market_cap_raw = summary_detail[symbol].get('marketCap', '')  
                    market_cap = safe_float(market_cap_raw, default=None)
                    if market_cap is not None:
                        market_cap = round(market_cap, 2)
                        category = categorize_market_cap(market_cap)
                        # Skip if category is not desired
                        if category not in desired_categories:
                            skipped_count += 1
                            print(f"[{datetime.now()}] Skipped {symbol}: category {category} not in {desired_categories}")
                            continue
                    previousclose_raw = summary_detail[symbol].get('previousClose', '')
                    yesterday_close = safe_float(previousclose_raw, default=0.0, strip_dollar=True)
                    if yesterday_close != 0.0:  # Only round if not default
                        yesterday_close = round(yesterday_close, 2)

                    #52 weeks high
                    week_high_52 = summary_detail[symbol].get('fiftyTwoWeekHigh',0) 
                    if week_high_52 is None or (isinstance(week_high_52, float) and np.isnan(week_high_52)):
                            week_high_52 = 0.0
                    else:
                        week_high_52 = float(week_high_52)
                    # 52 weeks low
                    week_low_52 = summary_detail[symbol].get('fiftyTwoWeekLow', 0)
                    if week_low_52 is None or (isinstance(week_low_52, float) and np.isnan(week_low_52)):
                        week_low_52 = 0.0
                    else:
                        week_low_52 = float(week_low_52)
                    # Volume 
                    #regularMarketTurnover = summary_detail[symbol].get('regularMarketVolume', 0)
                    averageTurnover = summary_detail[symbol].get('averageVolume', 0)
                    if averageTurnover is None or (isinstance(averageTurnover, float) and np.isnan(averageTurnover)):
                            averageTurnover = 0.0
                    else:
                        averageTurnover = float(averageTurnover)

                    averageTurnover10Day = summary_detail[symbol].get('averageVolume10days', 0)
                    if averageTurnover10Day is None or (isinstance(averageTurnover10Day, float) and np.isnan(averageTurnover10Day)):
                            averageTurnover10Day = 0.0
                    else:
                        averageTurnover10Day = float(averageTurnover10Day)


                    # market price
                    symbol_quote = quote.get(symbol, {})
                    # Check if quote is an error response
                    if  isinstance(symbol_quote , str):
                        continue
                    # turnover
                    Turnover = symbol_quote.get('regularMarketVolume', 0) 
                    if Turnover is None or (isinstance(Turnover, float) and np.isnan(Turnover)):
                            Turnover = 0.0
                    else:
                        Turnover = float(Turnover)

                    current_price = symbol_quote.get('regularMarketPrice', 0) 
                    if current_price is None or (isinstance(current_price, float) and np.isnan(current_price)):
                            current_price = 0.0
                    else:
                        current_price = float(current_price)
                    
                    type = symbol_quote.get('quoteType','')
                    # 
                    #company_name = all_stocks[j].get('name', '')
                    company_name=symbol_quote.get('shortName', '')
                    # Store data in memory
                    stock_data.append({
                        'symbol': symbol.upper(),
                        'sector': sector,
                        'industry': industry,
                        'company_name': company_name,
                        'category': category,
                        'type':type,
                        'last_updated': timestamp,
                        'current_price': float(current_price),
                        'yesterday_close': yesterday_close,
                        'week_high_52': week_high_52,
                        'week_low_52': week_low_52,
                        'Turnover': Turnover, 
                        'averageTurnover': averageTurnover, 
                        'averageTurnover10Day': averageTurnover10Day 
                    })
                    #print(f"{symbol} , {company_name} , {type})")
                except Exception as e:
                    error_count += 1
                    warning_log.append(f"Error processing {symbol} in batch {batch_idx + 1}: {str(e)}")
                    print(f"[{datetime.now()}] Error processing {symbol}: {str(e)}")
                    time.sleep(1)  # Avoid rate limits

            # Pause between batches
            time.sleep(1)
        except Exception as e:
            error_count += len(batch)
            warning_log.append(f"Error fetching yahooquery data for batch {batch_idx + 1}: {str(e)}")
            print(f"[{datetime.now()}] Error fetching yahooquery data for batch {batch_idx + 1}: {str(e)}")
            time.sleep(5)  # Longer pause for batch errors

    # Write to database in one batch
    if stock_data:
        try:
            conn = sqlite3.connect('stocks.db')
            c = conn.cursor()
            print(f"[{datetime.now()}] Writing {len(stock_data)} records to stocks.db...")
            c.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    sector TEXT, 
                    industry TEXT,
                    company_name TEXT,
                    category TEXT,
                    type TEXT,
                    last_updated TEXT,
                    current_price REAL,
                    yesterday_close REAL,
                    week_high_52 REAL,
                    week_low_52 REAL, 
                    Turnover REAL, 
                    averageTurnover REAL, 
                    averageTurnover10Day REAL
                )
            ''')
            c.executemany('''
                INSERT OR REPLACE INTO stocks (symbol, sector, industry, company_name, category, type, last_updated, current_price, yesterday_close, week_high_52, week_low_52, Turnover, averageTurnover, averageTurnover10Day)
                VALUES (:symbol, :sector, :industry, :company_name, :category, :type, :last_updated, :current_price, :yesterday_close, :week_high_52, :week_low_52, :Turnover, :averageTurnover, :averageTurnover10Day)
            ''', stock_data)
            conn.commit()
            print(f"[{datetime.now()}] Successfully inserted {len(stock_data)} records into stocks.db")
        except sqlite3.Error as e:
            warning_log.append(f"SQLite error during batch insert: {str(e)}")
            print(f"[{datetime.now()}] SQLite error during batch insert: {str(e)}")
        finally:
            conn.close()
    else:
        print(f"[{datetime.now()}] No valid stock data to insert into database")

    # Summary
    print(f"[{datetime.now()}] Summary: {len(stock_data)} valid stocks inserted, {skipped_count} skipped, {error_count} errors")
    return stock_data, skipped_count, error_count

def get_current_trading_date():
    today = date.today()
    return today

def main():
    warning_log = []
    current_date = get_current_trading_date()
    last_date = get_last_updated()
    if current_date != last_date:
        new_record = True
        print(f"[{datetime.now()}] New trading day detected, initializing and clearing database...")
        init_db()
        clear_db_table(warning_log)
    else:
        new_record = False
        print(f"[{datetime.now()}] Same trading day, updating existing records...")

    start_time = time.time()
    print(f"[{datetime.now()}] Starting stock information update...")

    # Step 1: Fetch and store stock codes in memory
    all_stocks = []
    exchanges = ["nasdaq", "nyse", "amex"]
    for exchange in tqdm(exchanges, desc="Fetching exchanges", unit="exchange"):
        rows = fetch_stocks(exchange, warning_log)
        all_stocks.extend(rows)
        print(f"[{datetime.now()}] Retrieved {len(rows)} stocks from {exchange}")

    print(f"[{datetime.now()}] Total stocks fetched: {len(all_stocks)}")
    if not all_stocks:
        print(f"[{datetime.now()}] No stocks to process. Using fallback symbols.")
        all_stocks = [{'symbol': 'AAPL'}, {'symbol': 'MSFT'}, {'symbol': 'GOOGL'}, {'symbol': 'QCOM'}]

    symbols = [s.upper() for s in [stock.get('symbol', '') for stock in all_stocks] if s and all(c.isalnum() or c in ['.', '-'] for c in s)]
    print(f"[{datetime.now()}] Processing {len(symbols)} valid symbols")
    if not symbols:
        print(f"[{datetime.now()}] No valid symbols. Using fallback symbols for testing.")
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'QCOM']

    # Step 2 & 3: Fetch price data and store in database
    print(f"[{datetime.now()}] Fetching and storing price data from Yahoo Finance...")
    #get_stock_info(symbols, all_stocks, warning_log, new_record)
    fetch_price_data(symbols, all_stocks, warning_log, new_record)
    print(f"[{datetime.now()}] Update complete!")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{datetime.now()}] Total time taken: {elapsed_time:.4f} seconds")
    print(f"[{datetime.now()}] Warning log: {warning_log}")

    # Push to GitHub
    print(f"[{datetime.now()}] Pushing stocks.db to GitHub...")
    os.system("git add stocks.db")
    os.system(f'git commit -m "Update stocks.db at {datetime.now()}"')
    push_result = os.system("git push origin main")
    if push_result == 0:
        print(f"[{datetime.now()}] Successfully pushed stocks.db to GitHub")
    else:
        print(f"[{datetime.now()}] Failed to push stocks.db to GitHub, exit code: {push_result}")

if __name__ == "__main__":
    main()
