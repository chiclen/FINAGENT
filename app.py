import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
import yfinance as yf
import dateutil.parser  # For parsing ISO date string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import subprocess
import os
import pytz  # For US timezone
import sys  # For sys.executable

# Database setup
DB_FILE = 'stocks.db'
No_of_news = 10

# Ensure full-width layout
st.set_page_config(layout="wide")

def get_latest_update_time():
    """Retrieve the latest update time from the Stocks table in US Eastern Time."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(last_updated)
        FROM Stocks
    """)
    result = cursor.fetchone()[0]  # Fetch the result
    conn.close()
    
    if result:
        # Parse ISO format and convert to US Eastern Time
        us_tz = pytz.timezone('America/New_York')
        latest_time = datetime.fromisoformat(result.replace('Z', '+00:00')).astimezone(us_tz)
        return latest_time.strftime("%d-%m-%y %H:%M:%S")
    return "No updates found in the database."
# ------------Index chart
# ——— ADD INDEX OVERLAYS (S&P 500 & Nasdaq) ———
def add_index_overlay(fig, df_main, index_ticker, name, color, visible, period, interval, indexRow):
    try:
        index_data = yf.Ticker(index_ticker).history(period, interval)
        if not index_data.empty:
            # Normalize to % change from first value (for overlay)
            #index_data['Pct_Change'] = (index_data['Close'] / index_data['Close'].iloc[0] - 1) * 100
            
            # Resample to match main df index if needed
            #index_data = index_data.reindex(df_main.index, method='nearest')
            
            fig.add_trace(go.Scatter(
                x=index_data.index,
                y=index_data['Close'],  # Shift to match stock scale approx
                name=name,
                line=dict(color=color, width=2),
                visible=visible  # 'legendonly' or False to hide initially
            ), row=indexRow, col=1)
    except:
        pass  # Silent fail if index data unavailable

# ——————————— ANIMATED GIF CHART FOR MOBILE (Using ChartGif.com) ———————————
def get_animated_gif_chart(symbol, timeframe):
    symbol = symbol.upper().strip()
    url=(f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p={timeframe}&s=l"
    f"&ta_st=rsi,macd"           # ← RSI + MACD stacked under price
    f"&ta0_p=period14"           # RSI period
    f"&ta1_p=12,26,9")            # MACD fast,slow,signal
    return url

# Detect mobile browsers (works for iPhone Safari, Android Chrome, etc.)
def is_mobile():
    user_agent = st.context.headers.get("User-Agent", "").lower()
    mobile_keywords = ["iphone", "ipad", "android", "mobile", "silk", "kindle", "windows phone"]
    return any(keyword in user_agent for keyword in mobile_keywords)
    #return True

def disable_chart_zoom():
    if is_mobile():
        st.markdown(f"""
        <style>
        /* Target only the chart image by adding a unique class */
        .no-zoom-chart img {{
            pointer-events: none !important;
            touch-action: pan-x pan-y !important;
            user-select: none !important;
            -webkit-user-drag: none !important;
            -webkit-touch-callout: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stocks_data():
    """Load all stocks data from the database into a DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT symbol, company_name, category, current_price, yesterday_close,
               week_high_52, week_low_52
        FROM Stocks
        ORDER BY symbol ASC
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Database query failed: {e}")
        conn.close()
        return pd.DataFrame()
    
    conn.close()
    
    if not df.empty:
        try:
            df.columns = [col.strip() for col in df.columns]
            if 'symbol' not in df.columns:
                st.error("Column 'symbol' not found in database. Available columns: " + ", ".join(df.columns))
                return pd.DataFrame()
            
            df['company_name'] = df['company_name']
            invalid_rows = df[df[['current_price', 'week_high_52', 'week_low_52']].isna().any(axis=1)]
            if not invalid_rows.empty:
                df = df.dropna(subset=['current_price', 'week_high_52', 'week_low_52'])
            
            df['pct_to_high'] = df.apply(
                lambda x: (x['current_price'] / x['week_high_52']) * 100 if x['week_high_52'] > 0 else float('inf'),
                axis=1
            )
            df['pct_to_low'] = df.apply(
                lambda x: (x['current_price'] / x['week_low_52']) * 100 if x['week_low_52'] > 0 else float('inf'),
                axis=1
            )
            df['dist_to_high_pct'] = 100 - df['pct_to_high']
            df['dist_to_low_pct'] = df['pct_to_low'] - 100
            df['chart'] = df['symbol']
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return pd.DataFrame()
    else:
        st.warning("No data returned from database query.")
    return df

def load_watchlist_from_db():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT symbol FROM watchlist ORDER BY added_at DESC", conn)
        return df['symbol'].str.upper().tolist()
    except:
        return []
    finally:
        conn.close()

def save_watchlist_to_db(symbols):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM watchlist")  # clear old
    for sym in symbols:
        conn.execute("INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (sym.upper(),))
    conn.commit()
    conn.close()

def add_to_watchlist(symbol):
    symbol = symbol.upper()
    current = load_watchlist_from_db()
    if symbol not in current:
        current.append(symbol)
        save_watchlist_to_db(current)

def remove_from_watchlist(symbol):
    symbol = symbol.upper()
    current = load_watchlist_from_db()
    if symbol in current:
        current.remove(symbol)
        save_watchlist_to_db(current)
        st.rerun()  # refresh immediately

@st.cache_data(ttl=3600)  # Cache for 1 hour due to yfinance rate limits
def fetch_news(symbol):
    if (symbol==""): 
        return None
    try:
        time.sleep(1)  # Avoid rate limits
        stock = yf.Ticker(symbol.upper())
        news = stock.news
        if news:
            news_data = []
            for item in news[:No_of_news]:
                content = item.get("content", {})
                URL = content.get("canonicalUrl", {})
                urladdress = URL.get('url', 'N/A')
                if urladdress == 'N/A' or not urladdress.startswith(('http://', 'https://')):
                    link_html = 'N/A'
                else:
                    link_html = f'<a href="{urladdress}" target="_blank">Read Article</a>'
                pub_date = content.get('pubDate', 'N/A')
                if pub_date and pub_date != 'N/A':
                    try:
                        pub_date = dateutil.parser.isoparse(pub_date).strftime('%Y-%m-%d <br> %H:%M')
                    except (ValueError, TypeError):
                        try:
                            pub_date = datetime.fromtimestamp(int(pub_date)).strftime('%Y-%m-%d </br> %H:%M')
                        except (ValueError, TypeError):
                            pub_date = 'N/A'
                news_data.append({
                    "Published": pub_date,
                    "Title": content.get('title', 'N/A'),
                    "Summary": content.get('summary', 'N/A'),
                    "Link": link_html
                })
            df = pd.DataFrame(news_data)
            st.markdown("""
                <style>
                .news-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .news-table th, .news-table td {
                    padding: 8px;
                    text-align: left;
                    border: 1px solid #ddd;
                    vertical-align: top !important;
                }
                .news-table th:nth-child(1), .news-table td:nth-child(1) { width: 10%; }
                .news-table th:nth-child(2), .news-table td:nth-child(2) { width: 35%; }
                .news-table th:nth-child(3), .news-table td:nth-child(3) { width: 45%; }
                .news-table th:nth-child(4), .news-table td:nth-child(4) { width: 10%; }
                </style>
            """, unsafe_allow_html=True)
            return df
        return None
    except Exception as e:
        return None
    
def fetch_stock_chart(symbol, period1, interval1, index_choice1):
    try:
        #print(f"Interval:{interval1}, Period: {period1}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period1, interval=interval1)
        if df.empty:
            st.error(f"No historical data returned for {symbol}.")
            return None
        
        # ——— Detect mobile ———
        mobile = is_mobile()
        # ——— Always calculate indicators that both versions need ———
        df['MA5']  = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']     = ema12 - ema26
        df['Signal']   = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram']= df['MACD'] - df['Signal']

        # RSI
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        rs    = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ← AVERAGE TURNOVER LINE
        # Auto-choose window based on timeframe
        if interval1 in ["15m", "30m", "60m"]:
            window = 50
        elif period1 in ["5d", "1mo", "3mo"]:
            window = 20
        elif period1 in ["6mo", "1y"]:
            window = 50
        else:  # 2y, 5y, max
            window = 200

        # Safe window: never larger than available non-NaN data
        available_volume = df['Volume'].dropna()
        safe_window = min(window, len(available_volume))

        if safe_window >= 1:
            df['Avg_Volume'] = df['Volume'].rolling(window=safe_window, min_periods=1).mean()
        else:
            df['Avg_Volume'] = None  # no data
         
        # ——————————————— MOBILE VERSION (simple & clean) ———————————————
        epsRow = 1
        indexRow = 2
        symbolRow = 2
        volumeRow = 3
        macdRow = 4
        rsiRow = 5
        if not mobile:
            # —— All the extra indicators you had before (Bollinger, squeeze, signals) ——
            if "None" in index_choice1: 
                fig = make_subplots(
                    rows=5, cols=1,
                    row_heights=[0.15, 0.6, 0.4, 0.20, 0.10],# Price 40%, Volume 30%, MACD 15%, RSI 15%
                    shared_xaxes=True,
                    vertical_spacing=0.03,   
                    subplot_titles=[f"{symbol} Quarterly EPS", "{symbol} Candle Stick", "Volume", "MACD", "RSI"]
                )
            else:

                if "S&P 500" in index_choice1:
                    name = "Index: S&P 500"
                
                if "Nasdaq" in index_choice1:
                    name =  "Index: Nasdaq"

                fig = make_subplots(
                    rows=6, cols=1,
                    row_heights=[0.15, 0.3, 0.6, 0.2, 0.2, 0.2],# index 30% , Price 40%, Volume 20%, MACD 5%, RSI 5%
                    shared_xaxes=True,
                    vertical_spacing=0.03,   
                    subplot_titles=[f"{symbol} Quarterly EPS", name,"Technical Analysis", "Volume", "MACD", "RSI"]
                )

                # index 
                epsRow = 1
                indexRow = 2
                symbolRow = 3
                volumeRow = 4
                macdRow = 5
                rsiRow = 6

                if "S&P 500" in index_choice1:
                    add_index_overlay(fig, df, "^GSPC", "S&P 500", "gray", True, period1, interval1, indexRow)
                if "Nasdaq" in index_choice1:
                    add_index_overlay(fig, df, "^IXIC", "Nasdaq", "purple", True, period1, interval1, indexRow)


            # ─── Synchronized EPS Chart ─────────────────────────────────────
            # Get the exact date range from the main chart's data
            get_quarterly_eps_overlay(fig, symbol, period1, interval1, epsRow)
            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],low=df['Low'], close=df['Close'], name='Candlestick'), row=symbolRow, col=1)

            # Moving Averages
            for ma, col in zip(['MA5','MA20','MA50'], ['orange','green','blue']):
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=col, width=1)), row=symbolRow, col=1)
                # Volume + Average Line

            fig.add_trace(go.Scatter(x=df.index, y=df['Avg_Volume'], name="Avg Volume", line=dict(color='black', width=2)), row=volumeRow, col=1)

            # Volume
            # Create color list: green if up day, red if down day
            colors = ['green' if row['Close'] >= row['Open'] else 'red' 
                    for _, row in df.iterrows()]

            # Add volume bars with matching colors
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors,      # ← THIS IS THE KEY LINE
                marker_line_width=0,
                opacity=0.8
            ), row=volumeRow, col=1)  # or your volume row number

            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],   name='MACD',   line=dict(color='blue')), row=macdRow, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')),   row=macdRow, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color='gray'), row=macdRow, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=rsiRow, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row=rsiRow, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=rsiRow, col=1)

            
            # adding index
            # Then in chart building:

            fig.update_layout(
                title=f"{symbol} • {period1} • {interval1}",
                height=1200,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=80, t=60, b=40),
                template='plotly'
            )
            fig.update_xaxes(rangeslider_visible=False)
            return fig

    except Exception as e:
        st.error(f"Failed to fetch or process chart for {symbol}: {e}")
        return None


def get_quarterly_eps_overlay(fig, symbol, period1, interval1, epsRow):
    """
    Fetches quarterly reported EPS data using yfinance.
    Returns a sorted DataFrame with datetime index and 'EPS' column.
    """
    try:
        ticker = yf.Ticker(symbol)
        main_df = ticker.history(period=period1, interval=interval1)
        main_date_range = (main_df.index.min(), main_df.index.max()) if not main_df.empty else None
        eps_df = main_df

        # Primary method: earnings_dates (includes reported EPS)
        if hasattr(ticker, 'earnings_dates') and ticker.earnings_dates is not None:
            temp = ticker.earnings_dates
            #print (f"temp:{temp}")
            if 'Reported EPS' in temp.columns:
                eps_df = temp[['Reported EPS']].dropna().rename(columns={'Reported EPS': 'EPS'})

        # Fallback: quarterly_earnings (older method)
        if eps_df is None or eps_df.empty:
            if hasattr(ticker, 'quarterly_earnings') and ticker.quarterly_earnings is not None:
                eps_df = ticker.quarterly_earnings[['Earnings']].rename(columns={'Earnings': 'EPS'})

        if eps_df is None or eps_df.empty:
            st.warning(f"No quarterly EPS data available for {symbol}.")
            return pd.DataFrame()
        else:
            # 3. IMPORTANT: Filter EPS to the same period as main chart (if you want control)
            if main_date_range:
                start, end = main_date_range
                eps_df = eps_df[(eps_df.index >= start) & (eps_df.index <= end)]
                #print(f"Filtered EPS rows: {len(eps_df)}")

        # Clean index: remove timezone if present and sort
        if eps_df.index.tz is not None:
            eps_df.index = eps_df.index.tz_localize(None)
        eps_df = eps_df.sort_index()

        if eps_df.empty:
            st.info("No EPS data to display.")
            return None

        #print(f"eps_df:{eps_df}")
        fig.add_trace(
            go.Scatter(
                x=eps_df.index,
                y=eps_df['EPS'],
                mode='lines+markers+text',
                name='Quarterly EPS',
                line=dict(color='#2ca02c', width=4, dash='solid'),
                marker=dict(size=8, symbol='circle', color='white', line=dict(width=3, color='#2ca02c')),
                text=[f"{val:.2f}" for val in eps_df['EPS']],
                textposition="top center",
                textfont=dict(size=8, color="#2ca02c"),
                hovertemplate='%{x|%Y-%m-%d}<br>EPS: %{y:.2f}<extra></extra>'
            ), row = epsRow, col=1
        )
    except Exception as e:
        st.error(f"Failed to fetch company info for {symbol}: {e}")
    return None

@st.cache_data(ttl=7200)
def fetch_company_info(symbol):
    if (symbol== ""):
        return None
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        if info:
            company_data = {
                "Name": info.get("longName", "N/A"),
                "Symbol": info.get("symbol", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": f"${info.get('marketCap', 0):,.0f}" if info.get("marketCap") else "N/A",
                "Website": f'<a href="{info.get("website", "#")}" target="_blank">{info.get("website", "N/A")}</a>' if info.get("website") else "N/A",
                "Description": info.get("longBusinessSummary", "N/A")
            }
            return pd.DataFrame([company_data])
        return None
    except Exception as e:
        st.error(f"Failed to fetch company info for {symbol}: {e}")
        return None

def main():
    # Add CSS for minimizing symbol column and button visibility

    st.session_state.selected_symbol = ""
    # Layout
    st.subheader("Stock Analyzer")

    # Load data
    df = load_stocks_data()
    if df.empty:
        st.warning("No data found in the database or an error occurred. Run DataFeed first!")
        st.info("Click the 'Download Data from yahoo' button above to populate the database.")
        return
    # === FIX COLUMN NAMES ONCE AND FOR ALL ===
    df.columns = df.columns.str.strip()                    # remove any spaces
    df.columns = df.columns.str.replace(' ', '_')          # spaces → underscores
    df.columns = df.columns.str.lower()                    # all lowercase
    
    # Force exact column names we need
    column_mapping = {
        'last_updated': 'last_updated',
        'lastupdated': 'last_updated',
        'lastupdate': 'last_updated',
        'updated': 'last_updated',
        'date': 'last_updated',
        'timestamp': 'last_updated',
    }
    df = df.rename(columns=column_mapping)
    
    # Ensure these columns exist (create if missing)
    for col in ['symbol', 'company_name', 'current_price', 'week_high_52', 
                'week_low_52', 'dist_to_high_pct', 'dist_to_low_pct', 'last_updated']:
        if col not in df.columns:
            df[col] = None  # or appropriate default

    # Column config with progress bars
    column_config = {
        'symbol': st.column_config.TextColumn("Symbol", width=1),
        'current_price': st.column_config.NumberColumn("Price", format="$%.2f", width=1),
        'week_high_52': st.column_config.NumberColumn("52W High", format="$%.2f", width=1),
        'week_low_52': st.column_config.NumberColumn("52W Low", format="$%.2f", width=1),
        'dist_to_high_pct': st.column_config.ProgressColumn(
            "% from High",
            help="Distance to 52-week high as percentage",
            format="%.2f%%",
            min_value=0,
            max_value=100,
            width=5
        ),
        'dist_to_low_pct': st.column_config.ProgressColumn(
            "% from Low",
            help="Distance to 52-week low as percentage",
            format="%.2f%%",
            min_value=0,
            max_value=100,
            width=5
        ),
        'company_name': st.column_config.TextColumn("Name", width=60)
    }

    # Get latest update time
    latest_time = get_latest_update_time()

    # Sidebar for filters (disabled during run)
    st.sidebar.header("Filters")
    threshold = st.sidebar.slider(
        "Closeness Threshold (%)", 0.0, 30.0, 10.0,
        disabled=False
    )
    cap_filter = st.sidebar.multiselect(
        "Filter by Market Cap",
        options=sorted(df['category'].unique()) if 'category' in df.columns else [],
        disabled=False
    )
    view_type = st.sidebar.radio(
        "View Mode", ["Close to High", "Close to Low"],
        disabled=False
    )
    st.sidebar.markdown("---")  # separator
    st.sidebar.header("Watch List")
    new_symbol = st.sidebar.text_input(
        "Add stock to watchlist",
        placeholder="e.g. AAPL, TSLA, NVDA",
        label_visibility="collapsed",
        key="add_stock_input"
    )

    add_btn = st.sidebar.button("Add", width="stretch", type="primary")

    if add_btn:
        if new_symbol:
            symbol = new_symbol.strip().upper()
            if len(symbol) < 1 or len(symbol) > 10:
                st.error("Invalid symbol length")
            elif symbol in st.session_state.my_watchlist:
                st.info(f"{symbol} is already in your watchlist")
            elif symbol not in df['symbol'].str.upper().values:
                st.warning(f"{symbol} not found in database")
            else:
                add_to_watchlist(symbol)
                disabled=False
        else:
            st.error("Please enter a symbol")

    st.markdown("---")  # separator

    # Load watchlist at app start
    st.session_state.my_watchlist = load_watchlist_from_db()

    if st.session_state.my_watchlist:
        st.subheader("Watch list")
        # Filter only symbols that exist in main df
        valid_symbols = [s for s in st.session_state.my_watchlist if s in df['symbol'].str.upper().values]
        if not valid_symbols:
            st.warning("None of your watchlist symbols are in the database.")
            if st.button("Clear invalid watchlist"):
                st.session_state.my_watchlist = []
                save_watchlist_to_db([])
                st.rerun()
        else:
            watch_df = df[df['symbol'].str.upper().isin(valid_symbols)].copy()

            # Ensure all required columns exist
            required_cols = ["symbol",  "current_price", 
                            "dist_to_high_pct", "dist_to_low_pct","company_name"]
            for col in required_cols:
                if col not in watch_df.columns:
                    watch_df[col] = "—"

            # Sort by how close to 52W high
            watch_df = watch_df.sort_values("dist_to_high_pct", ascending=True)

            # Final display columns
            display_cols = ["symbol",  "current_price",
                            "dist_to_high_pct", "dist_to_low_pct", "company_name"]

            st.dataframe(
                watch_df[display_cols],
                hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "dist_to_high_pct": st.column_config.NumberColumn("% from High", format="%.1f%%"),
                    "dist_to_low_pct": st.column_config.NumberColumn("% from Low", format="%.1f%%"),
                    "company_name": st.column_config.TextColumn("Name"),
                },
                on_select="rerun",
                selection_mode="single-row",
                key="watchlist_final"
            )
            symbol = ""
            # Row click → show chart
            selected = st.session_state.get("watchlist_final", {}).get("selection", {})
            if selected and selected.get("rows"):
                idx = selected["rows"][0]
                symbol = watch_df.iloc[idx]["symbol"]
                st.session_state.selected_symbol = symbol    

        # Clear button
        if st.button("Clear Watchlist", type="secondary"):
            if symbol !="": 
                remove_from_watchlist(symbol)
                st.rerun()

    # Stock Info Table
    filtered_df = df[df['category'].isin(cap_filter)] if cap_filter else df
    if filtered_df.empty:
        st.warning(f"No stocks match the selected market caps: {', '.join(cap_filter)}")
        return

    for cap in sorted(cap_filter):
        cap_df = filtered_df[filtered_df['category'] == cap]
        if cap_df.empty:
            st.info(f"No {cap} stocks available.")
            continue

        if view_type == "Close to High":
            close_to_high = cap_df[cap_df['dist_to_high_pct'] <= threshold].copy()
            close_to_high = close_to_high.sort_values('dist_to_high_pct')
            if not close_to_high.empty:
                st.subheader(f"{cap} Stocks Close to 52-Week High ({latest_time})")
                st.subheader(f"(≤ {threshold}%)")
                display_columns = ['symbol', 'current_price', 'week_high_52', 'dist_to_high_pct', 'company_name']
                selected_row = st.dataframe(
                    close_to_high[display_columns],
                    column_config={k: v for k, v in column_config.items() if k in display_columns},
                    key=f"high_{cap}",
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True
                )
                if selected_row.selection.rows:
                    st.session_state.selected_symbol = close_to_high.iloc[selected_row.selection.rows[0]]['symbol']
        elif view_type == "Close to Low":
            close_to_low = cap_df[cap_df['dist_to_low_pct'] <= threshold].copy()
            close_to_low = close_to_low.sort_values('dist_to_low_pct')
            if not close_to_low.empty:
                st.subheader(f"{cap} Stocks Close to 52-Week Low ({latest_time})")
                st.subheader(f"(≤ {threshold}%)")
                display_columns = ['symbol', 'current_price', 'week_low_52', 'dist_to_low_pct', 'company_name']
                selected_row = st.dataframe(
                    close_to_low[display_columns],
                    column_config={k: v for k, v in column_config.items() if k in display_columns},
                    key=f"low_{cap}",
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True
                )
                if selected_row.selection.rows:
                    st.session_state.selected_symbol = close_to_low.iloc[selected_row.selection.rows[0]]['symbol']
        else:  # Both
            close_to_high = cap_df[cap_df['dist_to_high_pct'] <= threshold].copy()
            close_to_low = cap_df[cap_df['dist_to_low_pct'] <= threshold].copy()
            if not close_to_high.empty:
                st.subheader(f"{cap} Stocks Close to 52-Week High ({latest_time})")
                st.subheader(f"(≤ {threshold}%)")
                display_columns = ['symbol', 'current_price', 'week_high_52', 'dist_to_high_pct', 'company_name']
                selected_row = st.dataframe(
                    close_to_high[display_columns],
                    column_config={k: v for k, v in column_config.items() if k in display_columns},
                    key=f"high_{cap}",
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True
                )
                if selected_row.selection.rows:
                    st.session_state.selected_symbol = close_to_high.iloc[selected_row.selection.rows[0]]['symbol']
            if not close_to_low.empty:
                st.subheader(f"{cap} Stocks Close to 52-Week Low ({latest_time})")
                st.subheader(f"(≤ {threshold}%)")
                display_columns = ['symbol', 'current_price', 'week_low_52', 'dist_to_low_pct', 'company_name']
                selected_row = st.dataframe(
                    close_to_low[display_columns],
                    column_config={k: v for k, v in column_config.items() if k in display_columns},
                    key=f"low_{cap}",
                    on_select="rerun",
                    selection_mode="single-row",
                    hide_index=True
                )
                if selected_row.selection.rows:
                    st.session_state.selected_symbol = close_to_low.iloc[selected_row.selection.rows[0]]['symbol']
        
    tabs = st.tabs(["Chart", "News", "Company Info"])
    with tabs[0]:
        if st.session_state.selected_symbol:
            sym = st.session_state.selected_symbol
            if is_mobile():
                period_options = [
                    "Day",
                    "Week",
                    "Month",
                ]

                # Map user-friendly names to yfinance parameters
                period_map = {
                    "Day": ("1y", "1d","d"),
                    "Week": ("5y", "1d","w"),
                    "Month": ("10y", "1d","m")
                }

                if 'chart_period' not in st.session_state:
                    st.session_state.chart_period = "Day"

                selected_period = st.radio(
                    "Time Period",
                    options=period_options,
                    index=1,  # default = "5 Years" (change as needed)
                    horizontal=True,
                    label_visibility="collapsed"  # hides the label above
                )
            else: 
                st.subheader(f"Chart — {st.session_state.selected_symbol}")

                period_options = [
                    "15 Minutes", "30 Minutes", "1 Week", "1 Month", "3 Months",
                    "6 Months", "1 Year", "5 Years", "10 Years", "All Available"
                ]

                # Horizontal buttons — looks great on mobile & desktop
                selected_period = st.radio(
                    "Time Period",
                    options=period_options,
                    index=7,  # default = "5 Years" (change as needed)
                    horizontal=True,
                    label_visibility="collapsed"  # hides the label above
                )

                # Optional: Add custom styling for larger buttons
                st.markdown("""
                <style>
                div.row-widget.stRadio > div {
                    flex-direction: row;
                    gap: 10px;
                }
                div.row-widget.stRadio > div label {
                    background-color: #f0f2f6;
                    padding: 10px 20px;
                    border-radius: 8px;
                    border: 1px solid #d0d7de;
                    font-weight: 500;
                }
                div.row-widget.stRadio > div label[data-checked="true"] {
                    background-color: #1f77b4 !important;
                    color: white !important;
                    border-color: #1f77b4;
                }
                </style>
                """, unsafe_allow_html=True)
                                
                # === NEW: Dropdown for time period ===
                # Map user-friendly names to yfinance parameters
                period_map = {
                    "15 Minutes": ("1d", "15m", "i"),
                    "30 Minutes": ("1d", "30m","i"),
                    "1 Week": ("5d", "1d", "d"),
                    "1 Month":    ("1mo", "1d","d"),
                    "3 Months": ("3mo", "1d","d"),
                    "6 Months": ("6mo", "1d","d"),
                    "1 Year": ("1y", "1d","d"),
                    "5 Years": ("5y", "1d","w"),
                    "10 Years": ("10y", "1d","m"),
                    "All Available": ("max", "1mo","m")  # "max" with monthly for very long history
                }
                # Remember user's last choice
                if 'chart_period' not in st.session_state:
                    st.session_state.chart_period = "5 Years"

                index_choice = st.radio(
                    "Compare With",
                    options=["None", "S&P 500", "Nasdaq"],
                    index=0,
                    horizontal=True
                )

            period = st.session_state.chart_period

            selected_period, selected_interval , chart_index= period_map.get(selected_period)
            # ——————— MOBILE: Show Animated GIF ———————
            if is_mobile():
                st.markdown("### Animated Chart (Mobile View)")
                gif_url = get_animated_gif_chart(sym, chart_index)
                st.markdown(f"""
                <img src="{gif_url}" 
                    style="width:100%; border-radius:8px; pointer-events:none; touch-action:pan-x pan-y; user-select:none;">
                """, unsafe_allow_html=True)
                st.caption("Live animated chart • Double-tap to zoom • Powered by ChartGif.com")
            else:
                # Update session state
                # === Generate chart with selected period ===
                fig = fetch_stock_chart(sym, selected_period, selected_interval,index_choice)

                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.warning("No chart data available for this timeframe.")
    with tabs[1]:    
        df2 = fetch_news(st.session_state.selected_symbol)
        if df2 is not None and not df2.empty:
            st.subheader(f"News for {st.session_state.selected_symbol}")
            st.markdown(df2.to_html(escape=False, index=False, classes="news-table"), unsafe_allow_html=True)
        else:
            st.warning("No news data available.")
    with tabs[2]:     
        df_company = fetch_company_info(st.session_state.selected_symbol)
        if df_company is not None and not df_company.empty:
            st.subheader(f"Company Info for {st.session_state.selected_symbol}")
            st.markdown(df_company[["Name", "Symbol", "Sector", "Industry", "Market Cap", "Website"]].to_html(escape=False, index=False, classes="company-table"), unsafe_allow_html=True)
            st.markdown("**Description**:")
            st.write(df_company["Description"].iloc[0])
        else:
            st.warning("No company info available.")

if __name__ == "__main__":
    main()