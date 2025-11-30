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
    
def fetch_stock_chart(symbol, period1, interval1):
    try:
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

        # ——————————————— MOBILE VERSION (simple & clean) ———————————————
        if mobile:
            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.55, 0.15, 0.15, 0.15],
                shared_xaxes=True,
                vertical_spacing=0.02,
                subplot_titles=(symbol, "Volume", "MACD", "RSI")
            )

            # Candlestick + MAs
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA5'],  name="MA5",  line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color="green")),  row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name="MA50", line=dict(color="blue")),   row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                                 marker_color='rgba(128,128,128,0.5)'), row=2, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],   name="MACD",   line=dict(color="blue")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="red")),   row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name="Hist", marker_color="gray"),         row=3, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="purple")), row=4, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

            fig.update_layout(
                height=700,
                showlegend=False,
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=50, b=10),
                template="plotly_white"
            )
            return fig

        # ——————————————— DESKTOP VERSION (your original full chart) ———————————————
        else:
            # —— All the extra indicators you had before (Bollinger, squeeze, signals) ——
            df['BB_Mid']  = df['Close'].rolling(20).mean()
            df['BB_Std']  = df['Close'].rolling(20).std()
            df['BB_Upper']= df['BB_Mid'] + df['BB_Std']*2
            df['BB_Lower']= df['BB_Mid'] - df['BB_Std']*2
            df['BB_Width']= (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
            df['BB_Squeeze'] = df['BB_Width'] < 0.04
            df['BB_Buy']  = df['Close'] <= df['BB_Lower']
            df['BB_Sell'] = df['Close'] >= df['BB_Upper']

            fig = make_subplots(
                rows=4, cols=1,
                row_heights=[0.5, 0.15, 0.2, 0.15],
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f"{symbol} Technical Analysis", "Volume", "MACD", "RSI"]
            )

            # Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)

            # Moving Averages
            for ma, col in zip(['MA5','MA20','MA50'], ['orange','green','blue']):
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=col, width=1)), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.2)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid', line=dict(color='black')), row=1, col=1)

            # Buy/Sell signals
            buy  = df[df['BB_Buy']]
            sell = df[df['BB_Sell']]
            fig.add_trace(go.Scatter(x=buy.index,  y=buy['Close'],  mode='markers', name='BB Buy',
                                     marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='BB Sell',
                                     marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

            # Squeeze shading
            squeeze = df[df['BB_Squeeze']]
            if not squeeze.empty:
                fig.add_shape(type="rect", x0=squeeze.index.min(), x1=squeeze.index.max(),
                              y0=0, y1=1, fillcolor="yellow", opacity=0.3, layer="below", line_width=0, row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                                 marker_color='rgba(128,128,128,0.5)'), row=2, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],   name='MACD',   line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')),   row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color='gray'), row=3, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=4, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",   row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

            fig.update_layout(
                title=f"{symbol} • {period1} • {interval1}",
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                margin=dict(l=40, r=40, t=60, b=40),
                template='plotly'
            )
            fig.update_xaxes(rangeslider_visible=False)
            return fig

    except Exception as e:
        st.error(f"Failed to fetch or process chart for {symbol}: {e}")
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
    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        script_path = "DataFeedFromYahoo.py"

    # Load data
    df = load_stocks_data()
    if df.empty:
        st.warning("No data found in the database or an error occurred. Run DataFeed first!")
        st.info("Click the 'Download Data from yahoo' button above to populate the database.")
        return

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
        "View Mode", ["Close to High", "Close to Low", "Both"],
        disabled=False
    )

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
            else: 
                # === NEW: Dropdown for time period ===
                period_options = [
                    "15 Minutes",
                    "30 Minutes",
                    "1 Week",
                    "1 Month",
                    "3 Months",
                    "6 Months",
                    "1 Year",
                    "5 Years",
                    "10 Years",
                    "All Available"
                ]

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

            period = st.session_state.chart_period

            selected_period = st.selectbox(
                "Chart Timeframe",
                options=period_options,
                index=period_options.index(st.session_state.chart_period),
                key="period_selector"
            )

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
                fig = fetch_stock_chart(sym, selected_period, selected_interval)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No chart data available for this timeframe.")
        else:
            st.info("Select a stock to view its chart.")
    
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
