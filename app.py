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

def fetch_stock_chart(symbol):
    """Fetch 5-year historical data and create a responsive candlestick chart with MA5, MA20, MA50, volume, RSI, MACD, Bollinger Bands, and squeeze signals."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            st.error(f"No historical data returned for {symbol}.")
            return None
        
        # Calculate moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands (20-period SMA with 2 std devs)
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
        
        # Bollinger Band Squeeze signal: Bandwidth < threshold (e.g., 4% or rolling min)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
        df['BB_Squeeze'] = df['BB_Width'] < 0.04  # Threshold for squeeze; adjust based on asset (or use df['BB_Width'].rolling(125).min())
        
        # Detect Bollinger Bands buy/sell (touch lower/upper)
        df['BB_Buy'] = (df['Close'] <= df['BB_Lower'])
        df['BB_Sell'] = (df['Close'] >= df['BB_Upper'])
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Histogram'] = df['MACD'] - df['Signal']
        
        # Create subplots: 4 rows (price + Bollinger, volume, MACD, RSI)
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f"{symbol} 5-Year Stock Price with MAs & Bollinger Bands", "Volume", "MACD", "RSI"]
        )
        
        # Candlestick (row 1)
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ),
            row=1, col=1
        )
        
        # Moving Averages (row 1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='green', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Bollinger Bands overlay (row 1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.2)'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid', line=dict(color='black', width=1)),
            row=1, col=1
        )
        
        # Bollinger Bands signals (buy/sell markers on price chart)
        buy_signals = df[df['BB_Buy']]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='BB Buy Signal',
                marker=dict(symbol='triangle-up', size=10, color='green', line=dict(width=1, color='darkgreen'))
            ),
            row=1, col=1
        )
        
        sell_signals = df[df['BB_Sell']]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='BB Sell Signal',
                marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='darkred'))
            ),
            row=1, col=1
        )
        
        # Bollinger Squeeze signals (shaded areas on price chart during squeeze)
        squeeze_periods = df[df['BB_Squeeze']]
        if not squeeze_periods.empty:
            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=squeeze_periods.index.min(),
                x1=squeeze_periods.index.max(),
                y0=0,
                y1=1,
                fillcolor="yellow",
                opacity=0.3,
                layer="below",
                line=dict(width=0),
                row=1, col=1
            )
            fig.add_annotation(
                text="Squeeze",
                x=squeeze_periods.index.mean(),
                y=0.95,
                yref="paper",
                showarrow=False,
                font=dict(color="black", size=12),
                row=1, col=1
            )
        
        # Volume bars (row 2)
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(128, 128, 128, 0.5)'
            ),
            row=2, col=1
        )
        
        # MACD (row 3)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color='gray'),
            row=3, col=1
        )
        
        # RSI (row 4)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)),
            row=4, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", row=4, col=1)
        
        # Update layout for responsiveness (no range slider on RSI)
        fig.update_layout(
            title=f"{symbol} 5-Year Stock Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2_title="Volume",
            yaxis3_title="MACD",
            yaxis4_title="RSI",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            template='plotly',
            width=None,
            height=800  # Increased height for additional subplots
        )
        
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=4, col=1)  # Removed slider under RSI
        
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
        if (st.session_state.selected_symbol!=""):
            st.subheader(f"5-Year Chart for {st.session_state.selected_symbol}")
            fig = fetch_stock_chart(st.session_state.selected_symbol)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.warning(f"No chart data available for {st.session_state.selected_symbol}.")
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
