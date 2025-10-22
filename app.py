import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, date
import yfinance as yf
from datetime import datetime
import dateutil.parser  # For parsing ISO date string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Database setup
DB_FILE = 'stocks.db'
No_of_news = 10

# Ensure full-width layout
st.set_page_config(layout="wide")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stocks_data():
    """Load all stocks data from the database into a DataFrame."""
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT symbol, category, last_updated, current_price, yesterday_close,
               week_high_52, week_low_52, ma_5, ma_10, ma_20
        FROM Stocks
        ORDER BY symbol ASC
    """
    try:
        df = pd.read_sql_query(query, conn)
        #st.write(f"Debug: Retrieved {len(df)} rows from database.")
    except Exception as e:
        st.error(f"Database query failed: {e}")
        conn.close()
        return pd.DataFrame()
    
    conn.close()
    
    if not df.empty:
        try:
            df.columns = [col.strip() for col in df.columns]
            #st.write("Debug: Available columns in DataFrame:", ", ".join(df.columns))
            if 'symbol' not in df.columns:
                st.error("Column 'symbol' not found in database. Available columns: " + ", ".join(df.columns))
                return pd.DataFrame()
            
            df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
            invalid_rows = df[df[['current_price', 'week_high_52', 'week_low_52']].isna().any(axis=1)]
            if not invalid_rows.empty:
                #st.warning(f"Found {len(invalid_rows)} rows with missing price data. Skipping those rows.")
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
            #st.write("Debug: Category distribution:", df['category'].value_counts().to_dict())
            df['chart'] = df['symbol']
        except Exception as e:
            st.error(f"Error processing data: {e}")
            return pd.DataFrame()
    else:
        st.warning("No data returned from database query.")
    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour due to yfinance rate limits
def fetch_news(symbol):
    # Fetch news
    try:
        time.sleep(1)  # Avoid rate limits
        stock = yf.Ticker(symbol.upper())
        news = stock.news
        if news:
            # Prepare data for table
            news_data = []
            for item in news[:No_of_news]:  # Limit to 20
                content = item.get("content", {})
                URL = content.get("canonicalUrl", {})
                urladdress = URL.get('url', 'N/A')
                # Validate URL
                if urladdress == 'N/A' or not urladdress.startswith(('http://', 'https://')):
                    link_html = 'N/A'  # Non-clickable if invalid
                else:
                    link_html = f'<a href="{urladdress}" target="_blank">Read Article</a>'
                # Handle pubDate (ISO string or timestamp)
                pub_date = content.get('pubDate', 'N/A')
                if pub_date and pub_date != 'N/A':
                    try:
                        # Try parsing as ISO date string
                        pub_date = dateutil.parser.isoparse(pub_date).strftime('%Y-%m-%d <br> %H:%M')
                    except (ValueError, TypeError):
                        # Fallback to timestamp
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
            # Convert to DataFrame
            df = pd.DataFrame(news_data)
            
            # Add custom CSS for table column widths
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
                    vertical-align: top !important; /* Align content to top */
                }
                .news-table th:nth-child(1), .news-table td:nth-child(1) { /* Published */
                    width: 10%;
                }
                .news-table th:nth-child(2), .news-table td:nth-child(2) { /* Title */
                    width: 35%;
                }
                .news-table th:nth-child(3), .news-table td:nth-child(3) { /* Summary */
                    width: 45%;
                }
                .news-table th:nth-child(4), .news-table td:nth-child(4) { /* Link */
                    width: 10%;
                }
                </style>
            """, unsafe_allow_html=True)
            
            return df
        else:
            return None
    except Exception as e:
        #print(f"Debug Error: {str(e)}")
        return None

def fetch_stock_chart(symbol):
    """Fetch 5-year historical data and create a responsive candlestick chart with MA5, MA20, MA50, and volume."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            st.error(f"No historical data returned for {symbol}.")
            return None
        
        #st.write(f"Debug: Fetched {len(df)} rows of historical data for {symbol}.")
        
        # Calculate moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Create subplots: 2 rows (price, volume)
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f"{symbol} 5-Year Stock Price with MA5, MA20, MA50", "Volume"]
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
        
        # Update layout for responsiveness
        fig.update_layout(
            title=f"{symbol} 5-Year Stock Price with Volume",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            autosize=True,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis2=dict(anchor='y2'),
            template='plotly',
            width=None,
            height=None
        )
        
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Failed to fetch or process chart for {symbol}: {e}")
        return None
    
@st.cache_data(ttl=7200)
def fetch_company_info(symbol):
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
        else:
            return None
    except Exception as e:
        st.warning(f"Failed to fetch company info for {symbol}: {e}")
        return None
    
def main():
    # Add CSS for minimizing symbol column
    st.markdown("""
    <style>
    /* Fix the width of the symbol column (first column) */
    .stDataFrame table th:first-child,
    .stDataFrame table td:first-child {
        width: 40px !important;
        min-width: 40px !important;
        max-width: 40px !important;
        white-space: nowrap; /* Prevent text wrapping */
        overflow: hidden; /* Hide overflow */
        text-overflow: ellipsis; /* Show ellipsis for long text */
    }
    /* Ensure consistent behavior on mobile */
    @media (max-width: 768px) {
        .stDataFrame table th:first-child,
        .stDataFrame table td:first-child {
            width: 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
            font-size: 12px; /* Slightly smaller font for mobile */
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None

    # Load data
    df = load_stocks_data()
    if df.empty:
        st.warning("No data found in the database or an error occurred. Run your update script first!")
        return

    # Column config with minimized symbol width
    column_config = {
    'symbol': st.column_config.TextColumn("Symbol", width=30),  # Fixed width of 60px
    'current_price': st.column_config.NumberColumn("Current Price", format="$%.2f", width="small"),
    'week_high_52': st.column_config.NumberColumn("52W High", format="$%.2f", width="small"),
    'week_low_52': st.column_config.NumberColumn("52W Low", format="$%.2f", width="small"),
    'dist_to_high_pct': st.column_config.NumberColumn("% from High", format="%.2f%%", width="small"),
    'dist_to_low_pct': st.column_config.NumberColumn("% from Low", format="%.2f%%", width="small"),
    'last_updated': st.column_config.DatetimeColumn("Last Updated", format="YYYY-MM-DD HH:mm", width="medium"),
    }

    # Create two columns: left for table, right for chart and news
    col1, col2 = st.columns([1, 2], gap="small")  # Adjust ratio as needed

    with col1:
        # Sidebar for filters (unchanged)
        st.sidebar.header("Filters")
        threshold = st.sidebar.slider("Closeness Threshold (%)", 0.0, 30.0, 10.0)
        cap_filter = st.sidebar.multiselect(
            "Filter by Market Cap",
            options=sorted(df['category'].unique()) if 'category' in df.columns else [],
        )
        view_type = st.sidebar.radio("View Mode", ["Close to High", "Close to Low", "Both"])

        # Stock Info Table
        filtered_df = df[df['category'].isin(cap_filter)] if cap_filter else df
        if filtered_df.empty:
            st.warning(f"No stocks match the selected market caps: {', '.join(cap_filter)}")
            return

        column_config = {
            'symbol': st.column_config.TextColumn("Symbol", width="30"),
            'current_price': st.column_config.NumberColumn("Current Price", format="$%.2f", width="small"),
            'week_high_52': st.column_config.NumberColumn("52W High", format="$%.2f", width="small"),
            'week_low_52': st.column_config.NumberColumn("52W Low", format="$%.2f", width="small"),
            'dist_to_high_pct': st.column_config.NumberColumn("% from High", format="%.2f%%", width="small"),
            'dist_to_low_pct': st.column_config.NumberColumn("% from Low", format="%.2f%%", width="small"),
            'last_updated': st.column_config.DatetimeColumn("Last Updated", format="YYYY-MM-DD HH:mm", width="medium"),
        }

        for cap in sorted(cap_filter):
            cap_df = filtered_df[filtered_df['category'] == cap]
            if cap_df.empty:
                st.info(f"No {cap} stocks available.")
                continue

            if view_type == "Close to High":
                close_to_high = cap_df[cap_df['dist_to_high_pct'] <= threshold].copy()
                close_to_high = close_to_high.sort_values('dist_to_high_pct')
                if not close_to_high.empty:
                    st.subheader(f"{cap} Stocks Close to 52-Week High ")
                    st.subheader(f"(≤ {threshold}%)")
                    display_columns = ['symbol', 'current_price', 'week_high_52', 'dist_to_high_pct', 'last_updated']
                    selected_row = st.dataframe(
                        close_to_high[display_columns],
                        use_container_width=True,
                        column_config={k: v for k, v in column_config.items() if k in display_columns},
                        key=f"high_{cap}",
                        on_select="rerun",
                        selection_mode="single-row",
                        hide_index=True  # Hide the index column
                    )
                    if selected_row.selection.rows:
                        st.session_state.selected_symbol = close_to_high.iloc[selected_row.selection.rows[0]]['symbol']
            elif view_type == "Close to Low":
                close_to_low = cap_df[cap_df['dist_to_low_pct'] <= threshold].copy()
                close_to_low = close_to_low.sort_values('dist_to_low_pct')
                if not close_to_low.empty:
                    st.subheader(f"{cap} Stocks Close to 52-Week Low")
                    st.subheader(f"(≤ {threshold}%)")
                    display_columns = ['symbol', 'current_price', 'week_low_52', 'dist_to_low_pct', 'last_updated']
                    selected_row = st.dataframe(
                        close_to_low[display_columns],
                        use_container_width=True,
                        column_config={k: v for k, v in column_config.items() if k in display_columns},
                        key=f"low_{cap}",
                        on_select="rerun",
                        selection_mode="single-row",
                        hide_index=True  # Hide the index column
                    )
                    if selected_row.selection.rows:
                        st.session_state.selected_symbol = close_to_low.iloc[selected_row.selection.rows[0]]['symbol']
            else:  # Both
                close_to_high = cap_df[cap_df['dist_to_high_pct'] <= threshold].copy()
                close_to_low = cap_df[cap_df['dist_to_low_pct'] <= threshold].copy()
                if not close_to_high.empty:
                    st.subheader(f"{cap} Stocks Close to 52-Week High")
                    st.subheader(f"(≤ {threshold}%)")
                    display_columns = ['symbol', 'current_price', 'week_high_52', 'dist_to_high_pct', 'last_updated']
                    selected_row = st.dataframe(
                        close_to_high[display_columns],
                        use_container_width=True,
                        column_config={k: v for k, v in column_config.items() if k in display_columns},
                        key=f"high_{cap}",
                        on_select="rerun",
                        selection_mode="single-row",
                        hide_index=True  # Hide the index column
                    )
                    if selected_row.selection.rows:
                        st.session_state.selected_symbol = close_to_high.iloc[selected_row.selection.rows[0]]['symbol']
                if not close_to_low.empty:
                    st.subheader(f"{cap} Stocks Close to 52-Week Low")
                    st.subheader(f"(≤ {threshold}%)")
                    display_columns = ['symbol', 'current_price', 'week_low_52', 'dist_to_low_pct', 'last_updated']
                    selected_row = st.dataframe(
                        close_to_low[display_columns],
                        use_container_width=True,
                        column_config={k: v for k, v in column_config.items() if k in display_columns},
                        key=f"low_{cap}",
                        on_select="rerun",
                        selection_mode="single-row",
                        hide_index=True  # Hide the index column
                    )
                    if selected_row.selection.rows:
                        st.session_state.selected_symbol = close_to_low.iloc[selected_row.selection.rows[0]]['symbol']

    with col2:
        # Chart and News in Tabs
        if st.session_state.selected_symbol:
            tabs = st.tabs(["Chart", "News", "Company Info"])
            with tabs[0]:
                st.subheader(f"5-Year Chart for {st.session_state.selected_symbol}")
                fig = fetch_stock_chart(st.session_state.selected_symbol)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"No chart data available for {st.session_state.selected_symbol}.")
            with tabs[1]:
                st.subheader(f"News for {st.session_state.selected_symbol}")
                df2 = fetch_news(st.session_state.selected_symbol)
                if df2 is not None and not df2.empty:
                    st.markdown(df2.to_html(escape=False, index=False, classes="news-table"), unsafe_allow_html=True)
                else:
                    st.warning("No news data available.")
            with tabs[2]:
                st.subheader(f"Company Info for {st.session_state.selected_symbol}")
                df_company = fetch_company_info(st.session_state.selected_symbol)
                if df_company is not None and not df_company.empty:
                    # Display key fields as a table
                    st.markdown(df_company[["Name", "Symbol", "Sector", "Industry", "Market Cap", "Website"]].to_html(escape=False, index=False, classes="company-table"), unsafe_allow_html=True)
                    # Display description as text
                    st.markdown("**Description**:")
                    st.write(df_company["Description"].iloc[0])
                else:
                    st.warning("No company info available.")
        else:
            st.info("Select a stock from the table to view its chart, news and company info.")

    # Refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.session_state.selected_symbol = None
        st.rerun()


if __name__ == "__main__":
    main()