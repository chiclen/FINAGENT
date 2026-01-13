import streamlit as st
import pandas as pd
from finvizfinance.screener.overview import Overview
from io import BytesIO

# ────────────────────────────────────────────────
# App Title & Description
# ────────────────────────────────────────────────
st.title("Customizable Finviz Stock Screener")
st.markdown("""
Change filters below → results update automatically.  
""")

# ────────────────────────────────────────────────
# Sidebar: Filter Controls
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    # exchange
    exchange_options = ['Any', 'AMEX', 'NASDAQ', 'NYSE']
    exchange = st.selectbox("Exchange", exchange_options, index=exchange_options.index('Any') )
    # market cap
    market_cap_options = [
        'Any', 'Mega ($200bln and more)', 'Large ($10bln to $200bln)',
        'Mid ($2bln to $10bln)', 'Small ($300mln to $2bln)', 'Micro ($50mln to $300mln)',
        'Nano (under $50mln)', '+Large (over $10bln)', '+Mid (over $2bln)',
        '+Small (over $300mln)', '+Micro (over $50mln)'
    ]
 
    market_cap = st.selectbox("Market Cap", market_cap_options, index=market_cap_options.index('+Small (over $300mln)'))
    # country
    country_option = ['Any', 'USA', 'Foreign (ex-USA)', 'Asia', 'Europe', 
                      'Latin America', 'BRIC', 'Argentina', 'Australia', 
                      'Bahamas', 'Belgium', 'BeNeLux', 'Bermuda', 'Brazil', 
                      'Canada', 'Cayman Islands', 'Chile', 'China', 'China & Hong Kong',
                      'Colombia', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany', 
                      'Greece', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 
                      'Ireland', 'Israel', 'Italy', 'Japan', 'Kazakhstan', 'Luxembourg', 
                      'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Netherlands', 'New Zealand', 
                      'Norway', 'Panama', 'Peru', 'Philippines', 'Portugal', 'Russia', 
                      'Singapore', 'South Africa', 'South Korea', 'Spain', 'Sweden', 
                      'Switzerland', 'Taiwan', 'Turkey', 'United Arab Emirates', 
                      'United Kingdom', 'Uruguay']
    country = st.selectbox("Country", country_option, index=country_option.index('Any'))
    # EPS
    eps_option = ['Any', 'Negative (<0%)', 'Positive (>0%)', 'Positive Low (0-10%)',
                 'High (>25%)', 'Under 5%', 'Under 10%', 'Under 15%', 
                 'Under 20%', 'Under 25%', 'Under 30%', 'Over 5%', 
                 'Over 10%', 'Over 15%', 'Over 20%', 'Over 25%', 'Over 30%']    
    EPS_this_yr = st.selectbox("EPS growththis year", eps_option, index=eps_option.index('Any'))
    EPS_next_yr = st.selectbox("EPS growthnext year", eps_option, index=eps_option.index('Any'))
    EPS_past_5_yr = st.selectbox("EPS growthpast 5 years", eps_option, index=eps_option.index('Any'))
    EPS_next_5_yr = st.selectbox("EPS growthnext 5 years", eps_option, index=eps_option.index('Any'))
    EPS_qtr_over_qtr = st.selectbox("EPS growthqtr over qtr", eps_option, index=eps_option.index('Any'))


   
    

    avg_vol_options = ['Any', 'Under 50K', 'Over 50K', 'Over 100K', 'Over 300K', 'Over 500K', 'Over 750K', 'Over 1M']
    avg_volume = st.selectbox("Average Volume", avg_vol_options, index=avg_vol_options.index('Over 300K'))

    change_options = ['Any', 'Up 5%', 'Up 8%', 'Up 10%', 'Up 20%', 'Down 5%', 'Down 10%']
    change = st.selectbox("Daily Change", change_options, index=change_options.index('Up 8%'))

    # Technical Filters (SMA)  # SMA 50 option
    sma50_options = ['Any', 'Price below SMA50', 'Price 10% below SMA50', 'Price 20% below SMA50', 
                    'Price 30% below SMA50', 'Price 40% below SMA50', 'Price 50% below SMA50', 
                    'Price above SMA50', 'Price 10% above SMA50', 'Price 20% above SMA50', 
                    'Price 30% above SMA50', 'Price 40% above SMA50', 'Price 50% above SMA50', 
                    'Price crossed SMA50', 'Price crossed SMA50 above', 'Price crossed SMA50 below', 
                    'SMA50 crossed SMA20', 'SMA50 crossed SMA20 above', 'SMA50 crossed SMA20 below', 
                    'SMA50 crossed SMA200', 'SMA50 crossed SMA200 above', 'SMA50 crossed SMA200 below', 
                    'SMA50 above SMA20', 'SMA50 below SMA20', 'SMA50 above SMA200', 'SMA50 below SMA200']
    
    sma50 = st.selectbox("50-Day Simple Moving Average", sma50_options, index=sma50_options.index('Price above SMA50'))
    # SMA 200 
    sma200_options = ['Any', 'Price below SMA200', 'Price 10% below SMA200', 'Price 20% below SMA200', 
                    'Price 30% below SMA200', 'Price 40% below SMA200', 'Price 50% below SMA200', 
                    'Price 60% below SMA200', 'Price 70% below SMA200', 'Price 80% below SMA200', 
                    'Price 90% below SMA200', 'Price above SMA200', 'Price 10% above SMA200', 
                    'Price 20% above SMA200', 'Price 30% above SMA200', 'Price 40% above SMA200', 
                    'Price 50% above SMA200', 'Price 60% above SMA200', 'Price 70% above SMA200', 
                    'Price 80% above SMA200', 'Price 90% above SMA200', 'Price 100% above SMA200', 
                    'Price crossed SMA200', 'Price crossed SMA200 above', 'Price crossed SMA200 below', 
                    'SMA200 crossed SMA20', 'SMA200 crossed SMA20 above', 'SMA200 crossed SMA20 below', 
                    'SMA200 crossed SMA50', 'SMA200 crossed SMA50 above', 'SMA200 crossed SMA50 below', 
                    'SMA200 above SMA20', 'SMA200 below SMA20', 'SMA200 above SMA50', 'SMA200 below SMA50']
    
    sma200 = st.selectbox("200-Day Simple Moving Average", sma200_options, index=sma200_options.index('Price above SMA200'))
    # change
    change_options = ['Any', 'Up', 'Up 1%', 'Up 2%', 'Up 3%', 'Up 4%', 'Up 5%', 
                      'Up 6%', 'Up 7%', 'Up 8%', 'Up 9%', 'Up 10%', 'Up 15%', 
                      'Up 20%', 'Down', 'Down 1%', 'Down 2%', 'Down 3%', 
                      'Down 4%', 'Down 5%', 'Down 6%', 'Down 7%', 'Down 8%', 
                      'Down 9%', 'Down 10%', 'Down 15%', 'Down 20%']
    
    change = st.selectbox("Change", change_options, index=change_options.index('Up 8%'))
    # Average volume
    average_vol_options = ['Any', 'Up', 'Up 1%', 'Up 2%', 'Up 3%', 'Up 4%', 'Up 5%', 
                      'Up 6%', 'Up 7%', 'Up 8%', 'Up 9%', 'Up 10%', 'Up 15%', 
                      'Up 20%', 'Down', 'Down 1%', 'Down 2%', 'Down 3%', 
                      'Down 4%', 'Down 5%', 'Down 6%', 'Down 7%', 'Down 8%', 
                      'Down 9%', 'Down 10%', 'Down 15%', 'Down 20%']
    
    average_volume = st.selectbox("Average Volume", average_vol_options, index=average_vol_options.index('Up 8%'))
    # Price
    price_options = ['Any', 'Under $1', 'Under $2', 'Under $3', 'Under $4', 
                     'Under $5', 'Under $7', 'Under $10', 'Under $15', 'Under $20', 
                     'Under $30', 'Under $40', 'Under $50', 'Over $1', 'Over $2', 
                     'Over $3', 'Over $4', 'Over $5', 'Over $7','Over $10', 'Over $15', 
                     'Over $20', 'Over $30', 'Over $40', 'Over $50', 'Over $60', 'Over $70', 
                     'Over $80', 'Over $90', 'Over $100', '$1 to $5', '$1 to $10', '$1 to $20', 
                     '$5 to $10', '$5 to $20', '$5 to $50', '$10 to $20','$10 to $50', '$20 to $50', '$50 to $100']
    
    price = st.selectbox("Price", price_options, index=price_options.index('Over $2'))
    # Sector & Industry (multi-select for flexibility)
    sector = st.multiselect("Sector", ['Any', 'Basic Materials', 'Communication Services', 'Consumer Cyclical',
                                       'Consumer Defensive', 'Energy', 'Financial', 'Healthcare',
                                       'Industrials', 'Real Estate', 'Technology', 'Utilities'],
                            default=['Any'])
    # Other common ones
    country = st.selectbox("Country", ['Any', 'USA', 'Canada', 'China', 'United Kingdom'], index=0)

# ────────────────────────────────────────────────
# Build filters_dict from user selections
# ────────────────────────────────────────────────
filters_dict = {}

if market_cap != 'Any':
    filters_dict['Market Cap.'] = market_cap

if price != 'Any':
    filters_dict['Price'] = price

if avg_volume != 'Any':
    filters_dict['Average Volume'] = avg_volume

if change != 'Any':
    filters_dict['Change'] = change

if sma50 != 'Any':
    filters_dict['50-Day Simple Moving Average'] = sma50.replace('Relation', '').strip()  # clean up
if sma200 != 'Any':
    filters_dict['200-Day Simple Moving Average'] = sma200.replace('Relation', '').strip()

if 'Any' not in sector:
    filters_dict['Sector'] = ', '.join(sector)  # finvizfinance usually takes one, but try comma-separated

if country != 'Any':
    filters_dict['Country'] = country

# ────────────────────────────────────────────────
# Fetch & Display Results
# ────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache 5 min to reduce requests
def fetch_stocks(filters):
    try:
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        df = foverview.screener_view()
        return df
    except Exception as e:
        st.error(f"Error applying filters: {e}\nTry fewer/more compatible filters.")
        return pd.DataFrame()

df = fetch_stocks(filters_dict)

if not df.empty:
    st.success(f"Found **{len(df)}** stocks matching your filters.")

    # Show interactive table
    st.dataframe(
        df.style.format(precision=2),
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Change": st.column_config.NumberColumn("Change", format="%.2f%%"),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        }
    )

    # Export
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Scanner Results')
        return output.getvalue()

    excel_data = to_excel(df)
    st.download_button(
        label="📥 Export to Excel",
        data=excel_data,
        file_name="finviz_custom_scanner.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("No results found. Try loosening some filters (e.g., remove Change or SMA conditions).")

st.caption("Note: Some filter combinations may return 0 results or cause library errors — Finviz is strict. Data as of real-time scrape.")