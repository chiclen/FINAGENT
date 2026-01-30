import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import dateutil.parser  # For parsing ISO date string


class StockData:
    """
    Reusable class for getting stock news, basic info, and historical data
    Designed to be used both interactively and in machine learning pipelines
    """
    def __init__(self, ticker: str):
        """
        Initialize with a stock ticker symbol
        """
        self.ticker = ticker.upper()
        self._yf = yf.Ticker(self.ticker)           # yfinance object (cached)
        # Internal caches - prevent repeated API calls
        self._news_cache: Optional[List[Dict]] = None
        self._info_cache: Optional[Dict] = None
        self._history_cache: Dict[str, pd.DataFrame] = {}
        
        # Timestamp of last update (for cache invalidation)
        self._last_news_update = None
    
    
    def set_period_and_interval(self, period: str, interval: str) -> None:            
        self._period = period
        self._interval = interval
        
        try:
            self._df = self._yf.history(self._period, self._interval)
        except Exception as e:
            print(f"Error fetching history for {self.ticker}: {e}")
            return pd.DataFrame()

    # ──────────────── Basic Company Information ────────────────
    '''
    @property
    def info(self) -> pd.DataFrame:
        """Get company profile (sector, industry, summary, etc)"""
        if self._info_cache is None:
            try:
                self._info_cache = self._yf.info
                info = self._info_cache
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
            except Exception as e:
                print(f"Error fetching info for {self.ticker}: {e}")
                self._info_cache = {}
                return None
    '''
    @property
    def period(self) -> str:
        return self._period
    
    @property
    def interval(self) -> str:
        return self._interval

    @property
    def company_name(self) -> str:
        return self.info.get('longName', self.ticker)

    @property
    def sector(self) -> str:
        return self.info.get('sector', 'N/A')
    
    @property
    def symbol(self) -> str:
        return self.info.get('symbol', 'N/A')

    @property
    def summary(self) -> str:
        return self.info.get('longBusinessSummary', 'No description available')

    # ──────────────── News ────────────────
    def get_news(self, max_items: int = 10, force_refresh: bool = False) -> List[Dict]:
        """
        Get recent news articles for the stock
        Returns list of dicts with title, publisher, link, time, summary
        """
        # Simple cache invalidation (refresh every 15 minutes)
        should_refresh = (
            self._news_cache is None or
            force_refresh or
            (self._last_news_update and 
             (datetime.now() - self._last_news_update).total_seconds() > 900)
        )

        if should_refresh:
            try:
                raw_news = self._yf.news[:max_items]
                
                self._news_cache = []
                for item in raw_news:
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

                    self._news_cache.append({
                        "Published": pub_date,
                        "Title": content.get('title', 'N/A'),
                        "Summary": content.get('summary', 'N/A'),
                        "Link": link_html
                    })
                
                self._last_news_update = datetime.now()
                
            except Exception as e:
                print(f"Error fetching news for {self.ticker}: {e}")
                self._news_cache = []

        return self._news_cache or []

    # ──────────────── Historical Data (very useful for ML) ────────────────

    def get_history(self, cache: bool = True) -> pd.DataFrame:
        """
        Get OHLCV + adjusted data
        period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        """
        key = f"{self.period}_{self.interval}"
        
        if cache and key in self._history_cache:
            return self._history_cache[key]

        # ——— Always calculate indicators that both versions need ———
        self._df['MA5']  = self._df['Close'].rolling(5).mean()
        self._df['MA20'] = self._df['Close'].rolling(20).mean()
        self._df['MA50'] = self._df['Close'].rolling(50).mean()

        # MACD
        ema12 = self._df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self._df['Close'].ewm(span=26, adjust=False).mean()
        self._df['MACD']     = ema12 - ema26
        self._df['Signal']   = self._df['MACD'].ewm(span=9, adjust=False).mean()
        self._df['Histogram']= self._df['MACD'] - self._df['Signal']

        # RSI
        delta = self._df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = -delta.where(delta < 0, 0).rolling(14).mean()
        rs    = gain / loss
        self._df['RSI'] = 100 - (100 / (1 + rs))

        # ← AVERAGE TURNOVER LINE
        # Auto-choose window based on timeframe
        if self._interval in ["15m", "30m", "60m"]:
            window = 50
        elif self._period in ["5d", "1mo", "3mo"]:
            window = 20
        elif self._period in ["6mo", "1y"]:
            window = 50
        else:  # 2y, 5y, max
            window = 200

        # Safe window: never larger than available non-NaN data
        available_volume = self._df['Volume'].dropna()
        safe_window = min(window, len(available_volume))

        if safe_window >= 1:
            self._df['Avg_Volume'] = self._df['Volume'].rolling(window=safe_window, min_periods=1).mean()
        else:
            self._df['Avg_Volume'] = None  # no data
        return self._df

    def get_EPS(self, cache: bool = True) -> pd.DataFrame:
        main_date_range = (self._df.index.min(), self._df.index.max()) if not self._df.empty else None
        eps_df = self._df
        ticker = self._yf

        # Primary method: earnings_dates (includes reported EPS)
        if hasattr(ticker, 'earnings_dates') and ticker.earnings_dates is not None:
            temp = ticker.earnings_dates
            #print (f"Earning date: {temp}")
            if 'Reported EPS' in temp.columns:
                eps_df = temp[['Reported EPS']].dropna().rename(columns={'Reported EPS': 'EPS'})

        # Fallback: quarterly_earnings (older method)
        if eps_df is None or eps_df.empty:
            if hasattr(ticker, 'quarterly_earnings') and ticker.quarterly_earnings is not None:
                eps_df = ticker.quarterly_earnings[['Earnings']].rename(columns={'Earnings': 'EPS'})

        if eps_df is None or eps_df.empty:
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
            return None
    
        return eps_df
    
    @property
    def info(self) -> dict:
        if self._info_cache is None:
            try:
                raw_info = self._yf.info
                print(f"[DEBUG] Type of {self.ticker}.info: {type(raw_info)}")
                if raw_info is None:
                    print(f"[DEBUG] {self.ticker}.info is None")
                    raw_info = {}
                elif not isinstance(raw_info, dict):
                    print(f"[DEBUG] {self.ticker}.info is unexpected type: {type(raw_info)}")
                    raw_info = {}
                self._info_cache = raw_info
            except Exception as e:
                print(f"[ERROR] Failed to fetch .info for {self.ticker}: {e}")
                self._info_cache = {}
        return self._info_cache
    # ──────────────── Quick ML-friendly feature extractor ────────────────
# ─────────────────────────────────────────────────────────────────────
#                               Usage Examples
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create instance for one stock
    company = StockData("PLTR")

    print(f"Company: {company.company_name}")
    print(f"Sector : {company.sector}")
    print(f"Code : {company.symbol}")
    print(f"Summary: {company.summary[:120]}...")

    
    try:
        info = company.info
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
            #df_company = pd.DataFrame([company_data])
            #print (df_company)        
    except Exception as e:
        print(f"Failed to fetch company info for {company.ticker}: {e}")

