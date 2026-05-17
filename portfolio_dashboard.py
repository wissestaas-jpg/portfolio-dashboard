import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, date
import os

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "Transactions.csv")
ACCOUNT_FILE = os.path.join(DATA_DIR, "Account.csv")

# ── DEGIRO already records splits as buy/sell transactions, no manual adjustment needed ──

ISIN_TO_TICKER = {
    "US67066G1040": "NVDA",
    "US88160R1014": "TSLA",
    "US0231351067": "AMZN",
    "US0378331005": "AAPL",
    "NL0010273215": "ASML.AS",
    "NL0012866412": "BESI.AS",
    "NL0000334118": "ASM.AS",
    "NL0000852523": "TWEKA.AS",
    "NL0015000GX8": "ENVI.AS",
    "DE000A28M8D0": "VBTC.DE",
    "US62914V1061": "NIO",
    "CA1380357048": "WEED.TO",
    "CA1380351009": "WEED.TO",
    "US86260J1025": "STRN",
    "NL0009805522": "NBIS",
    "BE0003818359": "GLPG.AS",
    "US00165C1045": "AMC",
    "NL0013332471": "TOM2.AS",
    "US6549022043": "NOK",
    "NL0000852580": "BOKA.AS",
    "US0079031078": "AMD",
    "NL0000440584": "ORDI.AS",
    "US18914F1030": "CLOV",
    "CA98980M1095": "ZOM",
    "US47010C4096": "JAGX",
    "US1714391026": "LCID",
    "KYG8251K1076": "IPOC",
    "NL0009767532": "ACCEL.AS",
}


def parse_dutch(s):
    if pd.isna(s) or str(s).strip() == "":
        return 0.0
    try:
        return float(str(s).replace(".", "").replace(",", "."))
    except ValueError:
        return 0.0


# Manual overrides for tickers where yfinance returns no sector/country
TICKER_META_OVERRIDE = {
    "VBTC.DE": {"sector": "Crypto Assets", "country": "Global", "beta": 1.8},
}

SECTOR_COLORS = {
    "Technology":             "#2E86AB",
    "Consumer Cyclical":      "#A23B72",
    "Communication Services": "#F18F01",
    "Industrials":            "#4CAF50",
    "Crypto Assets":          "#FF9800",
    "Healthcare":             "#E91E63",
    "Financial Services":     "#795548",
    "Energy":                 "#FF5722",
    "Utilities":              "#9C27B0",
    "Real Estate":            "#607D8B",
    "Basic Materials":        "#8BC34A",
    "Cash":                   "#44BBA4",
    "Other":                  "#90A4AE",
}


# ── Market data ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_exchange_rate():
    hist = yf.Ticker("EURUSD=X").history(period="5d")
    return hist["Close"].iloc[-1] if not hist.empty else 1.08


@st.cache_data(ttl=300)
def get_current_price(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return hist["Close"].iloc[-1] if not hist.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_historical_prices(ticker, start_date=None):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date, period="max" if start_date else "max")
        return hist["Close"] if not hist.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=86400)
def get_ticker_info(ticker):
    """Fetch sector, country and beta from yfinance (cached 24h)."""
    if ticker in TICKER_META_OVERRIDE:
        return TICKER_META_OVERRIDE[ticker]
    try:
        info = yf.Ticker(ticker).info
        return {
            "sector":  info.get("sector") or "Other",
            "country": info.get("country") or "Unknown",
            "beta":    info.get("beta"),
        }
    except Exception:
        return {"sector": "Other", "country": "Unknown", "beta": None}


def compute_risk(portfolio_df, total_hist, cash_eur, total_value):
    """
    Returns a dict with:
      - sector_weights, country_weights, currency_weights  (for charts)
      - beta, volatility, max_drawdown, sharpe  (scalar metrics)
      - hhi  (concentration index 0-1)
    """
    # ── Sector / country / currency breakdown ────────────────────────────────
    sector_vals   = {}
    country_vals  = {}
    currency_vals = {}

    for _, row in portfolio_df.iterrows():
        ticker = row["Ticker"]
        mv     = row["Market Value (EUR)"]
        curr   = row["Currency"]
        meta   = get_ticker_info(ticker)

        sector  = meta.get("sector", "Other") or "Other"
        country = meta.get("country", "Unknown") or "Unknown"

        sector_vals[sector]   = sector_vals.get(sector, 0) + mv
        country_vals[country] = country_vals.get(country, 0) + mv
        currency_vals[curr]   = currency_vals.get(curr, 0) + mv

    # Add cash
    if cash_eur > 0:
        sector_vals["Cash"]   = sector_vals.get("Cash", 0) + cash_eur
        country_vals["Cash"]  = country_vals.get("Cash", 0) + cash_eur
        currency_vals["EUR"]  = currency_vals.get("EUR", 0) + cash_eur

    def to_pct(d):
        total = sum(d.values())
        return {k: v / total * 100 for k, v in sorted(d.items(), key=lambda x: -x[1])}

    sector_pct   = to_pct(sector_vals)
    country_pct  = to_pct(country_vals)
    currency_pct = to_pct(currency_vals)

    # ── Weighted beta ────────────────────────────────────────────────────────
    beta_sum = 0.0
    beta_weight = 0.0
    for _, row in portfolio_df.iterrows():
        meta = get_ticker_info(row["Ticker"])
        b = meta.get("beta")
        if b is not None:
            w = row["Market Value (EUR)"] / total_value
            beta_sum    += b * w
            beta_weight += w
    weighted_beta = beta_sum / beta_weight if beta_weight > 0 else None

    # ── Volatility & Sharpe (1Y daily returns) ───────────────────────────────
    vol_1y = sharpe = None
    if not total_hist.empty and len(total_hist) > 60:
        hist_1y = total_hist.last("252D")
        rets = hist_1y.pct_change().dropna()
        if len(rets) > 20:
            vol_1y  = rets.std() * (252 ** 0.5) * 100   # annualised %
            ann_ret = ((hist_1y.iloc[-1] / hist_1y.iloc[0]) ** (252 / len(hist_1y)) - 1) * 100
            rf      = 2.5  # risk-free rate %
            sharpe  = (ann_ret - rf) / vol_1y if vol_1y > 0 else None

    # ── Max drawdown (all-time) ───────────────────────────────────────────────
    max_dd = None
    if not total_hist.empty:
        roll_max = total_hist.cummax()
        dd = (total_hist - roll_max) / roll_max * 100
        max_dd = dd.min()

    # ── HHI concentration (stocks only, 0=diversified → 1=one stock) ─────────
    weights = portfolio_df["Market Value (EUR)"] / total_value
    hhi = (weights ** 2).sum()

    return {
        "sector_pct":   sector_pct,
        "country_pct":  country_pct,
        "currency_pct": currency_pct,
        "beta":         weighted_beta,
        "volatility":   vol_1y,
        "max_drawdown": max_dd,
        "sharpe":       sharpe,
        "hhi":          hhi,
        "top_weight":   weights.max() * 100,
        "top_name":     portfolio_df.loc[weights.idxmax(), "Product"] if not portfolio_df.empty else "",
    }


# ── Account / transaction parsing ───────────────────────────────────────────

def load_transactions(filepath):
    df = pd.read_csv(filepath, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    df["Aantal_parsed"] = df["Aantal"].apply(
        lambda x: parse_dutch(x) if not isinstance(x, (int, float)) else float(x)
    )
    df["Totaal_EUR_parsed"] = df["Totaal EUR"].apply(parse_dutch) if "Totaal EUR" in df.columns else 0
    return df


def parse_account_csv(filepath):
    """Returns (deposits_df, eur_balance_series, net_deposited_eur).
    filepath can be a path string or a file-like object (BytesIO)."""
    if filepath is None:
        return pd.DataFrame(), pd.Series(dtype=float), 0.0
    if isinstance(filepath, str) and not os.path.exists(filepath):
        return pd.DataFrame(), pd.Series(dtype=float), 0.0

    df = pd.read_csv(filepath, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    df["Amount"] = df["Unnamed: 8"].apply(parse_dutch)
    df["Currency"] = df["Mutatie"].fillna("")
    df["Balance"] = df["Unnamed: 10"].apply(parse_dutch)
    df["BalanceCurr"] = df["Saldo"].fillna("")
    df = df.sort_values("Datum")

    # External deposits: real money wired in from outside
    deposit_patterns = ["ideal storting", "ideal deposit", "flatex storting",
                        "storting", "iDEAL Deposit"]
    def is_deposit(row):
        desc = str(row["Omschrijving"]).lower()
        return (any(p.lower() in desc for p in deposit_patterns)
                and "reservation" not in desc
                and row["Amount"] > 0
                and row["Currency"] == "EUR")

    # External withdrawals
    def is_withdrawal(row):
        desc = str(row["Omschrijving"]).lower()
        return ("processed flatex withdrawal" in desc and row["Amount"] < 0)

    dep_mask = df.apply(is_deposit, axis=1)
    wd_mask = df.apply(is_withdrawal, axis=1)

    deposits_df = df[dep_mask][["Datum", "Amount", "Omschrijving"]].copy()
    deposits_df = deposits_df.rename(columns={"Amount": "Deposit"})

    withdrawals_total = abs(df[wd_mask]["Amount"].sum())
    deposits_total = deposits_df["Deposit"].sum()
    net_deposited = deposits_total - withdrawals_total

    # EUR cash balance history from Saldo column
    eur_rows = df[df["BalanceCurr"] == "EUR"][["Datum", "Balance"]].copy()
    eur_rows = eur_rows.sort_values("Datum")
    # Take last entry per day
    eur_daily = eur_rows.groupby("Datum")["Balance"].last()

    # Build a full daily series from first date to today
    if not eur_daily.empty:
        full_idx = pd.date_range(eur_daily.index.min(), date.today(), freq="D")
        eur_balance = eur_daily.reindex(full_idx).ffill()
    else:
        eur_balance = pd.Series(dtype=float)

    return deposits_df, eur_balance, net_deposited


def compute_holdings(df):
    holdings = {}
    for _, row in df.sort_values("Datum").iterrows():
        isin = row["ISIN"]
        qty = row["Aantal_parsed"]
        cost_eur = row["Totaal_EUR_parsed"]

        if isin not in holdings:
            holdings[isin] = {
                "product": row["Product"],
                "isin": isin,
                "shares": 0,
                "total_bought": 0.0,
                "total_bought_shares": 0.0,
                "total_sold_proceeds": 0.0,
            }

        holdings[isin]["shares"] += qty

        # Corporate actions (splits, mergers) have no Order ID
        oid_a = row.get("Order ID", None)
        oid_b = row.get("Unnamed: 17", None)
        has_order_id = (not pd.isna(oid_a) and str(oid_a).strip() != "") or \
                       (not pd.isna(oid_b) and str(oid_b).strip() != "")

        if has_order_id:
            if qty < 0:
                holdings[isin]["total_sold_proceeds"] += cost_eur
            else:
                holdings[isin]["total_bought"] += abs(cost_eur)
                holdings[isin]["total_bought_shares"] += qty

        holdings[isin]["product"] = row["Product"]

    return {k: v for k, v in holdings.items() if abs(v["shares"]) > 0}


def get_portfolio_value(holdings, eur_usd_rate):
    rows = []
    for isin, h in holdings.items():
        ticker = ISIN_TO_TICKER.get(isin)
        if not ticker:
            continue
        price = get_current_price(ticker)
        if price is None:
            continue

        currency = "EUR" if ticker.endswith((".AS", ".DE")) else "USD"
        price_eur = price if currency == "EUR" else price / eur_usd_rate
        market_value = h["shares"] * price_eur

        if h["total_bought_shares"] > 0 and h["shares"] < h["total_bought_shares"]:
            cost_basis = (h["total_bought"] / h["total_bought_shares"]) * h["shares"]
        else:
            cost_basis = h["total_bought"]

        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

        rows.append({
            "Product": h["product"],
            "ISIN": isin,
            "Ticker": ticker,
            "Shares": h["shares"],
            "Price": price,
            "Currency": currency,
            "Price (EUR)": price_eur,
            "Market Value (EUR)": market_value,
            "Cost Basis (EUR)": cost_basis,
            "P&L (EUR)": pnl,
            "P&L (%)": pnl_pct,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def compute_stock_history(holdings_frozen, start_date_str):
    """Compute historical stock portfolio value from start_date."""
    start_date = pd.Timestamp(start_date_str)
    all_series = {}

    fx_hist = get_historical_prices("EURUSD=X", start_date=start_date_str)

    for isin, h in holdings_frozen:
        ticker = ISIN_TO_TICKER.get(isin)
        if not ticker or h["shares"] == 0:
            continue
        hist = get_historical_prices(ticker, start_date=start_date_str)
        if hist.empty:
            continue

        # Strip timezone
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index

        currency = "EUR" if ticker.endswith((".AS", ".DE")) else "USD"
        if currency == "USD" and not fx_hist.empty:
            fx = fx_hist.copy()
            fx.index = fx.index.tz_localize(None) if fx.index.tz else fx.index
            combined = pd.DataFrame({"price": hist, "fx": fx})
            combined = combined.ffill().dropna()
            hist = combined["price"] / combined["fx"]

        all_series[isin] = hist * h["shares"]

    if not all_series:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_series)
    df.index = pd.to_datetime(df.index).normalize()
    df = df.ffill().bfill()
    return df.sum(axis=1)


def v(amount, blur, fmt="€{:,.2f}"):
    """Format a euro value, or return blurred placeholder."""
    return "€ ••••" if blur else fmt.format(amount)


def main():
    import io
    st.title("Portfolio Dashboard")

    # ── Session state init ────────────────────────────────────────────────────
    if "tx_bytes" not in st.session_state:
        st.session_state.tx_bytes   = None
    if "acct_bytes" not in st.session_state:
        st.session_state.acct_bytes = None

    # Auto-load from local files the first time (local runs only).
    # We read into bytes so the rest of the code is identical whether
    # files came from disk or from an upload widget.
    if st.session_state.tx_bytes is None and os.path.exists(TRANSACTIONS_FILE):
        with open(TRANSACTIONS_FILE, "rb") as f:
            st.session_state.tx_bytes = f.read()
    if st.session_state.acct_bytes is None and os.path.exists(ACCOUNT_FILE):
        with open(ACCOUNT_FILE, "rb") as f:
            st.session_state.acct_bytes = f.read()

    # ── Upload UI ─────────────────────────────────────────────────────────────
    # Keep expander open until Transactions.csv is loaded
    with st.expander("📂 Upload your DEGIRO exports",
                     expanded=st.session_state.tx_bytes is None):
        st.caption("Download these from DEGIRO → Portfolio/Account → Export")
        col1, col2 = st.columns(2)
        with col1:
            up_tx = st.file_uploader("Transactions.csv (required)", type="csv", key="up_tx")
            if up_tx is not None:
                st.session_state.tx_bytes = up_tx.getvalue()
        with col2:
            up_acct = st.file_uploader("Account.csv (optional — adds cash & deposit info)", type="csv", key="up_acct")
            if up_acct is not None:
                st.session_state.acct_bytes = up_acct.getvalue()

    if st.session_state.tx_bytes is None:
        st.info("Upload your DEGIRO **Transactions.csv** above to get started.")
        return

    # ── Always work from BytesIO — never write to disk ────────────────────────
    tx_source   = io.BytesIO(st.session_state.tx_bytes)
    acct_source = io.BytesIO(st.session_state.acct_bytes) if st.session_state.acct_bytes else None

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df_tx = load_transactions(tx_source)
    except Exception as e:
        st.error(f"⚠️ Could not read Transactions.csv: {e}\n\nPlease re-upload a valid DEGIRO Transactions export.")
        st.session_state.tx_bytes = None
        return

    holdings = compute_holdings(df_tx)
    eur_usd = get_exchange_rate()
    deposits_df, eur_balance_hist, net_deposited = parse_account_csv(acct_source) if acct_source else (pd.DataFrame(), pd.Series(dtype=float), 0.0)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.markdown(f"**EUR/USD:** {eur_usd:.4f}")
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%H:%M:%S')}")
    if st.sidebar.button("🔄 Refresh prices"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # Blur toggle
    if "blur" not in st.session_state:
        st.session_state.blur = False
    if st.sidebar.button("🙈 Blur numbers" if not st.session_state.blur else "👁️ Show numbers"):
        st.session_state.blur = not st.session_state.blur
        st.rerun()
    blur = st.session_state.blur
    if blur:
        st.sidebar.caption("Absolute values hidden — sharing mode on")

    # ── Current portfolio value ───────────────────────────────────────────────
    portfolio_df = get_portfolio_value(holdings, eur_usd)
    if portfolio_df.empty:
        st.error("Could not fetch prices for any holdings.")
        return

    stock_value = portfolio_df["Market Value (EUR)"].sum()
    cash_eur = eur_balance_hist.iloc[-1] if not eur_balance_hist.empty else 0.0
    total_value = stock_value + cash_eur

    # ── Historical portfolio (stocks + cash) ──────────────────────────────────
    if not deposits_df.empty:
        first_deposit_date = deposits_df["Datum"].min()
    else:
        first_deposit_date = pd.Timestamp("2019-05-06")

    start_str = first_deposit_date.strftime("%Y-%m-%d")

    holdings_frozen = tuple((isin, dict(h)) for isin, h in holdings.items())
    stock_hist = compute_stock_history(holdings_frozen, start_str)

    if not stock_hist.empty and not eur_balance_hist.empty:
        stock_hist.index = pd.to_datetime(stock_hist.index).normalize()
        eur_balance_hist.index = pd.to_datetime(eur_balance_hist.index).normalize()
        combined = pd.DataFrame({"stocks": stock_hist, "cash": eur_balance_hist})
        combined = combined.ffill().fillna(0)
        total_hist = combined["stocks"] + combined["cash"]
    else:
        total_hist = stock_hist

    ath = total_hist.max() if not total_hist.empty else total_value
    ath_date = total_hist.idxmax() if not total_hist.empty else None
    distance_from_ath = ((total_value - ath) / ath) * 100 if ath > 0 else 0

    # ── Top metrics ───────────────────────────────────────────────────────────
    real_pnl = total_value - net_deposited
    real_pnl_pct = (real_pnl / net_deposited * 100) if net_deposited > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Portfolio Value", v(total_value, blur))
    col2.metric("Stocks", v(stock_value, blur))
    col3.metric("Cash", v(cash_eur, blur))
    col4.metric("Net Deposited", v(net_deposited, blur))
    col5.metric("Real Return", v(real_pnl, blur), f"{real_pnl_pct:+.1f}%")

    # ── ATH metric ────────────────────────────────────────────────────────────
    ath_date_str = ath_date.strftime("%d %b %Y") if ath_date is not None else ""
    ath_col1, _ = st.columns([1, 3])
    ath_col1.metric(
        f"All-Time High  ({ath_date_str})",
        v(ath, blur),
        f"{distance_from_ath:+.1f}% from ATH",
    )

    st.divider()

    # ── Period returns ────────────────────────────────────────────────────────
    st.subheader("Returns by Period")
    periods = {
        "1 Week": 5,
        "1 Month": 21,
        "YTD": None,
        "TTM": 252,
        "All Time": -1,
    }

    period_cols = st.columns(len(periods))
    for i, (label, days) in enumerate(periods.items()):
        if total_hist.empty:
            period_cols[i].metric(label, "N/A")
            continue

        if days == -1:
            start_val = total_hist.iloc[0]
        elif days is None:
            ytd_start = pd.Timestamp(f"{date.today().year}-01-01")
            future = total_hist[total_hist.index >= ytd_start]
            start_val = future.iloc[0] if not future.empty else total_hist.iloc[0]
        else:
            idx = max(0, len(total_hist) - days - 1)
            start_val = total_hist.iloc[idx]

        end_val = total_hist.iloc[-1]
        ret_abs = end_val - start_val
        ret_pct = ((end_val - start_val) / start_val * 100) if start_val > 0 else 0
        # value = absolute (blurred when on), delta = % (always visible)
        period_cols[i].metric(label, v(ret_abs, blur, "€{:+,.0f}"), f"{ret_pct:+.1f}%")

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.subheader("Portfolio Value Over Time (stocks + cash)")

    chart_opts = ["All Time", "5 Years", "3 Years", "1 Year", "YTD", "6 Months", "3 Months", "1 Month"]
    chart_sel = st.selectbox("Period", chart_opts, index=0)

    today = pd.Timestamp(date.today())
    cutoffs = {
        "All Time": None,
        "5 Years": today - pd.DateOffset(years=5),
        "3 Years": today - pd.DateOffset(years=3),
        "1 Year": today - pd.DateOffset(years=1),
        "YTD": pd.Timestamp(f"{date.today().year}-01-01"),
        "6 Months": today - pd.DateOffset(months=6),
        "3 Months": today - pd.DateOffset(months=3),
        "1 Month": today - pd.DateOffset(months=1),
    }
    cutoff = cutoffs[chart_sel]
    chart_data = total_hist[total_hist.index >= cutoff] if cutoff is not None else total_hist

    if not chart_data.empty:
        fig = go.Figure()

        # Portfolio value area
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data.values,
            mode="lines",
            name="Portfolio (stocks + cash)",
            fill="tozeroy",
            line=dict(color="#2E86AB", width=2),
            fillcolor="rgba(46, 134, 171, 0.12)",
            hovertemplate="€%{y:,.0f}<extra>Portfolio</extra>" if not blur else "%{x}<extra>Portfolio</extra>",
        ))

        # Net deposited reference line
        if not deposits_df.empty:
            dep_cum = deposits_df.sort_values("Datum").copy()
            dep_cum["Cumulative"] = dep_cum["Deposit"].cumsum()
            dep_cum = dep_cum.set_index("Datum")["Cumulative"]
            full_dep_idx = pd.date_range(dep_cum.index.min(), today, freq="D")
            dep_series = dep_cum.reindex(full_dep_idx).ffill().fillna(0)
            dep_series.index = pd.to_datetime(dep_series.index).normalize()
            if cutoff is not None:
                dep_series = dep_series[dep_series.index >= cutoff]

            fig.add_trace(go.Scatter(
                x=dep_series.index,
                y=dep_series.values,
                mode="lines",
                name="Net Deposited",
                line=dict(color="#FF6B35", width=2, dash="dot"),
                hovertemplate="€%{y:,.0f}<extra>Deposited</extra>" if not blur else "%{x}<extra>Deposited</extra>",
            ))

        # Deposit events — vertical lines always shown; labels hidden when blurred
        if not deposits_df.empty:
            dep_in_range = deposits_df.copy()
            if cutoff is not None:
                dep_in_range = dep_in_range[dep_in_range["Datum"] >= cutoff]

            for _, dep in dep_in_range.iterrows():
                dep_date = dep["Datum"]
                nearby = chart_data[chart_data.index >= dep_date]
                y_pos = nearby.iloc[0] if not nearby.empty else chart_data.iloc[-1]
                fig.add_vline(
                    x=dep_date,
                    line=dict(color="rgba(255, 107, 53, 0.5)", width=1.5, dash="dash"),
                )
                if not blur:
                    fig.add_annotation(
                        x=dep_date,
                        y=y_pos * 1.02,
                        text=f"+€{dep['Deposit']:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color="#FF6B35"),
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="#FF6B35",
                        borderwidth=1,
                    )

        # ATH line — value label hidden when blurred, date always shown
        if ath > 0:
            ath_label = f"ATH — {ath_date_str}" if blur else f"ATH €{ath:,.0f} — {ath_date_str}"
            fig.add_hline(
                y=ath,
                line=dict(color="gold", width=1.5, dash="dash"),
                annotation_text=ath_label,
                annotation_position="top right",
            )

        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="" if blur else "Value (EUR)",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        if blur:
            fig.update_yaxes(showticklabels=False, showgrid=True)
        else:
            fig.update_yaxes(tickprefix="€", tickformat=",.0f")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Risk & Exposure ───────────────────────────────────────────────────────
    st.subheader("📊 Risk & Exposure")

    risk = compute_risk(portfolio_df, total_hist, cash_eur, total_value)

    # Key risk metrics row
    r1, r2, r3, r4, r5 = st.columns(5)

    # Beta
    beta_val = risk["beta"]
    beta_str = f"{beta_val:.2f}" if beta_val is not None else "N/A"
    beta_delta = "High sensitivity" if beta_val and beta_val > 1.5 else \
                 "Moderate" if beta_val and beta_val > 1.0 else "Low sensitivity"
    r1.metric("Portfolio Beta", beta_str, beta_delta,
              delta_color="inverse" if beta_val and beta_val > 1.5 else
                          "off" if beta_val and beta_val > 1.0 else "normal")

    # Volatility
    vol = risk["volatility"]
    vol_str = f"{vol:.1f}%" if vol is not None else "N/A"
    vol_delta = "High" if vol and vol > 30 else "Moderate" if vol and vol > 15 else "Low"
    r2.metric("Volatility (1Y ann.)", vol_str, vol_delta,
              delta_color="inverse" if vol and vol > 30 else "off" if vol and vol > 15 else "normal")

    # Max drawdown
    mdd = risk["max_drawdown"]
    mdd_str = f"{mdd:.1f}%" if mdd is not None else "N/A"
    mdd_delta = "Severe" if mdd and mdd < -40 else "Significant" if mdd and mdd < -20 else "Moderate"
    r3.metric("Max Drawdown", mdd_str, mdd_delta, delta_color="off")

    # Sharpe
    sharpe = risk["sharpe"]
    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
    sharpe_delta = "Excellent" if sharpe and sharpe > 1.5 else \
                   "Good" if sharpe and sharpe > 1.0 else \
                   "Fair" if sharpe and sharpe > 0.5 else "Poor"
    r4.metric("Sharpe Ratio (1Y)", sharpe_str, sharpe_delta,
              delta_color="normal" if sharpe and sharpe > 1.0 else
                          "off" if sharpe and sharpe > 0.5 else "inverse")

    # Concentration
    hhi = risk["hhi"]
    hhi_str = f"{hhi:.3f}"
    hhi_delta = f"Top: {risk['top_name'].split()[0]}  ({risk['top_weight']:.1f}%)"
    hhi_label = "High conc." if hhi > 0.25 else "Moderate" if hhi > 0.10 else "Diversified"
    r5.metric(f"HHI Concentration ({hhi_label})", hhi_str, hhi_delta, delta_color="off")

    st.caption("Beta: market sensitivity (1 = moves with market). Volatility: annualised standard deviation. "
               "Max drawdown: largest peak-to-trough decline. Sharpe: return per unit of risk (risk-free = 2.5%). "
               "HHI: 0 = perfectly diversified, 1 = single stock.")

    # Charts row
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("**Sector Allocation**")
        sec = risk["sector_pct"]
        labels = list(sec.keys())
        values = list(sec.values())
        colors = [SECTOR_COLORS.get(l, SECTOR_COLORS["Other"]) for l in labels]

        fig_sec = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            sort=True,
            direction="clockwise",
        ))
        fig_sec.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    with chart_col2:
        # Country breakdown
        st.markdown("**Geographic Exposure**")
        ctry = risk["country_pct"]
        ctry_labels = list(ctry.keys())
        ctry_vals   = [ctry[k] for k in ctry_labels]

        fig_ctry = go.Figure(go.Bar(
            y=ctry_labels,
            x=ctry_vals,
            orientation="h",
            marker=dict(color="#2E86AB"),
            text=[f"{x:.1f}%" for x in ctry_vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        ))
        fig_ctry.update_layout(
            height=170,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(showticklabels=False, range=[0, max(ctry_vals) * 1.25]),
            yaxis=dict(autorange="reversed"),
            bargap=0.3,
        )
        st.plotly_chart(fig_ctry, use_container_width=True)

        # Currency breakdown
        st.markdown("**Currency Exposure**")
        curr = risk["currency_pct"]
        curr_labels = list(curr.keys())
        curr_vals   = [curr[k] for k in curr_labels]
        curr_colors = {"EUR": "#44BBA4", "USD": "#2E86AB"}

        fig_curr = go.Figure(go.Bar(
            y=curr_labels,
            x=curr_vals,
            orientation="h",
            marker=dict(color=[curr_colors.get(l, "#90A4AE") for l in curr_labels]),
            text=[f"{x:.1f}%" for x in curr_vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        ))
        fig_curr.update_layout(
            height=120,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(showticklabels=False, range=[0, max(curr_vals) * 1.25]),
            yaxis=dict(autorange="reversed"),
            bargap=0.3,
        )
        st.plotly_chart(fig_curr, use_container_width=True)

    # Position concentration bar chart
    st.markdown("**Position Sizes** (% of total portfolio incl. cash)")
    conc_df = portfolio_df[["Product", "Market Value (EUR)"]].copy()
    conc_df["Weight (%)"] = conc_df["Market Value (EUR)"] / total_value * 100
    conc_df["Label"] = conc_df["Product"].str.split().str[0:2].str.join(" ")
    conc_df = conc_df.sort_values("Weight (%)", ascending=True)

    fig_conc = go.Figure(go.Bar(
        y=conc_df["Label"],
        x=conc_df["Weight (%)"],
        orientation="h",
        marker=dict(
            color=conc_df["Weight (%)"],
            colorscale=[[0, "#44BBA4"], [0.5, "#2E86AB"], [1, "#A23B72"]],
            showscale=False,
        ),
        text=[f"{w:.1f}%" for w in conc_df["Weight (%)"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
    ))
    # Cash bar
    cash_pct = cash_eur / total_value * 100
    fig_conc.add_trace(go.Bar(
        y=["Cash"],
        x=[cash_pct],
        orientation="h",
        marker=dict(color="#44BBA4"),
        text=[f"{cash_pct:.1f}%"],
        textposition="outside",
        hovertemplate=f"<b>Cash</b>: {cash_pct:.1f}%<extra></extra>",
    ))
    fig_conc.update_layout(
        height=max(300, (len(conc_df) + 1) * 32),
        margin=dict(l=0, r=50, t=10, b=0),
        xaxis=dict(showticklabels=False, range=[0, conc_df["Weight (%)"].max() * 1.3]),
        yaxis=dict(autorange="reversed"),
        bargap=0.25,
        showlegend=False,
    )
    st.plotly_chart(fig_conc, use_container_width=True)

    st.divider()

    # ── Holdings table ────────────────────────────────────────────────────────
    st.subheader("Holdings")
    display_df = portfolio_df.sort_values("Market Value (EUR)", ascending=False).reset_index(drop=True)
    display_df["Weight (%)"] = (display_df["Market Value (EUR)"] / total_value * 100).round(1)

    if blur:
        # Show only non-monetary columns + percentages
        show_cols = ["Product", "Ticker", "P&L (%)", "Weight (%)"]
        fmt = {"P&L (%)": "{:+.1f}%", "Weight (%)": "{:.1f}%"}
        styled = display_df[show_cols].style.format(fmt).map(
            lambda val: "color: green" if isinstance(val, float) and val > 0
                        else "color: red" if isinstance(val, float) and val < 0 else "",
            subset=["P&L (%)"]
        )
    else:
        show_cols = ["Product", "Ticker", "Shares", "Price", "Currency",
                     "Market Value (EUR)", "Cost Basis (EUR)", "P&L (EUR)", "P&L (%)", "Weight (%)"]
        fmt = {
            "Price": "{:.2f}",
            "Market Value (EUR)": "€{:,.2f}",
            "Cost Basis (EUR)": "€{:,.2f}",
            "P&L (EUR)": "€{:,.2f}",
            "P&L (%)": "{:+.1f}%",
            "Weight (%)": "{:.1f}%",
        }
        styled = display_df[show_cols].style.format(fmt).map(
            lambda val: "color: green" if isinstance(val, float) and val > 0
                        else "color: red" if isinstance(val, float) and val < 0 else "",
            subset=["P&L (EUR)", "P&L (%)"]
        )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Deposit history ───────────────────────────────────────────────────────
    with st.expander("💰 Deposit History"):
        if not deposits_df.empty:
            dep_display = deposits_df.copy()
            dep_display["Cumulative"] = dep_display["Deposit"].cumsum()
            dep_display["Datum"] = dep_display["Datum"].dt.strftime("%d %b %Y")
            if blur:
                st.dataframe(
                    dep_display[["Datum", "Omschrijving"]].rename(
                        columns={"Datum": "Date", "Omschrijving": "Description"}
                    ),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.dataframe(
                    dep_display.rename(columns={
                        "Datum": "Date", "Deposit": "Amount (EUR)",
                        "Omschrijving": "Description", "Cumulative": "Cumulative (EUR)"
                    }).style.format({"Amount (EUR)": "€{:,.2f}", "Cumulative (EUR)": "€{:,.2f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.markdown(f"**Total deposited:** €{deposits_df['Deposit'].sum():,.2f}  |  **Net:** €{net_deposited:,.2f}")

    # ── Transaction history ───────────────────────────────────────────────────
    with st.expander("📋 Transaction History"):
        show_tx_cols = ["Datum", "Product", "ISIN"] if blur else ["Datum", "Product", "ISIN", "Aantal", "Koers", "Totaal EUR"]
        st.dataframe(
            df_tx[show_tx_cols].sort_values("Datum", ascending=False),
            use_container_width=True, hide_index=True,
        )


if __name__ == "__main__":
    main()
