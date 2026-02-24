import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz Portfolio Lab", layout="wide")

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload adjusted close CSV", type=["csv"])
rf = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.5, 0.06, 0.005)
long_only = st.sidebar.checkbox("Long-only (no short selling)", True)
trading_days = st.sidebar.number_input("Trading days per year", 200, 365, 252, 1)
num_pts = st.sidebar.slider("Frontier points", 20, 150, 60, 5)
top_hold = st.sidebar.slider("Top holdings to display", 5, 30, 10, 1)

st.title("📊 Mean–Variance Portfolio Lab (Markowitz)")
st.caption("Designed by Prof. Shalini Velappan, IIM Trichy.")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_prices(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, how="all")
    return df

if uploaded is None:
    st.warning("Upload a CSV of adjusted closing prices to begin.")
    st.stop()

prices = load_prices(uploaded)

returns = prices.pct_change().dropna(how="all")
returns = returns.dropna(axis=1, how="all")

tickers = returns.columns.tolist()
n = len(tickers)

# ===============================
# EXPECTED RETURNS
# ===============================

# Arithmetic mean (Markowitz)
mu_daily = returns.mean()
mu = mu_daily * trading_days
mu_vec = mu.values

# ===== Geometric (CAGR) =====
clean_prices = prices[tickers].dropna()

total_years = (clean_prices.index[-1] - clean_prices.index[0]).days / 365.25
cagr = (clean_prices.iloc[-1] / clean_prices.iloc[0]) ** (1 / total_years) - 1

geo_vec = cagr.values

# ===============================
# COVARIANCE MATRIX
# ===============================
clean_returns = returns.dropna()

try:
    lw = LedoitWolf().fit(clean_returns.values)
    cov_mat = lw.covariance_ * trading_days
except:
    cov_mat = clean_returns.cov().values * trading_days

# ensure positive definite
eigmin = np.linalg.eigvalsh(cov_mat).min()
if eigmin <= 1e-10:
    cov_mat += np.eye(n) * (abs(eigmin) + 1e-8)

# ===============================
# OPTIMIZATION
# ===============================
bounds = None if not long_only else tuple((0.0, 1.0) for _ in range(n))
cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

def port_return(w):
    return float(w @ mu_vec)

def port_vol(w):
    return float(np.sqrt(w.T @ cov_mat @ w))

def min_var_portfolio():
    x0 = np.repeat(1.0/n, n)
    res = minimize(lambda w: float(w.T @ cov_mat @ w),
                   x0, method='SLSQP',
                   bounds=bounds,
                   constraints=(cons_sum,))
    return res.x

def max_sharpe_portfolio():
    def neg_sharpe(w):
        v = port_vol(w)
        return - (port_return(w) - rf) / v if v > 0 else 1e9

    x0 = np.repeat(1.0/n, n)
    res = minimize(neg_sharpe,
                   x0, method='SLSQP',
                   bounds=bounds,
                   constraints=(cons_sum,))
    return res.x

def min_var_for_target(target):
    cons = (cons_sum,
            {'type':'eq', 'fun': lambda w: port_return(w) - target})

    x0 = np.repeat(1.0/n, n)
    res = minimize(lambda w: float(w.T @ cov_mat @ w),
                   x0, method='SLSQP',
                   bounds=bounds,
                   constraints=cons)
    return res.x if res.success else None

# ===============================
# RUN OPTIMIZATION
# ===============================
w_mv = min_var_portfolio()
w_ms = max_sharpe_portfolio()

r_mv, v_mv = port_return(w_mv), port_vol(w_mv)
r_ms, v_ms = port_return(w_ms), port_vol(w_ms)

geo_r_mv = float(w_mv @ geo_vec)
geo_r_ms = float(w_ms @ geo_vec)

# ===============================
# FRONTIER
# ===============================
targets = np.linspace(min(mu_vec), max(mu_vec), num_pts)

frontier_rs, frontier_vs, frontier_ws = [], [], []

for t in targets:
    w = min_var_for_target(t)
    if w is not None:
        frontier_ws.append(w)
        frontier_rs.append(port_return(w))
        frontier_vs.append(port_vol(w))

# ===============================
# DISPLAY
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Global Minimum Variance")
    st.metric("Return (Arithmetic)", f"{r_mv:.2%}")
    st.metric("Return (Geometric)", f"{geo_r_mv:.2%}")
    st.metric("Volatility", f"{v_mv:.2%}")

with col2:
    sharpe = (r_ms - rf)/v_ms
    st.subheader("⭐ Tangency Portfolio")
    st.metric("Return (Arithmetic)", f"{r_ms:.2%}")
    st.metric("Return (Geometric)", f"{geo_r_ms:.2%}")
    st.metric("Volatility", f"{v_ms:.2%}")
    st.metric("Sharpe", f"{sharpe:.2f}")

# ===============================
# PLOT
# ===============================
fig = make_subplots(rows=1, cols=2,
                    column_widths=[0.6, 0.4],
                    specs=[[{"type":"scatter"}, {"type":"bar"}]])

fig.add_trace(go.Scatter(x=frontier_vs, y=frontier_rs,
                         mode='lines', name='Frontier'), row=1, col=1)

fig.add_trace(go.Scatter(x=[v_mv], y=[r_mv],
                         mode='markers', name='Min Var'), row=1, col=1)

fig.add_trace(go.Scatter(x=[v_ms], y=[r_ms],
                         mode='markers', name='Tangency'), row=1, col=1)

# CML
cml_x = np.linspace(0, max(frontier_vs), 50)
cml_y = rf + sharpe * cml_x
fig.add_trace(go.Scatter(x=cml_x, y=cml_y,
                         line=dict(dash='dash'),
                         name='CML'), row=1, col=1)

# weights
sel_idx = len(frontier_ws)//2
weights = pd.Series(frontier_ws[sel_idx], index=tickers)\
            .sort_values(ascending=False)\
            .head(top_hold)

fig.add_trace(go.Bar(x=weights.index, y=weights.values),
              row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# FRONTIER SELECTION
# ===============================
idx = st.slider("Select frontier portfolio",
                0, len(frontier_ws)-1, sel_idx)

st.dataframe(
    pd.Series(frontier_ws[idx], index=tickers)
      .sort_values(ascending=False)
      .head(top_hold)
      .to_frame("Weight")
)
