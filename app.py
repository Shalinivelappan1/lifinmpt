import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

st.set_page_config(page_title="Markowitz Portfolio Lab", layout="wide")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload adjusted close CSV", type=["csv"])
rf = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.5, 0.06, 0.005)
long_only = st.sidebar.checkbox("Long-only (no short selling)", True)
num_pts = st.sidebar.slider("Frontier points", 20, 150, 60)
top_hold = st.sidebar.slider("Top holdings to display", 5, 30, 10)

st.title("📊 Mean–Variance Portfolio Lab (Markowitz)")
st.caption("Designed by Prof. Shalini Velappan, IIM Trichy")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_prices(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    df = df.dropna(axis=1, how="all")
    return df

if uploaded is None:
    st.warning("Upload a CSV file to begin.")
    st.stop()

prices = load_prices(uploaded)

returns = prices.pct_change().dropna()
tickers = returns.columns.tolist()
n = len(tickers)

# =====================================================
# AUTO DETECT DATA FREQUENCY
# =====================================================
date_diffs = prices.index.to_series().diff().dropna()
avg_days = date_diffs.dt.days.mean()

if avg_days > 25:
    periods_per_year = 12
    freq_label = "Monthly data detected"
else:
    periods_per_year = 252
    freq_label = "Daily data detected"

st.sidebar.info(freq_label)

# =====================================================
# EXPECTED RETURNS
# =====================================================
mu_period = returns.mean()
mu = mu_period * periods_per_year
mu_vec = mu.values

# Geometric CAGR
clean_prices = prices[tickers].dropna()
total_years = (clean_prices.index[-1] - clean_prices.index[0]).days / 365.25
cagr = (clean_prices.iloc[-1] / clean_prices.iloc[0]) ** (1 / total_years) - 1
geo_vec = cagr.values

# =====================================================
# COVARIANCE MATRIX
# =====================================================
clean_returns = returns.dropna()

try:
    lw = LedoitWolf().fit(clean_returns.values)
    cov_mat = lw.covariance_ * periods_per_year
except:
    cov_mat = clean_returns.cov().values * periods_per_year

# Ensure positive definite
eigmin = np.linalg.eigvalsh(cov_mat).min()
if eigmin <= 1e-10:
    cov_mat += np.eye(n) * (abs(eigmin) + 1e-8)

# =====================================================
# OPTIMIZATION FUNCTIONS
# =====================================================
bounds = None if not long_only else tuple((0, 1) for _ in range(n))
cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

def port_return(w):
    return float(w @ mu_vec)

def port_vol(w):
    return float(np.sqrt(w.T @ cov_mat @ w))

def min_var():
    x0 = np.repeat(1/n, n)
    res = minimize(lambda w: w.T @ cov_mat @ w,
                   x0, bounds=bounds, constraints=cons)
    return res.x

def max_sharpe():
    def neg_sharpe(w):
        v = port_vol(w)
        return -(port_return(w) - rf) / v if v > 0 else 1e9
    x0 = np.repeat(1/n, n)
    res = minimize(neg_sharpe,
                   x0, bounds=bounds, constraints=cons)
    return res.x

def frontier_for_target(target):
    cons2 = (cons,
             {'type': 'eq', 'fun': lambda w: port_return(w) - target})
    x0 = np.repeat(1/n, n)
    res = minimize(lambda w: w.T @ cov_mat @ w,
                   x0, bounds=bounds, constraints=cons2)
    return res.x if res.success else None

# =====================================================
# RUN OPTIMIZATION
# =====================================================
w_mv = min_var()
w_ms = max_sharpe()

r_mv, v_mv = port_return(w_mv), port_vol(w_mv)
r_ms, v_ms = port_return(w_ms), port_vol(w_ms)

geo_r_mv = float(w_mv @ geo_vec)
geo_r_ms = float(w_ms @ geo_vec)

# =====================================================
# FRONTIER
# =====================================================
targets = np.linspace(min(mu_vec), max(mu_vec), num_pts)

frontier_rs, frontier_vs, frontier_ws = [], [], []

for t in targets:
    w = frontier_for_target(t)
    if w is not None:
        frontier_ws.append(w)
        frontier_rs.append(port_return(w))
        frontier_vs.append(port_vol(w))

# =====================================================
# DISPLAY RESULTS
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Global Minimum Variance")
    st.metric("Return (Arithmetic)", f"{r_mv:.2%}")
    st.metric("Return (Geometric)", f"{geo_r_mv:.2%}")
    st.metric("Volatility", f"{v_mv:.2%}")

with col2:
    sharpe = (r_ms - rf) / v_ms
    st.subheader("⭐ Tangency Portfolio")
    st.metric("Return (Arithmetic)", f"{r_ms:.2%}")
    st.metric("Return (Geometric)", f"{geo_r_ms:.2%}")
    st.metric("Volatility", f"{v_ms:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# =====================================================
# PLOT
# =====================================================
fig = make_subplots(rows=1, cols=2,
                    column_widths=[0.6, 0.4],
                    specs=[[{"type": "scatter"}, {"type": "bar"}]])

fig.add_trace(go.Scatter(x=frontier_vs, y=frontier_rs,
                         mode='lines', name='Efficient Frontier'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=[v_mv], y=[r_mv],
                         mode='markers', name='Min Variance'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=[v_ms], y=[r_ms],
                         mode='markers', name='Tangency'),
              row=1, col=1)

cml_x = np.linspace(0, max(frontier_vs), 50)
cml_y = rf + sharpe * cml_x

fig.add_trace(go.Scatter(x=cml_x, y=cml_y,
                         line=dict(dash='dash'),
                         name='Capital Market Line'),
              row=1, col=1)

sel_idx = len(frontier_ws) // 2
weights = pd.Series(frontier_ws[sel_idx], index=tickers)\
            .sort_values(ascending=False)\
            .head(top_hold)

fig.add_trace(go.Bar(x=weights.index, y=weights.values),
              row=1, col=2)

fig.update_xaxes(title_text="Volatility (σ)", row=1, col=1)
fig.update_yaxes(title_text="Expected Return (μ)", row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# LEARNING PANEL
# =====================================================
st.divider()
st.header("📚 Understanding the Mathematics")

with st.expander("Returns"):
    st.markdown(r"""
Single-period return:

\[
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\]

Arithmetic annual return:

\[
E[R] = \bar{R} \times \text{periods per year}
\]

Geometric return (CAGR):

\[
CAGR = \left(\frac{P_T}{P_0}\right)^{1/T} - 1
\]
""")

with st.expander("Volatility"):
    st.markdown(r"""
Volatility = standard deviation of returns

\[
\sigma = \sqrt{Var(R)}
\]

Annualised:

\[
\sigma_{annual} = \sigma_{period} \times \sqrt{\text{periods per year}}
\]
""")

with st.expander("Variance & Covariance"):
    st.markdown(r"""
Covariance matrix:

\[
\Sigma = Cov(R_i, R_j)
\]

Portfolio variance:

\[
\sigma_p^2 = w^T \Sigma w
\]
""")
    st.dataframe(pd.DataFrame(cov_mat, index=tickers, columns=tickers))

with st.expander("Correlation"):
    corr = returns.corr()
    st.dataframe(corr)

with st.expander("Portfolio Formulas"):
    st.markdown(r"""
Portfolio return:

\[
R_p = \sum w_i R_i
\]

Portfolio volatility:

\[
\sigma_p = \sqrt{w^T \Sigma w}
\]
""")
