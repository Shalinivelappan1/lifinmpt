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
# =====================================================
# 📚 LEARNING PANEL — FULL THEORY + EXPLANATION
# =====================================================
st.divider()
st.header("📚 Understanding the Mathematics Behind the Portfolio")

st.write("""
This section explains how returns, risk, variance, covariance,
correlation, and portfolio diversification are calculated.
""")

# =====================================================
# RETURNS
# =====================================================
with st.expander("1️⃣ Returns — What are we measuring?"):

    st.write("""
A return measures how much an asset’s price changes from one period to the next.
It tells us how much we gained or lost relative to the previous price.
""")

    st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

    st.write("""
Arithmetic return is the simple average return.
It is used in Markowitz optimisation.
""")

    st.latex(r"E[R] = \bar{R} \times \text{periods per year}")

    st.write("""
Geometric return (CAGR) measures compounded growth.
It reflects what investors actually earn over time.
""")

    st.latex(r"CAGR = \left(\frac{P_T}{P_0}\right)^{\frac{1}{T}} - 1")


# =====================================================
# VOLATILITY
# =====================================================
with st.expander("2️⃣ Volatility — How do we measure risk?"):

    st.write("""
Volatility measures how much returns fluctuate around their average.
Higher volatility means higher uncertainty and therefore higher risk.
""")

    st.write("Volatility is the square root of variance:")

    st.latex(r"\sigma = \sqrt{Var(R)}")

    st.write("""
Variance measures the average squared deviation of returns from the mean.
Squaring ensures negative and positive deviations both increase risk.
""")

    st.write("Annualised volatility scales risk to yearly terms:")

    st.latex(r"\sigma_{annual} = \sigma_{period} \times \sqrt{\text{periods per year}}")

    vol = returns.std() * np.sqrt(periods_per_year)

    st.write("Volatility of each stock in your dataset:")
    st.dataframe(vol.to_frame("Annual Volatility"))


# =====================================================
# VARIANCE & COVARIANCE
# =====================================================
with st.expander("3️⃣ Variance & Covariance — How assets move together"):

    st.write("""
Variance measures the risk of a single asset.

Covariance measures how two assets move together.
It is crucial for diversification.
""")

    st.write("Covariance formula:")

    st.latex(r"Cov(i,j) = E[(R_i - \mu_i)(R_j - \mu_j)]")

    st.write("""
Interpretation:
• Positive covariance → assets move together  
• Negative covariance → assets move in opposite directions  
• Zero → no linear relationship  
""")

    st.write("Covariance matrix used in optimisation:")

    st.latex(r"\Sigma = Cov(R_i, R_j)")

    cov_df = pd.DataFrame(cov_mat, index=tickers, columns=tickers)
    st.dataframe(cov_df.style.format("{:.4f}"))


# =====================================================
# CORRELATION
# =====================================================
with st.expander("4️⃣ Correlation — Standardised relationship"):

    st.write("""
Correlation standardises covariance to lie between -1 and +1.
It makes relationships easier to interpret.
""")

    st.latex(r"\rho_{ij} = \frac{Cov(i,j)}{\sigma_i \sigma_j}")

    st.write("""
Interpretation:
+1  → move perfectly together  
0   → independent  
-1  → move perfectly opposite  

Diversification works best when correlations are low or negative.
""")

    corr = returns.corr()
    st.write("Correlation matrix:")
    st.dataframe(corr.style.format("{:.2f}"))


# =====================================================
# PORTFOLIO RETURN
# =====================================================
with st.expander("5️⃣ Portfolio Return — Weighted average"):

    st.write("""
Portfolio return is simply the weighted average of individual asset returns.
""")

    st.latex(r"R_p = \sum_{i=1}^{n} w_i R_i")

    st.write("Min-variance portfolio weights:")
    st.dataframe(pd.Series(w_mv, index=tickers).to_frame("Weight"))

    st.write("Tangency portfolio weights:")
    st.dataframe(pd.Series(w_ms, index=tickers).to_frame("Weight"))


# =====================================================
# PORTFOLIO RISK
# =====================================================
with st.expander("6️⃣ Portfolio Risk — Why diversification works"):

    st.write("""
Portfolio risk depends not only on individual volatility,
but also on how assets move together.
""")

    st.latex(r"\sigma_p^2 = w^T \Sigma w")

    st.write("Portfolio volatility:")

    st.latex(r"\sigma_p = \sqrt{w^T \Sigma w}")

    st.write("""
If correlations are low, portfolio risk falls.

This is the core insight of Markowitz:
You don't need the best stock —
you need the best combination of stocks.
""")
