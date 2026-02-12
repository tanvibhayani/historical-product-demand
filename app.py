import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Product Demand Analysis",
    page_icon="📦",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
h1 {font-size:32px;}
h2 {font-size:24px;}
.block-container {padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("demand_model.pkl")
scaler = joblib.load("demand_scaler.pkl")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Historical Product Demand.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    df["Order_Demand"] = (
        df["Order_Demand"].astype(str)
        .str.replace(",", "")
        .str.replace(r"\(", "-", regex=True)
        .str.replace(r"\)", "", regex=True)
    )
    df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")
    df.dropna(subset=["Order_Demand"], inplace=True)

    df = df.sort_values("Date")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.dayofweek
    return df

df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("📅 Select Year")
selected_year = st.sidebar.selectbox(
    "Choose Year",
    sorted(df["Year"].unique())
)

df_year = df[df["Year"] == selected_year]

# ---------------- Title ----------------
st.title("📈 Historical Product Demand Analysis")
st.caption("Machine Learning based Demand Forecasting")

# ---------------- Tabs (ONLY 2) ----------------
tab1, tab2 = st.tabs(["📊 Year-wise Visualization", "🔮 Prediction & Data"])

# ================= TAB 1 : YEAR-WISE VISUALIZATION =================
with tab1:
    st.subheader(f"Demand Trend for Year {selected_year}")

    yearly_trend = df_year.groupby("Date")["Order_Demand"].sum()

    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(yearly_trend.index, yearly_trend.values)
    ax.set_xlabel("Date")
    ax.set_ylabel("Order Demand")
    ax.grid(alpha=0.3)

    st.pyplot(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Total Demand", f"{int(df_year['Order_Demand'].sum()):,}")
    c2.metric("📊 Avg Daily Demand", f"{int(yearly_trend.mean()):,}")
    c3.metric("📅 Records", df_year.shape[0])

# ================= TAB 2 : PREDICTION + VISUALIZATION + RAW DATA =================
with tab2:
    st.subheader("🔮 Predict Future Product Demand")

    future_date = st.date_input("Select Future Date")

    if st.button("Predict Demand"):
        X = pd.DataFrame(
            [[future_date.year, future_date.month, future_date.day, future_date.weekday()]],
            columns=["Year", "Month", "Day", "Weekday"]
        )

        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)

        st.success(f"📦 Predicted Demand: **{int(prediction[0]):,} units**")

        st.divider()

        st.subheader("📊 Recent Demand Trend")
        recent = df.tail(30).groupby("Date")["Order_Demand"].sum()

        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(recent.index, recent.values)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Orders")
        ax2.grid(alpha=0.3)

        st.pyplot(fig2, use_container_width=True)

        st.subheader("📄 Sample Raw Data")
        st.dataframe(df.tail(20), height=250)

st.caption("ML Project | Demand Forecasting using Regression")
