import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Product Demand Analysis",
    page_icon="📦",
    layout="centered"
)

# ---------------- Custom Minimal CSS ----------------
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
h1 {font-size: 26px;}
h2 {font-size: 20px;}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("📈 Historical Product Demand Analysis")
st.caption("ML-based demand forecasting using historical sales data")
st.divider()

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
        df["Order_Demand"]
        .astype(str)
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
st.sidebar.header("🔍 Filters")
years = st.sidebar.multiselect(
    "Select Year",
    sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())
)
df = df[df["Year"].isin(years)]

# ---------------- KPI Section ----------------
c1, c2, c3 = st.columns(3)
c1.metric("📦 Total Orders", f"{int(df['Order_Demand'].sum()):,}")
c2.metric("📊 Avg Daily Demand", f"{int(df.groupby('Date')['Order_Demand'].sum().mean()):,}")
c3.metric("📅 Records", df.shape[0])

st.divider()

# ---------------- Trend Chart ----------------
st.subheader("📊 Demand Trend")

daily_total = df.groupby("Date")["Order_Demand"].sum()

fig, ax = plt.subplots(figsize=(9,3))
ax.plot(daily_total.index, daily_total.values)
ax.set_xlabel("Date")
ax.set_ylabel("Orders")
ax.grid(alpha=0.3)

st.pyplot(fig, use_container_width=True)

st.divider()

# ---------------- Prediction Section ----------------
st.subheader("🔮 Predict Future Demand")

colA, colB = st.columns([2,1])

with colA:
    future_date = st.date_input("Select Future Date")

with colB:
    st.write("")   # spacing
    st.write("")   # spacing
    predict = st.button("Predict")

if predict:
    try:
        X = pd.DataFrame(
            [[future_date.year, future_date.month, future_date.day, future_date.weekday()]],
            columns=["Year", "Month", "Day", "Weekday"]
        )

        X_scaled = scaler.transform(X)
        result = model.predict(X_scaled)

        st.success(f"📦 Predicted Demand: **{int(result[0]):,} units**")
        st.info("Prediction is based on historical trends. Actual demand may vary.")

    except:
        st.error("Prediction failed. Please check model files.")

# ---------------- Raw Data ----------------
with st.expander("📄 View Sample Data"):
    st.dataframe(df.head(20), height=200)

st.caption("Machine Learning Project | Demand Forecasting using Regression")
