import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("Historical Product Demand.csv")

# Fix Date parsing
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Drop rows where Date is missing
df.dropna(subset=["Date"], inplace=True)

# Clean Order_Demand
df["Order_Demand"] = (
    df["Order_Demand"]
    .astype(str)
    .str.replace(",", "")
    .str.replace(r"\(", "-", regex=True)
    .str.replace(r"\)", "", regex=True)
)

# Convert to numeric & drop NaN
df["Order_Demand"] = pd.to_numeric(df["Order_Demand"], errors="coerce")
df.dropna(subset=["Order_Demand"], inplace=True)

# Feature engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.dayofweek

X = df[["Year", "Month", "Day", "Weekday"]]
y = df["Order_Demand"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save files
joblib.dump(model, "demand_model.pkl")
joblib.dump(scaler, "demand_scaler.pkl")

print("✅ demand_model.pkl & demand_scaler.pkl CREATED SUCCESSFULLY")
