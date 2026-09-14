import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load("models/stock_movement_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return model, scaler


try:
    model, scaler = load_model()

except Exception as e:

    st.error("❌ Model or scaler could not be loaded.")

    st.write("Make sure these files exist:")

    st.code(
        """
models/
├── stock_movement_model.pkl
└── scaler.pkl
        """
    )

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("📈 Stock Movement Prediction")

st.markdown(
    """
### Machine Learning Based Stock Prediction

Predict the **next trading day's movement**:

🟢 **UP** | 🟡 **HOLD** | 🔴 **DOWN**
"""
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Stock Settings")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

period = st.sidebar.selectbox(
    "Historical Data",
    ["1y", "2y", "5y", "10y"]
)

predict_button = st.sidebar.button(
    "🔮 Predict Movement"
)


# =========================================================
# FUNCTION TO DOWNLOAD DATA
# =========================================================

@st.cache_data
def get_stock_data(symbol, period):

    data = yf.download(
        symbol,
        period=period,
        auto_adjust=False,
        progress=False
    )

    return data


# =========================================================
# FUNCTION TO PREPARE FEATURES
# =========================================================

def prepare_features(data):

    df = data.copy()

    # Handle possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Daily return
    df["Return"] = df["Close"].pct_change()

    # 5-day return
    df["Return_5"] = df["Close"].pct_change(5)

    # Moving averages
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()

    # Volume change
    df["Volume_Change"] = df["Volume"].pct_change()

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove missing values
    df = df.dropna()

    return df


# =========================================================
# MAIN APPLICATION
# =========================================================

try:

    data = get_stock_data(stock, period)

    if data.empty:

        st.error(
            "❌ No stock data found. "
            "Please check the stock symbol."
        )

        st.stop()


    # =====================================================
    # CURRENT STOCK INFORMATION
    # =====================================================

    df = prepare_features(data)

    latest = df.iloc[-1]

    current_price = float(latest["Close"])


    # =====================================================
    # DISPLAY PRICE
    # =====================================================

    st.subheader(f"📊 {stock.upper()} Stock Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Price",
            f"${current_price:.2f}"
        )

    with col2:

        daily_return = float(latest["Return"]) * 100

        st.metric(
            "Today's Return",
            f"{daily_return:.2f}%"
        )

    with col3:

        volatility = float(latest["Volatility"]) * 100

        st.metric(
            "Volatility",
            f"{volatility:.2f}%"
        )


    # =====================================================
    # STOCK PRICE CHART
    # =====================================================

    st.subheader("📈 Historical Stock Price")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df.index,
        df["Close"]
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.set_title(
        f"{stock.upper()} Closing Price"
    )

    ax.grid(True)

    st.pyplot(fig)


    # =====================================================
    # TECHNICAL FEATURES
    # =====================================================

    st.subheader("📊 Machine Learning Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:

        st.metric(
            "5-Day Return",
            f"{float(latest['Return_5']) * 100:.2f}%"
        )

    with feature_col2:

        st.metric(
            "10-Day MA",
            f"${float(latest['MA10']):.2f}"
        )

    with feature_col3:

        st.metric(
            "20-Day MA",
            f"${float(latest['MA20']):.2f}"
        )


    # =====================================================
    # MACHINE LEARNING PREDICTION
    # =====================================================

    if predict_button:

        features = [
            "Return",
            "Return_5",
            "MA10",
            "MA20",
            "Volatility",
            "Volume_Change"
        ]

        X_latest = df[features].iloc[-1:]

        # Scaling
        X_scaled = scaler.transform(X_latest)

        # Prediction
        prediction = model.predict(X_scaled)[0]

        # =================================================
        # PREDICTION RESULT
        # =================================================

        if prediction == 1:

            movement = "🟢 UP"

            st.success(
                "Prediction: UP 📈"
            )

        elif prediction == 0:

            movement = "🟡 HOLD"

            st.warning(
                "Prediction: HOLD ⏸️"
            )

        else:

            movement = "🔴 DOWN"

            st.error(
                "Prediction: DOWN 📉"
            )


        # =================================================
        # PREDICTION PROBABILITY
        # =================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                X_scaled
            )[0]

            classes = model.classes_

            st.subheader(
                "🎯 Prediction Probability"
            )

            probability_data = []

            for cls, probability in zip(
                classes,
                probabilities
            ):

                if cls == 1:
                    name = "UP"

                elif cls == 0:
                    name = "HOLD"

                else:
                    name = "DOWN"

                probability_data.append(
                    {
                        "Movement": name,
                        "Probability": f"{probability * 100:.2f}%"
                    }
                )

            probability_df = pd.DataFrame(
                probability_data
            )

            st.table(probability_df)


        # =================================================
        # MODEL INPUT
        # =================================================

        st.subheader(
            "🔍 Current Model Input"
        )

        display_features = X_latest.T

        display_features.columns = [
            "Value"
        ]

        st.dataframe(
            display_features
        )


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as e:

    st.error(
        "❌ Something went wrong while processing the stock data."
    )

    st.exception(e)
