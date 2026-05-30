import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


# Page title
st.set_page_config(
    page_title="Smart Fridge AI Dashboard",
    layout="wide"
)

st.title("🧠 Smart Fridge AI Dashboard")


# Database connection
conn = sqlite3.connect("smart_fridge.db")

# Load inventory data
query = "SELECT * FROM inventory"

df = pd.read_sql_query(query, conn)

conn.close()


# Show inventory table
st.subheader("📦 Inventory Data")

st.dataframe(df)


# Analytics
if not df.empty:

    # Freshness counts
    freshness_counts = (
        df["freshness"]
        .value_counts()
    )

    st.subheader("📊 Freshness Analytics")

    # Chart
    fig, ax = plt.subplots()

    freshness_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Freshness")

    ax.set_ylabel("Count")

    st.pyplot(fig)

else:

    st.warning("⚠️ No inventory data found.")