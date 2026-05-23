import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation using K-Means Clustering")
st.write("This dashboard groups retail store customers based on their purchase history.")

df = pd.read_csv("Mall_Customers.csv")

st.subheader("📌 Dataset Preview")
st.dataframe(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("📊 Select Number of Clusters")

k = st.slider("Choose number of clusters", 2, 10, 5)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader("📋 Clustered Customer Data")
st.dataframe(df.head(20))

st.subheader("📈 Customer Segments")

fig, ax = plt.subplots(figsize=(8,5))
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments using K-Means")

st.pyplot(fig)

st.subheader("📌 Cluster Summary")

summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(summary)

st.success("K-Means clustering completed successfully!")