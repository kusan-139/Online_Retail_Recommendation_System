#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Retail Recommendation System — Final Auto-run Script
Features:
- Popularity-first analysis (Global / Country / Month)
- Item-Item Collaborative Filtering (CF)
- Frequently-Bought-Together (FBT)
- Presentation-ready Seaborn charts (readable labels, wrapped/truncated)
- Combined dashboard saved with date in filename
- Interactive plotting (works in VS Code and Colab)
- Outputs: CSVs + PNGs
Author: Kusan Chakraborty
"""
import os
import json
import textwrap
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
DATA_FILE = "OnlineRetail(1).xlsx"  # place this file in same folder
OUTPUT_PREFIX = "retail_outputs"
TOPN = 20  # change top-N globally here
TOP_ITEMS_CF = 5000  # cap for CF to control memory
TRUNCATE_LEN = 28  # max characters before truncation/wrapping
DASHBOARD_TOP_COUNTRIES = 3  # countries to show on dashboard country subplot
# ----------------------------------------------

# Enable interactive mode so plots show live in VS Code and remain until closed
plt.ion()

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ------------------- Unified label helpers -------------------
def shorten_and_wrap(text, max_len=TRUNCATE_LEN, max_lines=3):
    """Return text wrapped into lines of width max_len.
       Limits to max_lines and adds '...' if truncated."""
    if pd.isna(text):
        return ""
    s = str(text)
    if len(s) <= max_len:
        return s
    # wrap into lines
    lines = textwrap.wrap(s, width=max_len)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if len(lines[-1]) > max_len - 3:
            lines[-1] = lines[-1][:max_len - 3] + "..."
        else:
            lines[-1] = lines[-1] + "..."
    return "\n".join(lines)

def prepare_label_col(df, col="Description", max_len=TRUNCATE_LEN):
    """Add _LabelRaw and _Label columns to df for plotting labels."""
    df = df.copy()
    if col not in df.columns:
        df["_LabelRaw"] = ""
        df["_Label"] = ""
        return df
    df["_LabelRaw"] = df[col].fillna("").astype(str)
    df["_Label"] = df["_LabelRaw"].apply(lambda x: shorten_and_wrap(x, max_len=max_len))
    return df

# ------------------- Data Load & Clean -------------------
def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}. Put it in the same folder as this script.")
    print(f"Loading dataset from {path} ...")
    df = pd.read_excel(path, engine="openpyxl",
                       usecols=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df = df.dropna(subset=["CustomerID"]).copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]  # remove cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    return df

# ------------------- Descriptions & Popularity -------------------
def describe_data(df):
    return {
        "rows": len(df),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_items": int(df["StockCode"].nunique()),
        "countries": int(df["Country"].nunique()),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
    }

def _popularity_core(g):
    pop = (g.groupby("Description")
           .agg(buyers=("CustomerID", "nunique"),
                quantity=("Quantity", "sum"),
                revenue=("TotalPrice", "sum"))
           .sort_values(["buyers", "quantity", "revenue"], ascending=False))
    return pop

def popularity_global(df, topn=TOPN):
    return _popularity_core(df).head(topn).reset_index()

def popularity_by_country(df, topn=10):
    rows = []
    for country, g in df.groupby("Country"):
        pop = _popularity_core(g).head(topn).reset_index()
        pop.insert(0, "Country", country)
        rows.append(pop)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def popularity_by_month(df, topn=10):
    rows = []
    for (y, m), g in df.groupby(["Year", "Month"]):
        pop = _popularity_core(g).head(topn).reset_index()
        pop.insert(0, "Year", y); pop.insert(1, "Month", m)
        rows.append(pop)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ------------------- Item-Item CF -------------------
class ItemCF:
    def __init__(self, user_item_df, popularity_fallback):
        self.user_item = user_item_df
        X = self.user_item.to_numpy(dtype=np.float32)
        col_norms = np.sqrt((X**2).sum(axis=0, keepdims=True)) + 1e-9
        self.Xn = X / col_norms
        self.items = user_item_df.columns.tolist()
        self.idx_to_item = {i: self.items[i] for i in range(len(self.items))}
        self.item_to_idx = {v: k for k, v in self.idx_to_item.items()}
        self._sims = None
        self.popularity_fallback = popularity_fallback

    def _cosine_sims(self, chunk=800):
        if self._sims is not None:
            return self._sims
        Xn = self.Xn; n = Xn.shape[1]
        sims = np.zeros((n, n), dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims[start:end, :] = Xn[:, start:end].T @ Xn
        np.fill_diagonal(sims, 0.0)
        self._sims = sims
        return sims

    def recommend(self, user_id, topn=10, k_neighbors=50):
        if user_id not in self.user_item.index:
            return self.popularity_fallback[:topn]
        u = self.user_item.loc[user_id].to_numpy(dtype=np.float32)
        bought_idx = np.where(u > 0)[0]
        if len(bought_idx) == 0:
            return self.popularity_fallback[:topn]
        sims = self._cosine_sims()
        scores = np.zeros(sims.shape[0], dtype=np.float32)
        for bi in bought_idx:
            row = sims[bi]
            if k_neighbors and k_neighbors < len(row):
                topk = np.argpartition(-row, k_neighbors)[:k_neighbors]
                scores[topk] += row[topk]
            else:
                scores += row
        scores[bought_idx] = -np.inf
        top_idx = np.argpartition(-scores, topn)[:topn]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self.idx_to_item[i] for i in top_idx]

def build_user_item(df_train, top_items=TOP_ITEMS_CF):
    top_ids = (df_train.groupby("StockCode")["CustomerID"].nunique()
               .sort_values(ascending=False).head(top_items).index)
    t = df_train[df_train["StockCode"].isin(top_ids)].copy()
    ui = (t.assign(interaction=1)
          .drop_duplicates(subset=["CustomerID", "StockCode"])
          .pivot(index="CustomerID", columns="StockCode", values="interaction")
          .fillna(0).astype(np.float32))
    return ui

# ------------------- FBT -------------------
def build_fbt(df_train, whitelist=None):
    if whitelist is not None:
        df_train = df_train[df_train["StockCode"].isin(whitelist)].copy()
    co = defaultdict(Counter)
    for _, g in df_train.groupby("InvoiceNo"):
        items = list(set(g["StockCode"].tolist()))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                co[a][b] += 1
                co[b][a] += 1
    return co

def fbt_for_item(item_id, co_counts, popularity_index, topn=10):
    if item_id in co_counts and len(co_counts[item_id]) > 0:
        return [pid for pid, _ in co_counts[item_id].most_common(topn)]
    return [i for i in popularity_index if i != item_id][:topn]

# ------------------- Evaluation -------------------
def build_user_last_basket(df_part):
    if df_part.empty:
        return pd.DataFrame(columns=["CustomerID", "InvoiceNo", "StockCode"])
    last = df_part.sort_values("InvoiceDate").groupby("CustomerID").tail(1)
    inv_ids = last["InvoiceNo"].unique().tolist()
    inv_items = (df_part[df_part["InvoiceNo"].isin(inv_ids)][["InvoiceNo", "CustomerID", "StockCode"]]
                 .drop_duplicates().groupby(["CustomerID", "InvoiceNo"])["StockCode"].apply(list).reset_index())
    return inv_items

def hit_rate_at_k(eval_data, recommender, items_index, k=10, max_users=2000):
    if eval_data.empty:
        return float("nan")
    hits = 0; total = 0
    for _, row in eval_data.head(max_users).iterrows():
        truth = set([i for i in row["StockCode"] if i in items_index])
        if not truth: continue
        preds = recommender(row["CustomerID"], k)
        hits += int(len(truth.intersection(set(preds))) > 0)
        total += 1
    return hits / total if total > 0 else float("nan")

# ------------------- Plotting helpers -------------------
def save_and_show(fig, filename):
    ensure_dir(filename)
    fig.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.pause(0.1)

def plot_global(pop_global, out_png):
    pop = prepare_label_col(pop_global)
    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(data=pop.sort_values("buyers", ascending=True), x="buyers", y="_Label", ax=ax)
    ax.set_title(f"Top {len(pop_global)} Products Globally (by unique buyers)")
    ax.set_xlabel("Unique Buyers"); ax.set_ylabel("Product")
    save_and_show(fig, out_png)

def plot_country(pop_country, out_png, top_per_country=5):
    if pop_country.empty:
        return
    pop = pop_country.copy()
    top_countries = pop["Country"].value_counts().head(6).index.tolist()
    pop = pop[pop["Country"].isin(top_countries)]
    pop = pop.groupby("Country").head(top_per_country)
    pop = prepare_label_col(pop, col="Description", max_len=15)
    sns.set_palette("Set2")
    fig_height = max(8, 0.5 * len(pop))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.barplot(
        data=pop.sort_values("buyers", ascending=True),
        x="buyers", y="_Label", hue="Country",
        dodge=False, ax=ax
    )
    ax.set_title(f"Top {top_per_country} Products by Country (Top {len(top_countries)} Countries)")
    ax.set_xlabel("Unique Buyers")
    ax.set_ylabel("Product")
    plt.tight_layout()
    save_and_show(fig, out_png)

def plot_monthly(df, out_png):
    sns.set_palette("Set2")
    month_summary = df.groupby(["Year", "Month"])["CustomerID"].nunique().reset_index(name="unique_buyers")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.lineplot(data=month_summary, x="Month", y="unique_buyers", hue="Year", marker="o", ax=ax)
    ax.set_title("Unique Buyers per Month (by Year)")
    ax.set_xlabel("Month"); ax.set_ylabel("Unique Buyers")
    save_and_show(fig, out_png)

def plot_sample_recs(df_samples, out_png):
    if df_samples.empty:
        return
    row = df_samples.iloc[0]
    descs = row["Recommendations(Description)"]
    if isinstance(descs, str):
        try:
            descs = json.loads(descs.replace("'", '"'))
        except Exception:
            descs = [d.strip() for d in descs.strip("[]").split(",") if d.strip()]
    descs = [shorten_and_wrap(d, TRUNCATE_LEN) for d in descs]
    y = list(reversed(descs))
    x = list(range(1, len(y) + 1))
    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=x, y=y, orient="h", ax=ax)
    ax.set_title(f"Sample Recommendations for Customer {row['CustomerID']} (Top {len(x)})")
    ax.set_xlabel("Rank"); ax.set_ylabel("Recommended Items")
    save_and_show(fig, out_png)

def plot_fbt(df_fbt, out_png):
    if df_fbt.empty:
        return
    row = df_fbt.iloc[0]
    items = row["FBT(Description)"]
    if isinstance(items, str):
        try:
            items = json.loads(items.replace("'", '"'))
        except Exception:
            items = [d.strip() for d in items.strip("[]").split(",") if d.strip()]
    items = [shorten_and_wrap(d, TRUNCATE_LEN) for d in items]
    y = list(reversed(items))
    x = list(range(1, len(y) + 1))
    sns.set_palette("Set2")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=x, y=y, orient="h", ax=ax)
    ax.set_title("Frequently Bought Together (example for a top item)")
    ax.set_xlabel("Rank"); ax.set_ylabel("Items")
    save_and_show(fig, out_png)

def print_key_insights(pop_global, pop_country, df, df_fbt):
    print("\n===== KEY INSIGHTS =====")
    print("\nTop 5 Most Popular Products (Global):")
    top5 = pop_global.head(5)
    for idx, row in top5.iterrows():
        print(f"{idx + 1}. {row['Description']} - {row['buyers']} buyers")
    print("\nTop 3 Countries by Unique Buyers:")
    buyers_by_country = pop_country.groupby("Country")["buyers"].sum().nlargest(3)
    for i, (country, buyers) in enumerate(buyers_by_country.items(), start=1):
        print(f"{i}. {country} - {buyers} buyers")
    monthly_buyers = df.groupby(["Year", "Month"])["CustomerID"].nunique().reset_index()
    busiest = monthly_buyers.loc[monthly_buyers["CustomerID"].idxmax()]
    print(f"\nBusiest Month: {int(busiest['Month'])}/{int(busiest['Year'])} "
          f"with {busiest['CustomerID']} unique buyers")
    if not df_fbt.empty:
        print("\nTop 3 Frequently Bought Together Pairs (Example Items):")
        for i, row in df_fbt.head(3).iterrows():
            descs = row["FBT(Description)"]
            if isinstance(descs, str):
                try:
                    descs = json.loads(descs.replace("'", '"'))
                except:
                    descs = [d.strip() for d in descs.strip("[]").split(",") if d.strip()]
            print(f"{i + 1}. {', '.join(descs[:2])}")

# ------------------- Main -------------------
def main():
    df = load_and_clean(DATA_FILE)
    ds = describe_data(df)
    print("\nData summary:\n")
    for key, value in ds.items():
        print(f"{key}: {value}")
    pop_global = popularity_global(df, TOPN)
    pop_country = popularity_by_country(df, topn=max(10, TOPN // 2))
    pop_month = popularity_by_month(df, topn=max(10, TOPN // 2))
    pop_global.to_csv(f"{OUTPUT_PREFIX}_popularity_global.csv", index=False)
    pop_country.to_csv(f"{OUTPUT_PREFIX}_popularity_by_country.csv", index=False)
    pop_month.to_csv(f"{OUTPUT_PREFIX}_popularity_by_month.csv", index=False)
    print("\nSaved popularity CSVs.\n")
    plot_global(pop_global, f"{OUTPUT_PREFIX}_chart_global.png")
    plot_country(pop_country, f"{OUTPUT_PREFIX}_chart_country.png")
    plot_monthly(df, f"{OUTPUT_PREFIX}_chart_month.png")
    train = df[df["InvoiceDate"] <= (df["InvoiceDate"].max() - pd.Timedelta(days=30))].copy()
    if len(train) < 0.6 * len(df):
        s = df.sort_values("InvoiceDate"); split = int(0.8 * len(s))
        train = s.iloc[:split].copy()
    valid = df.drop(train.index)
    print("Train rows:", len(train), "\nValid rows:", len(valid))
    global_pop_items = train.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False).index.tolist()
    print("\nBuilding user-item matrix for CF (may take time on full dataset)...")
    ui = build_user_item(train, top_items=TOP_ITEMS_CF)
    items_index = ui.columns.tolist()
    model = ItemCF(ui, popularity_fallback=global_pop_items)
    # Build FBT
    co = build_fbt(train, whitelist=items_index)
    # Evaluation
    eval_data = build_user_last_basket(valid)
    def _rec(uid, k): return model.recommend(uid, topn=k)
    hr10 = hit_rate_at_k(eval_data, _rec, items_index, k=10, max_users=2000)
    print(f"\nHit@10 (validation last-invoice): {hr10 * 100:.2f}%\n" if hr10 == hr10 else "Hit@10 (validation last-invoice): N/A")
    # Sample recommendations CSV
    name_map = train.drop_duplicates("StockCode")[["StockCode", "Description"]].set_index("StockCode")["Description"].to_dict()
    top_users = train.groupby("CustomerID")["StockCode"].nunique().sort_values(ascending=False).head(10).index.tolist()
    samples = []
    for u in top_users:
        recs = model.recommend(u, topn=TOPN)
        samples.append({
            "CustomerID": u,
            "Recommendations(StockCode)": recs,
            "Recommendations(Description)": [name_map.get(x, str(x)) for x in recs]
        })
    df_samples = pd.DataFrame(samples)
    df_samples.to_csv(f"{OUTPUT_PREFIX}_sample_user_recommendations.csv", index=False)
    print("Saved sample user recommendations CSV.")
    # FBT CSV
    fbt_rows = []
    for it in global_pop_items[:5]:
        recs = fbt_for_item(it, co, global_pop_items, topn=TOPN)
        fbt_rows.append({
            "Item(StockCode)": it,
            "FBT(StockCode)": recs,
            "FBT(Description)": [name_map.get(x, str(x)) for x in recs]
        })
    df_fbt = pd.DataFrame(fbt_rows)
    df_fbt.to_csv(f"{OUTPUT_PREFIX}_fbt_recommendations.csv", index=False)
    print("Saved FBT CSV.")
    # Plot sample recs and FBT
    plot_sample_recs(df_samples, f"{OUTPUT_PREFIX}_chart_sample_recs.png")
    plot_fbt(df_fbt, f"{OUTPUT_PREFIX}_chart_fbt.png")
    # After saving FBT CSV
    print_key_insights(pop_global, pop_country, df, df_fbt)
    print("\nAll outputs written. Files:")
    out_files = [
        f"{OUTPUT_PREFIX}_popularity_global.csv",
        f"{OUTPUT_PREFIX}_popularity_by_country.csv",
        f"{OUTPUT_PREFIX}_popularity_by_month.csv",
        f"{OUTPUT_PREFIX}_sample_user_recommendations.csv",
        f"{OUTPUT_PREFIX}_fbt_recommendations.csv",
        f"{OUTPUT_PREFIX}_chart_global.png",
        f"{OUTPUT_PREFIX}_chart_country.png",
        f"{OUTPUT_PREFIX}_chart_month.png",
        f"{OUTPUT_PREFIX}_chart_sample_recs.png",
        f"{OUTPUT_PREFIX}_chart_fbt.png",
    ]
    for f in out_files:
        print(" -", f)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ------------------------
# Streamlit App UI
# ------------------------
st.set_page_config(page_title="🛒 Online Retail Recommendation & Analysis", layout="wide")
st.title("🛒 Online Retail Recommendation & Analysis System")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload Online Retail Dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    # ---- Clean Data ----
    df = df.dropna(subset=["CustomerID", "Description"])
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month

    # ✅ Fix ArrowTypeError for StockCode & other object columns
    id_columns = ["StockCode", "InvoiceNo", "CustomerID"]
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ---- Sidebar Options ----

    st.sidebar.header("Select Analysis / Recommendation")
    option = st.sidebar.radio(
        "Choose a module:",
        [
            "Global Popularity",
            "Popularity by Country",
            "Popularity by Month",
            "Sample User Recommendations",
            "Frequently Bought Together",
            "Popularity-Based Recommendations",
            "Item-Based Collaborative Filtering"
        ]
    )

    TOPN = 20

    # ---- Helper: Popularity Core ----
    def _popularity_core(g):
        return (g.groupby("Description")
                  .agg(buyers=("CustomerID", "nunique"),
                       quantity=("Quantity", "sum"),
                       revenue=("TotalPrice", "sum"))
                  .sort_values(["buyers", "quantity", "revenue"], ascending=False))

    # ---- Global Popularity ----
    if option == "Global Popularity":
        st.subheader("🌍 Top Products (Global)")
        pop_global = _popularity_core(df).head(TOPN).reset_index()
        st.dataframe(pop_global)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(data=pop_global, x="buyers", y="Description", ax=ax)
        ax.set_title("Top Products by Unique Buyers")
        st.pyplot(fig)
    
# ---- Popularity by Country ----
    elif option == "Popularity by Country":
        st.subheader("🌍 Product Popularity by Country")

    # --- Filters (Slicers) ---
        selected_country = st.multiselect(
        "🌐 Select Country:",
        options=df["Country"].unique(),
        default=["France", "Germany", "Spain"]   # ✅ only these 3 selected by default
        )

        selected_category = st.multiselect(
        "📦 Select Category:",
        options=df["Description"].unique(),
        default=df["Description"].unique()[:20]  # show first 20 by default
     )

    # --- Filter DataFrame based on slicers ---
        filtered_df = df[df["Country"].isin(selected_country)]
        filtered_df = filtered_df[filtered_df["Description"].isin(selected_category)]

    # Group by Country & Description
        country_counts = (
        filtered_df.groupby(["Country", "Description"])["CustomerID"]
        .nunique()
        .reset_index()
        )
        country_counts.rename(columns={"CustomerID": "buyers"}, inplace=True)

    # 🔹 Keep only top 15 products per country
        top_products = (
        country_counts.sort_values(["Country", "buyers"], ascending=[True, False])
        .groupby("Country")
        .head(15)
        )

    # --- Plot with Plotly ---
        fig = px.bar(
        top_products,
        x="buyers",
        y="Description",
        color="Country",
        orientation="h",
        title="Top 15 Products by Country",
        labels={"buyers": "Number of Buyers", "Description": "Product Description"},
        height=700
        )

    # Improve readability
        fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        legend_title="Country",
        template="plotly_dark"
        )   

        st.plotly_chart(fig, use_container_width=True)


    # ---- Popularity by Month ----
    elif option == "Popularity by Month":
        st.subheader("📅 Monthly Buyer Trends")
        month_summary = df.groupby(["Year", "Month"])["CustomerID"].nunique().reset_index(name="unique_buyers")
        st.dataframe(month_summary)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.lineplot(data=month_summary, x="Month", y="unique_buyers", hue="Year", marker="o", ax=ax)
        ax.set_title("Unique Buyers per Month")
        st.pyplot(fig)

    # ---- Sample User Recommendations (User-CF) ----
    elif option == "Sample User Recommendations":
        st.subheader("🎯 Sample User Recommendations (User-CF)")

        top_ids = df.groupby("StockCode")["CustomerID"].nunique().sort_values(ascending=False).head(1000).index
        t = df[df["StockCode"].isin(top_ids)].copy()
        ui = (t.assign(interaction=1)
                .drop_duplicates(["CustomerID", "StockCode"])
                .pivot(index="CustomerID", columns="StockCode", values="interaction")
                .fillna(0).astype(np.float32))

        X = ui.to_numpy()
        norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-9
        sims = (X.T @ X) / (norms.T @ norms)

        items = ui.columns.tolist()
        idx_to_item = {i: items[i] for i in range(len(items))}

        def recommend(user_id, topn=10):
            if user_id not in ui.index:
                return []
            u = ui.loc[user_id].to_numpy()
            bought = np.where(u > 0)[0]
            scores = sims[:, bought].sum(axis=1)
            scores[bought] = -np.inf
            top_idx = np.argsort(-scores)[:topn]
            return [idx_to_item[i] for i in top_idx]

        sample_users = np.random.choice(ui.index, size=min(5, len(ui)), replace=False)
        recs = [{"CustomerID": u, "Recommendations": ", ".join(map(str, recommend(u, 10)))} 
        for u in sample_users]

        st.dataframe(pd.DataFrame(recs))


    # ---- Frequently Bought Together ----
    elif option == "Frequently Bought Together":
        st.subheader("🔗 Frequently Bought Together")

        co = defaultdict(Counter)
        for _, g in df.groupby("InvoiceNo"):
            items = list(set(g["StockCode"].tolist()))
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    co[items[i]][items[j]] += 1
                    co[items[j]][items[i]] += 1

        top_item = df["StockCode"].value_counts().idxmax()
        top_fbt = co[top_item].most_common(10)

        desc_map = df.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()
        fbt_df = pd.DataFrame({
            "Item": [desc_map.get(top_item, top_item)] * len(top_fbt),
            "FBT Item": [desc_map.get(i, i) for i, _ in top_fbt],
            "Count": [c for _, c in top_fbt]
        })
        st.dataframe(fbt_df)

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.barplot(data=fbt_df, x="Count", y="FBT Item", ax=ax)
        ax.set_title(f"Frequently Bought Together with {desc_map.get(top_item)}")
        st.pyplot(fig)

    # ---- Popularity-Based Recommendations ----
    elif option == "Popularity-Based Recommendations":
        st.subheader("🔥 Top 10 Most Popular Products")
        popular_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        st.bar_chart(popular_products)

    # ---- Item-Based Collaborative Filtering ----
    elif option == "Item-Based Collaborative Filtering":
        st.subheader("🛍 Item-Based Recommendations")

        item_user_matrix = (
            df.pivot_table(index="Description", columns="CustomerID", values="Quantity", fill_value=0)
        )

        item_similarity = cosine_similarity(item_user_matrix)
        item_similarity_df = pd.DataFrame(
            item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index
        )

        product_list = list(item_user_matrix.index)
        selected_product = st.selectbox("Select a product to get recommendations:", product_list)

        if selected_product:
            st.write(f"✅ Because you selected **{selected_product}**, we recommend:")
            recommendations = (
                item_similarity_df[selected_product]
                .sort_values(ascending=False)
                .head(6)
                .index.tolist()[1:]
            )
            for rec in recommendations:
                st.write(f"👉 {rec}")
