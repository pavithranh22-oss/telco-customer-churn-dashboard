import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
LOW_T, HIGH_T = 0.40, 0.70  # risk tier thresholds

st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------
# 1. DATA LOADING AND CLEANING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates(subset="customerID")

    # TotalCharges has blank strings for brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].str.strip()
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m"]
    )
    return df


# ----------------------------------------------------------------------------
# 2. PREDICTION MODEL (logistic regression, no target leakage)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df):
    drop_cols = ["customerID", "Churn", "ChurnFlag", "TenureGroup"]
    X = df.drop(columns=drop_cols)
    y = df["ChurnFlag"]
    categorical = [c for c in X.columns if c not in NUMERIC]

    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    # Held-out test set for honest evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "cm": confusion_matrix(y_te, pred),
        "train_size": len(X_tr),
        "test_size": len(X_te),
    }

    # Out-of-fold probabilities so every customer's risk score is unseen-data based
    probs = cross_val_predict(pipe, X, y, cv=5, method="predict_proba")[:, 1]

    # Top drivers from model coefficients
    pipe.fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
    return metrics, probs, coefs


def risk_tier(p):
    return "High" if p >= HIGH_T else ("Medium" if p >= LOW_T else "Low")


def churn_rate_by(df, col):
    return df.groupby(col, observed=True)["ChurnFlag"].mean().mul(100).round(1).reset_index(name="Churn %")


# ----------------------------------------------------------------------------
# 3. LOAD DATA (local file or upload)
# ----------------------------------------------------------------------------
st.sidebar.title("📉 Telco Churn BI")
page = st.sidebar.radio(
    "Go to",
    ["Executive Overview", "Churn Drivers", "Customer Risk & Model", "Risk • Opportunity • Action"],
)

uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
if uploaded is not None:
    raw = load_data(uploaded)
elif os.path.exists(DATA_FILE):
    raw = load_data(DATA_FILE)
else:
    st.title("Telco Customer Churn Intelligence")
    st.info(f"Add *{DATA_FILE}* next to app.py or upload the CSV from the sidebar.")
    st.stop()

metrics, probs, coefs = train_model(raw)
df = raw.copy()
df["ChurnProbability"] = probs
df["RiskTier"] = df["ChurnProbability"].apply(risk_tier)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
internet = st.sidebar.multiselect(
    "Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique())
)
view = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet)]
if view.empty:
    st.warning("No customers match the filters.")
    st.stop()

# ----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------------------------------
if page == "Executive Overview":
    st.title("Executive Overview")
    st.caption("What is happening? Key performance indicators for the selected customers.")

    churned = view[view["ChurnFlag"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Churn rate", f"{view['ChurnFlag'].mean() * 100:.1f}%")
    c3.metric("Avg monthly charge", f"${view['MonthlyCharges'].mean():.2f}")
    c4.metric("Monthly revenue lost to churn", f"${churned['MonthlyCharges'].sum():,.0f}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(view, names="Churn", title="Churned vs retained", hole=0.5,
                     color="Churn", color_discrete_map={"Yes": "#e4572e", "No": "#2e86ab"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(churn_rate_by(view, "TenureGroup"), x="TenureGroup", y="Churn %",
                     title="Churn rate by tenure (trend over customer lifetime)", text="Churn %")
        st.plotly_chart(fig, use_container_width=True)

    worst = churn_rate_by(view, "Contract").sort_values("Churn %", ascending=False).iloc[0]
    st.success(
        f"*Key message:* {worst['Contract']} customers churn the most "
        f"({worst['Churn %']}%). Newer customers (0-12 months) are the most fragile."
    )

# ----------------------------------------------------------------------------
# PAGE 2: CHURN DRIVERS
# ----------------------------------------------------------------------------
elif page == "Churn Drivers":
    st.title("Churn Drivers")
    st.caption("Why is it happening? Churn rate by customer attribute.")

    options = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
               "PaperlessBilling", "SeniorCitizen", "Dependents", "Partner", "gender"]
    col = st.selectbox("Compare churn rate by", options)
    fig = px.bar(churn_rate_by(view, col).sort_values("Churn %"), x="Churn %", y=col,
                 orientation="h", text="Churn %", title=f"Churn rate by {col}")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.box(view, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly charges: churned vs retained")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(view, x="tenure", color="Churn", barmode="overlay", nbins=36,
                           title="Tenure distribution: churned vs retained")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strongest model drivers")
    top = pd.concat([coefs.head(8), coefs.tail(8)]).reset_index()
    top.columns = ["Feature", "Coefficient"]
    top["Effect"] = np.where(top["Coefficient"] > 0, "Raises churn risk", "Lowers churn risk")
    fig = px.bar(top, x="Coefficient", y="Feature", color="Effect", orientation="h",
                 color_discrete_map={"Raises churn risk": "#e4572e", "Lowers churn risk": "#2e86ab"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Association, not causation: the model finds patterns linked to churn, it does not prove why customers leave.")

# ----------------------------------------------------------------------------
# PAGE 3: CUSTOMER RISK AND MODEL
# ----------------------------------------------------------------------------
elif page == "Customer Risk & Model":
    st.title("Customer Risk & Model")
    st.caption("What might happen? Logistic regression churn model with risk scoring.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    st.caption(
        f"Trained on {metrics['train_size']:,} customers, tested on {metrics['test_size']:,} unseen customers. "
        "Recall = share of real churners the model catches; precision = how often a churn alert is right."
    )

    tn, fp, fn, tp = metrics["cm"].ravel()
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual: stayed", "Actual: churned"],
                      columns=["Predicted: stayed", "Predicted: churned"])
    st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix (test set)"),
                    use_container_width=True)

    tiers = view["RiskTier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    tiers.columns = ["Risk tier", "Customers"]
    st.plotly_chart(px.bar(tiers, x="Risk tier", y="Customers", color="Risk tier", text="Customers",
                           color_discrete_map={"High": "#e4572e", "Medium": "#f6ae2d", "Low": "#2e86ab"},
                           title=f"Risk tiers (Low < {LOW_T:.0%}, Medium {LOW_T:.0%}-{HIGH_T:.0%}, High >= {HIGH_T:.0%})"),
                    use_container_width=True)

    st.subheader("Top 20 highest-risk customers still active")
    active = view[view["ChurnFlag"] == 0].sort_values("ChurnProbability", ascending=False).head(20)
    show = active[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability", "RiskTier"]].copy()
    show["ChurnProbability"] = (show["ChurnProbability"] * 100).round(1)
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download full risk list (CSV)",
                       view[["customerID", "Contract", "tenure", "MonthlyCharges", "ChurnProbability", "RiskTier"]]
                       .to_csv(index=False).encode("utf-8"), "customer_risk_scores.csv", "text/csv")

# ----------------------------------------------------------------------------
# PAGE 4: RISK / OPPORTUNITY / ACTION
# ----------------------------------------------------------------------------
else:
    st.title("Risk • Opportunity • Action")
    st.caption("So what? Data-driven findings turned into management decisions.")

    by_contract = churn_rate_by(view, "Contract").sort_values("Churn %", ascending=False)
    by_tenure = churn_rate_by(view, "TenureGroup")
    by_internet = churn_rate_by(view, "InternetService").sort_values("Churn %", ascending=False)
    high_active = view[(view["RiskTier"] == "High") & (view["ChurnFlag"] == 0)]
    at_risk_rev = high_active["MonthlyCharges"].sum()
    best = by_contract.iloc[-1]

    st.error(
        f"*RISK:* {by_contract.iloc[0]['Contract']} customers churn at {by_contract.iloc[0]['Churn %']}%, "
        f"and {by_internet.iloc[0]['InternetService']} internet users churn at {by_internet.iloc[0]['Churn %']}%. "
        f"{len(high_active):,} active customers are high risk, putting about *${at_risk_rev:,.0f}/month* in revenue at risk."
    )
    st.success(
        f"*OPPORTUNITY:* {best['Contract']} customers churn only {best['Churn %']}%. "
        f"Moving customers from {by_contract.iloc[0]['Contract']} to longer contracts, and keeping them past their first year "
        f"(first-year churn is {by_tenure.iloc[0]['Churn %']}%), is the cheapest way to grow retention."
    )
    st.info(
        "*ACTION:*\n"
        "1. Launch a retention offer (discount or free upgrade) for high-risk, high-value customers.\n"
        "2. Run an onboarding programme for the first 12 months.\n"
        "3. Offer incentives to switch from month-to-month to 1- or 2-year contracts.\n"
        "4. Bundle tech support and online security, since customers without them churn more.\n"
        "5. Re-train the model monthly and track actual vs predicted churn."
    )
    st.caption(
        "Limits: the dataset has no customer feedback or competitor data, so the model shows associations only. "
        "Validate actions with A/B tests."
    )