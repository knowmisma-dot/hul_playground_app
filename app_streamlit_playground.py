# HUL ERP Simulator + Product Playground — Integrated Streamlit App

"""
Full Streamlit app that includes:
1. Synthetic ERP transaction generator (20k rows) for Beauty & Wellbeing.
2. Aggregation to customer-level features (RFM + behavioral).
3. XGBoost training on adopter_of_new_launch target.
4. Product Playground UI to create virtual products and compute affinity:
   - Cosine affinity
   - XGBoost predicted adoption probability across customers
   - Tunable compatibility scoring (weights, top-k averaging, thresholds)
5. SHAP explainability (feature importance and per-customer force plots) for adoption model.
6. A/B test simulation: pick sample of customers/outlets, simulate expected conversions & sales.
7. Downloads for datasets, model artifacts, and visuals.

Run:
    pip install -r requirements.txt
    streamlit run app_streamlit_playground.py

requirements.txt (important packages):
    streamlit
    pandas
    numpy
    scikit-learn
    lightgbm
    xgboost
    shap
    matplotlib
    joblib

Notes:
- This file is intended as a working prototype. For production, split training/inference, add auth, persistent storage, monitoring, and unit tests.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import io
import os

# -------------------------
# Config / paths
# -------------------------
TRANSACTION_CSV = 'synthetic_erp_transactions.csv'
CUSTOMER_FEATURES_CSV = 'customer_features.csv'
XGB_MODEL_PATH = 'xgb_cust_model.joblib'
VECTORIZER_PATH = 'product_vectorizer.joblib'
SCALER_BEHAV_PATH = 'customer_behavior_scaler.joblib'

N_TRANSACTIONS = 20000

# -------------------------
# Helper: Generate or load synthetic ERP transactions
# -------------------------

def generate_or_load_transactions(force_generate=False, n_customers=20000):
    """
    Generates synthetic ERP transactions for testing the HUL Virtual Trials app.
    Returns:
        df_tx: Pandas DataFrame of transactions
        launch_skus: list of SKUs used in the simulation
    """
    
    # Check if CSV exists (skip if not forcing)
    if not force_generate:
        try:
            df_tx = pd.read_csv("synthetic_transactions.csv")
            launch_skus = df_tx['sku'].unique().tolist()
            return df_tx, launch_skus
        except FileNotFoundError:
            pass

    # -------------------
    # Generate customers
    # -------------------
    customer_ids = [f"CUST{i:05d}" for i in range(1, n_customers + 1)]
    
    # -------------------
    # Define SKUs / products
    # -------------------
    sku_list = [
        {"sku": "DOVE_SHAMPOO_180ML", "price": 149, "category": "Haircare"},
        {"sku": "DOVE_SHAMPOO_360ML", "price": 249, "category": "Haircare"},
        {"sku": "LUX_SOAP_75G", "price": 45, "category": "Bath"},
        {"sku": "POND'S_CREAM_50G", "price": 99, "category": "Skincare"},
    ]
    
    launch_skus = [s["sku"] for s in sku_list]
    
    # -------------------
    # Channels
    # -------------------
    channels = ["Kirana", "Modern Trade", "Online", "Wholesale", "Direct"]
    
    # -------------------
    # Generate transactions
    # -------------------
    records = []
    start_date = datetime.today() - timedelta(days=365)
    
    for cust in customer_ids:
        n_tx = np.random.poisson(5)  # avg 5 transactions per customer
        for _ in range(n_tx):
            sku = np.random.choice(sku_list)
            date = start_date + timedelta(days=int(np.random.exponential(30)))
            units = int(np.random.choice([1,1,1,2,3], p=[0.6,0.1,0.1,0.15,0.05]))
            discount = round(np.random.choice([0,5,10,15,20], p=[0.5,0.2,0.15,0.1,0.05]), 2)
            price_per_unit = sku['price']
            net = round(units * price_per_unit * (1 - discount/100), 2)
            channel = np.random.choice(channels, p=[0.25,0.35,0.2,0.15,0.05])
            
            records.append({
                "customer_id": cust,
                "sku": sku['sku'],
                "category": sku['category'],
                "date": date.strftime("%Y-%m-%d"),
                "units": units,
                "discount": discount,
                "net_amount": net,
                "channel": channel
            })
    
    # -------------------
    # Build DataFrame
    # -------------------
    df_tx = pd.DataFrame(records)
    
    # Save for later reuse
    df_tx.to_csv("synthetic_transactions.csv", index=False)
    
    return df_tx, launch_skus

# -------------------------
# Aggregate to customer-level features
# -------------------------

def aggregate_customer_features(df):
    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Reference date for recency calculation
    reference_date = df['date'].max() + pd.Timedelta(days=1)

    # Recency
    recency = df.groupby('customer_id')['date'].max().reset_index()
    recency['recency_days'] = (reference_date - recency['date']).dt.days

    # Frequency
    frequency = df.groupby('customer_id').size().reset_index(name='frequency_tx')

    # Monetary
    monetary = df.groupby('customer_id')['net_sales_value'].agg(
        total_spend='sum', avg_ticket='mean'
    ).reset_index()

    # More metrics
    more = df.groupby('customer_id').agg(
        avg_discount=('discount_pct', 'mean'),
        avg_units=('units_sold', 'mean')
    ).reset_index()

    # Preferred channel
    pref_channel = (
        df.groupby(['customer_id', 'channel'])
        .size()
        .reset_index(name='cnt')
        .sort_values(['customer_id', 'cnt'], ascending=[True, False])
        .drop_duplicates('customer_id')
    )
    pref_channel = pref_channel[['customer_id', 'channel']].rename(
        columns={'channel': 'preferred_channel'}
    )

    # Top category
    top_cat = (
        df.groupby(['customer_id', 'category'])
        .size()
        .reset_index(name='cnt')
        .sort_values(['customer_id', 'cnt'], ascending=[True, False])
        .drop_duplicates('customer_id')
    )
    top_cat = top_cat[['customer_id', 'category']].rename(
        columns={'category': 'top_category'}
    )

    # Return rate
    ret = df.groupby('customer_id').apply(
        lambda x: (x['invoice_status'] == 'Returned').sum() / len(x)
    ).reset_index(name='return_rate')

    # Adopters of new launch (last 120 days)
    recent_cutoff = df['date'].max() - pd.Timedelta(days=120)
    adopters = (
        df[(df['is_launch_sku'] == 1) & (df['date'] >= recent_cutoff)]
        .groupby('customer_id')
        .size()
        .reset_index(name='adopt_count')
    )
    adopters['adopter_of_new_launch'] = 1

    # Merge all features
    parts = [recency[['customer_id', 'recency_days']], frequency, monetary, more, pref_channel, top_cat, ret]
    cust = parts[0]
    for p in parts[1:]:
        cust = cust.merge(p, on='customer_id', how='left')

    cust = cust.merge(
        adopters[['customer_id', 'adopter_of_new_launch']], 
        on='customer_id', how='left'
    )
    cust['adopter_of_new_launch'] = cust['adopter_of_new_launch'].fillna(0).astype(int)

    # Fill NaNs in numeric columns
    num_cols = ['frequency_tx', 'total_spend', 'avg_ticket', 'avg_discount', 'avg_units', 'return_rate']
    cust[num_cols] = cust[num_cols].fillna(0)

    # Save to CSV
    cust.to_csv(CUSTOMER_FEATURES_CSV, index=False)

    return cust


# -------------------------
# Product vectorizer & customer vectors
# -------------------------
PRODUCT_SCHEMA = {
    'category': ['skincare','haircare','oral_care','fragrance','personal_wash'],
    'sub_category': ['moisturizer','serum','cleanser','shampoo','conditioner','toothpaste','perfume','soap'],
    'tier': ['economy','standard','premium'],
    'channel_focus': ['Modern Trade','General Trade','E-commerce','Pharmacy','Salon']
}

def build_product_vectorizer():
    categorical_cols = ['category','sub_category','tier','channel_focus']
    numeric_cols = ['price','pack_size']
    cat_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    num_pipe = Pipeline([('scaler', StandardScaler())])
    vectorizer = ColumnTransformer([('cat', cat_pipe, categorical_cols), ('num', num_pipe, numeric_cols)], remainder='drop')
    # fit on a mock sample that covers combinations
    sample = []
    for cat in PRODUCT_SCHEMA['category']:
        for sub in PRODUCT_SCHEMA['sub_category'][:3]:
            for tier in PRODUCT_SCHEMA['tier']:
                for ch in PRODUCT_SCHEMA['channel_focus'][:2]:
                    sample.append({'category':cat,'sub_category':sub,'tier':tier,'channel_focus':ch,'price':200,'pack_size':100})
    sample_df = pd.DataFrame(sample)
    vectorizer.fit(sample_df)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return vectorizer

def build_customer_vectors(cust_df, vectorizer):
    cdf = cust_df.copy()
    cdf['category'] = cdf['top_category'].fillna('skincare')
    cdf['sub_category'] = cdf['top_category'].fillna('moisturizer')
    cdf['tier'] = 'standard'
    cdf['channel_focus'] = cdf['preferred_channel'].fillna('General Trade')
    cdf['price'] = cdf['avg_ticket'].fillna(100)
    cdf['pack_size'] = cdf['avg_units'].fillna(100)
    prod_like = cdf[['category','sub_category','tier','channel_focus','price','pack_size']]
    prod_vecs = vectorizer.transform(prod_like)
    behavior_cols = ['recency_days','frequency_tx','total_spend','avg_ticket','avg_discount','avg_units','return_rate']
    scaler = StandardScaler()
    beh = scaler.fit_transform(cdf[behavior_cols].fillna(0))
    cust_vectors = np.hstack([prod_vecs, beh])
    joblib.dump(scaler, SCALER_BEHAV_PATH)
    return cust_vectors, scaler

# -------------------------
# Train XGBoost model on customer vectors
# -------------------------

def train_xgb_customer(cust_df, vectorizer):
    X_vecs, scaler = build_customer_vectors(cust_df, vectorizer)
    y = cust_df['adopter_of_new_launch'].values
    X_train, X_test, y_train, y_test = train_test_split(X_vecs, y, test_size=0.25, stratify=y, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective':'binary:logistic','eval_metric':'auc','eta':0.05,'max_depth':5,'seed':42}
    bst = xgb.train(params, dtrain, num_boost_round=200, early_stopping_rounds=20, evals=[(dtrain,'train'),(dtest,'eval')], verbose_eval=False)
    joblib.dump(bst, XGB_MODEL_PATH)
    return bst, X_test, y_test

# -------------------------
# Compute affinity & compatibility
# -------------------------

def product_to_vector(product, vectorizer):
    pdf = pd.DataFrame([product])
    return vectorizer.transform(pdf)

def compute_affinity(product, cust_df, bst, vectorizer, weight_cos=0.4, weight_model=0.6, top_k=100):
    pvec = product_to_vector(product, vectorizer)
    cust_vecs, _ = build_customer_vectors(cust_df, vectorizer)
    # cosine on product-related slice
    prod_cols = pvec.shape[1]
    cust_prod_part = cust_vecs[:, :prod_cols]
    cos = cosine_similarity(cust_prod_part, pvec.reshape(1,-1)).flatten()
    # model probability: replace product part in cust_vecs with pvec
    X_model = cust_vecs.copy()
    X_model[:, :prod_cols] = np.repeat(pvec, X_model.shape[0], axis=0)
    dmat = xgb.DMatrix(X_model)
    prob = bst.predict(dmat)
    # compatibility score: weighted mean, and also top-k average
    mean_combo = weight_cos * np.mean(cos) + weight_model * np.mean(prob)
    topk_idx = np.argsort(prob)[-top_k:]
    topk_combo = weight_cos * np.mean(cos[topk_idx]) + weight_model * np.mean(prob[topk_idx])
    df_res = pd.DataFrame({'customer_id': cust_df['customer_id'].values, 'cosine_affinity': cos, 'pred_prob': prob})
    return {'compatibility_score_overall': float(mean_combo), 'compatibility_score_topk': float(topk_combo), 'results': df_res, 'top_customers': df_res.sort_values('pred_prob', ascending=False).head(top_k)}

def compatibility_verdict(score):
    if score >= 0.65:
        return 'High likelihood of success'
    elif score >= 0.5:
        return 'Moderate likelihood — consider pilots'
    else:
        return 'Low likelihood — redesign product/positioning'

# -------------------------
# SHAP explainability helpers
# -------------------------

def shap_explain(bst, X_sample, feature_names):
    # Use TreeExplainer
    expl = shap.TreeExplainer(bst)
    shap_values = expl.shap_values(X_sample)
    return expl, shap_values

# -------------------------
# A/B test simulation
# -------------------------

def simulate_ab_test(selected_customers, predicted_probs, baseline_conv_rate=0.02, n_runs=1000):
    # selected_customers: list of customer ids
    # predicted_probs: array aligned with selected_customers giving predicted conversion prob if targeted
    # baseline_conv_rate: expected conversion without targeting
    n = len(selected_customers)
    # Simulate conversions
    sim_target = np.random.binomial(1, predicted_probs, size=(n_runs, n)).sum(axis=1)
    sim_baseline = np.random.binomial(1, baseline_conv_rate, size=(n_runs, n)).sum(axis=1)
    # compute summary stats
    return {
        'target_mean_conv': float(sim_target.mean()),
        'target_std_conv': float(sim_target.std()),
        'baseline_mean_conv': float(sim_baseline.mean()),
        'baseline_std_conv': float(sim_baseline.std()),
        'uplift_mean': float(sim_target.mean() - sim_baseline.mean())
    }

# -------------------------
# Streamlit UI layout
# -------------------------

st.set_page_config(page_title='HUL Product Playground & Affinity', layout='wide')
st.title('HUL — Product Playground, Affinity Scoring, SHAP & A/B Simulator')

# Load or generate transactions
with st.sidebar.expander('Dataset'):
    if st.button('(Re)Generate synthetic ERP transactions (20k)'):
        df_tx, launch_skus = generate_or_load_transactions(force_generate=True)
        st.success('Generated and saved transactions.')
    else:
        df_tx, launch_skus = generate_or_load_transactions()
    st.write(f'{len(df_tx)} transactions loaded — {len(df_tx.customer_id.unique())} customers')
    if st.checkbox('Show sample transactions'):
        st.dataframe(df_tx.sample(200))

# Aggregate
if st.sidebar.button('Aggregate customer features'):
    with st.spinner('Aggregating...'):
        cust_df = aggregate_customer_features(df_tx)
    st.success(f'Aggregated to {len(cust_df)} customers')
else:
    if os.path.exists(CUSTOMER_FEATURES_CSV):
        cust_df = pd.read_csv(CUSTOMER_FEATURES_CSV)
    else:
        cust_df = aggregate_customer_features(df_tx)

st.sidebar.markdown('---')
st.sidebar.header('Product Playground')
# Product builder inputs
colp1, colp2, colp3 = st.sidebar.columns(3)
prod_category = colp1.selectbox('Category', PRODUCT_SCHEMA['category'])
prod_subcat = colp2.selectbox('Sub-category', PRODUCT_SCHEMA['sub_category'])
prod_tier = colp3.selectbox('Tier', PRODUCT_SCHEMA['tier'])
prod_channel = st.sidebar.selectbox('Channel focus', PRODUCT_SCHEMA['channel_focus'])
prod_price = st.sidebar.slider('Price (INR)', 50, 2000, 399)
prod_pack = st.sidebar.slider('Pack size (ml/g)', 30, 500, 100)

st.sidebar.markdown('---')
st.sidebar.header('Compatibility Tunables')
weight_cos = st.sidebar.slider('Weight: cosine similarity', 0.0, 1.0, 0.4)
weight_model = st.sidebar.slider('Weight: model probability', 0.0, 1.0, 0.6)
top_k = st.sidebar.number_input('Top-k customers to average (for topk score)', min_value=10, max_value=500, value=100)

st.sidebar.markdown('---')
st.sidebar.header('A/B Test Simulator')
baseline_rate = st.sidebar.slider('Baseline conversion rate (untargeted)', 0.0, 0.2, 0.02)
n_sim_runs = st.sidebar.number_input('Simulation runs', 100, 5000, 1000)

# Build vectorizer and model if not present
if os.path.exists(VECTORIZER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    vectorizer = build_product_vectorizer()

if os.path.exists(XGB_MODEL_PATH) and os.path.exists(SCALER_BEHAV_PATH):
    bst = joblib.load(XGB_MODEL_PATH)
else:
    st.info('Training XGBoost on aggregated customers (this runs once — may take ~30s)')
    bst, X_test_sample, y_test_sample = train_xgb_customer(cust_df, vectorizer)

# Product dict
product = {'category': prod_category, 'sub_category': prod_subcat, 'tier': prod_tier, 'channel_focus': prod_channel, 'price': prod_price, 'pack_size': prod_pack}

# Run affinity
if st.button('Compute affinity & compatibility'):
    with st.spinner('Computing affinity...'):
        res = compute_affinity(product, cust_df, bst, vectorizer, weight_cos=weight_cos, weight_model=weight_model, top_k=top_k)
    score_overall = res['compatibility_score_overall']
    score_topk = res['compatibility_score_topk']
    verdict = compatibility_verdict(score_topk)
    st.metric('Compatibility score (overall)', f'{score_overall:.3f}')
    st.metric(f'Compatibility score (top {top_k})', f'{score_topk:.3f}')
    st.success(verdict)

    # show distribution of pred_prob
    fig1, ax1 = plt.subplots()
    ax1.hist(res['results']['pred_prob'], bins=40)
    ax1.set_title('Predicted adoption probability distribution')
    st.pyplot(fig1)

    # show top customers
    st.markdown('### Top matching customers (by predicted probability)')
    st.dataframe(res['top_customers'].head(200))

    # SHAP explainability on top customers
    if st.checkbox('Show SHAP summary for model predictions (top customers)'):
        # Build X_model as in compute_affinity
        pvec = product_to_vector(product, vectorizer)
        cust_vecs, _ = build_customer_vectors(cust_df, vectorizer)
        X_model = cust_vecs.copy()
        pcols = pvec.shape[1]
        X_model[:, :pcols] = np.repeat(pvec, X_model.shape[0], axis=0)
        # sample subset for SHAP (to speed up)
        sample_idx = np.argsort(res['results']['pred_prob'])[ -min(500, len(res['results'])) : ]
        X_shap = X_model[sample_idx]
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_shap)
        # feature names: construct from vectorizer + behavior
        cat_names = list(vectorizer.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(['category','sub_category','tier','channel_focus']))
        num_names = list(vectorizer.transformers_[1][2]) if hasattr(vectorizer, 'transformers_') else ['price','pack_size']
        behavior_cols = ['recency_days','frequency_tx','total_spend','avg_ticket','avg_discount','avg_units','return_rate']
        feature_names = cat_names + ['price','pack_size'] + behavior_cols
        # summary plot
        fig_shap = shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
        st.pyplot(bbox_inches='tight')

    # A/B simulation
    if st.checkbox('Run A/B test simulation on top customers'):
        top_customers = res['top_customers']
        selected_n = st.slider('Select number of customers for test', 50, min(200, len(top_customers)), 200)
        sample_customers = top_customers.head(selected_n)
        sim = simulate_ab_test(sample_customers['customer_id'].tolist(), sample_customers['pred_prob'].values, baseline_conv_rate=baseline_rate, n_runs=n_sim_runs)
        st.write('A/B simulation results (per-sample conversions)')
        st.json(sim)

    # allow download of results
    csv_buf = io.StringIO()
    res['results'].to_csv(csv_buf, index=False)
    st.download_button('Download per-customer predictions', data=csv_buf.getvalue(), file_name='per_customer_predictions.csv')

st.markdown('---')
st.info('Tips: Tweak weights to increase or decrease emphasis on similarity vs predicted adoption. Use SHAP to understand what drives adoption in the target segment.')

# End of app
