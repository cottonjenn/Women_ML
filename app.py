import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from collections import Counter

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Gender Happiness Gap",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 3rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
}

.main-header h1 {
    font-size: 2.8rem;
    margin: 0;
    color: #e8c5a0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #a0b4c8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.metric-card .value {
    font-size: 2rem;
    font-weight: 600;
    color: #0f3460;
    font-family: 'DM Serif Display', serif;
}

.metric-card .label {
    font-size: 0.8rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

.section-header {
    border-left: 4px solid #e8c5a0;
    padding-left: 1rem;
    margin: 2rem 0 1rem 0;
}

.section-header h2 {
    margin: 0;
    font-size: 1.6rem;
    color: #1a1a2e;
}

.section-header p {
    margin: 0.25rem 0 0 0;
    color: #6b7280;
    font-size: 0.9rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>The Gender Happiness Gap</h1>
    <p>Predicting & Profiling Women's Wellbeing in America · GSS 1972–2024</p>
</div>
""", unsafe_allow_html=True)

# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/ACTUAL_qol.csv")
    # Structural NA fill (marriage-related)
    for col in ['divorce', 'widowed', 'hapmar']:
        df[col] = df[col].fillna('never married')

    df['agewed_missing'] = df['agewed'].isna().astype(int)
    df['agewed'] = df['agewed'].fillna(0)

    df = pd.get_dummies(df, columns=['divorce', 'widowed', 'hapmar'], drop_first=True)

    # Engineered features (for RF)
    df['age_health_interaction'] = df['age'] * df['health_new']
    df['years_married'] = (df['year'] - df['agewed']).fillna(0)
    df['is_very_happy'] = (df['happy_new'] == 2).astype(int)

    return df

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    st.error("⚠️ Data file not found. Please place `ACTUAL_qol.csv` in the `data/` folder.")
    data_loaded = False
    st.stop()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Navigation")
section = st.sidebar.radio(
    "Jump to section",
    ["📋 Dataset Overview", "📈 EDA: Happiness Trends", "🤖 Supervised Learning", "🔵 K-Means Clustering"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Source**  
General Social Survey (GSS)  
NORC at the University of Chicago  
1972–2024 · NSF-funded  
[gss.norc.org](https://gss.norc.org)
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
if section == "📋 Dataset Overview":
    st.markdown('<div class="section-header"><h2>Dataset Overview</h2><p>Key facts about the GSS data used in this project</p></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="value">{len(df):,}</div><div class="label">Respondents</div></div>', unsafe_allow_html=True)
    with col2:
        n_years = df['year'].nunique()
        st.markdown(f'<div class="metric-card"><div class="value">{n_years}</div><div class="label">Survey Waves</div></div>', unsafe_allow_html=True)
    with col3:
        pct_female = (df['sex_new'] == 1).mean()
        st.markdown(f'<div class="metric-card"><div class="value">{pct_female:.0%}</div><div class="label">Female Respondents</div></div>', unsafe_allow_html=True)
    with col4:
        pct_happy = (df['happy_new'] == 2).mean()
        st.markdown(f'<div class="metric-card"><div class="value">{pct_happy:.0%}</div><div class="label">Very Happy</div></div>', unsafe_allow_html=True)

    st.markdown("### Key Variables")
    vars_info = pd.DataFrame({
        "Variable": ["happy_new", "sex_new", "health_new", "life_new", "hapmar", "age", "educ", "relig", "marital"],
        "Description": [
            "Self-reported happiness (0=not too happy, 1=pretty happy, 2=very happy)",
            "Sex (0=male, 1=female)",
            "Self-rated health (1=poor → 4=excellent)",
            "Life excitement (1=dull, 2=routine, 3=exciting)",
            "Marital happiness (encoded)",
            "Respondent age",
            "Years of education",
            "Religious affiliation (encoded)",
            "Marital status (encoded)",
        ],
        "Type": ["Target", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature"]
    })
    st.dataframe(vars_info, use_container_width=True, hide_index=True)

    st.markdown("### Target Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Not Too Happy (0)", "Pretty Happy (1)", "Very Happy (2)"]
    counts = [df['happy_new'].eq(0).sum(), df['happy_new'].eq(1).sum(), df['happy_new'].eq(2).sum()]
    colors = ["#e8c5a0", "#0f3460", "#1a1a2e"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{count:,}\n({count/len(df):.1%})', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Happiness Level Distribution (All Respondents)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.15)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "EDA: Happiness Trends":
    st.markdown('<div class="section-header"><h2>Exploratory Data Analysis</h2><p>Visualizing the gender happiness gap across five decades</p></div>', unsafe_allow_html=True)

    # Trend over time
    st.markdown("#### Mean Happiness Score by Sex & Year")
    trend = df.groupby(['year', 'sex_new'])['happy_new'].mean().reset_index()
    trend['sex_label'] = trend['sex_new'].map({0: 'Male', 1: 'Female'})

    fig, ax = plt.subplots(figsize=(12, 5))
    for sex, color, ls in [('Female', '#e8c5a0', '-'), ('Male', '#0f3460', '--')]:
        sub = trend[trend['sex_label'] == sex]
        ax.plot(sub['year'], sub['happy_new'], label=sex, color=color, linewidth=2.5, linestyle=ls)
    ax.set_xlabel("Survey Year")
    ax.set_ylabel("Mean Happiness Score (0–2)")
    ax.set_title("The Gender Happiness Gap Over Time (GSS 1972–2024)")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=2006, color='gray', linestyle=':', alpha=0.5, label='Gap narrows ~2006')
    ax.axvline(x=2021, color='red', linestyle=':', alpha=0.4, label='Mode shift 2021')
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
    **Key observations:**
    - Women consistently reported slightly **higher happiness** than men from 1972–2006 (the female happiness advantage)
    - The gap **narrowed** around 2006–2008 and the two groups have largely converged since
    - A sharp drop in 2002–2004 is likely a **methodological artifact** (instrument change), not a real trend
    - The post-2021 decline reflects both pandemic-era wellbeing effects and the **GSS mode shift** to web-based collection
    """)

    st.markdown("---")
    st.markdown("#### Happiness Distribution by Sex")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Not Too Happy', 'Pretty Happy', 'Very Happy']
    colors = ['#e8c5a0', '#a0b4c8', '#0f3460']
    for i, (sex_val, sex_label) in enumerate([(0, 'Male'), (1, 'Female')]):
        sub = df[df['sex_new'] == sex_val]['happy_new'].value_counts(normalize=True).sort_index()
        axes[i].bar(labels, [sub.get(j, 0) for j in range(3)], color=colors, edgecolor='white')
        axes[i].set_title(f"{sex_label} Respondents")
        axes[i].set_ylabel("Proportion")
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_ylim(0, 0.65)
        for j, v in enumerate([sub.get(k, 0) for k in range(3)]):
            axes[i].text(j, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("#### Happiness vs. Health & Life Excitement")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    health_happy = df.groupby('health_new')['happy_new'].mean()
    life_happy = df.groupby('life_new')['happy_new'].mean()
    axes[0].bar(['Poor', 'Fair', 'Good', 'Excellent'], health_happy.values, color='#0f3460', edgecolor='white')
    axes[0].set_title("Mean Happiness by Health Status")
    axes[0].set_ylabel("Mean Happiness (0–2)")
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    axes[1].bar(['Dull', 'Routine', 'Exciting'], life_happy.values, color='#e8c5a0', edgecolor='white')
    axes[1].set_title("Mean Happiness by Life Excitement")
    axes[1].set_ylabel("Mean Happiness (0–2)")
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Supervised Learning
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Supervised Learning":
    st.markdown('<div class="section-header"><h2>Supervised Learning</h2><p>Logistic Regression & Random Forest classifiers for predicting happiness</p></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Logistic Regression", "Random Forest"])

    # ── Logistic Regression ──────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Logistic Regression: Predicting Very Happy vs. Not Too Happy")
        st.markdown("We subset respondents to only **Not Too Happy (0)** and **Very Happy (2)**, using SMOTE to address class imbalance.")

        with st.spinner("Training Logistic Regression..."):
            categorical_structural = ['divorce', 'widowed', 'hapmar']
            lr_df = pd.read_csv("data/ACTUAL_qol.csv")
            for col in categorical_structural:
                lr_df[col] = lr_df[col].fillna('never married')
            lr_df['agewed_missing'] = lr_df['agewed'].isna().astype(int)
            lr_df['agewed'] = lr_df['agewed'].fillna(0)
            lr_df = pd.get_dummies(lr_df, columns=categorical_structural, drop_first=True)

            happy_subset = lr_df[lr_df['happy_new'].isin([0, 2])].copy()
            happy_subset['happy_binary'] = happy_subset['happy_new'].apply(lambda x: 0 if x == 0 else 1)

            X_lr = happy_subset.drop(columns=['happy_new', 'happy_binary', 'id'])
            y_lr = happy_subset['happy_binary']
            X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42, stratify=y_lr)

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train_lr, y_train_lr)

            model_lr = LogisticRegression(max_iter=1000, random_state=42)
            model_lr.fit(X_res, y_res)
            y_pred_lr = model_lr.predict(X_test_lr)

        report = classification_report(y_test_lr, y_pred_lr, output_dict=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="value">{report["accuracy"]:.2f}</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="value">{report["1"]["precision"]:.2f}</div><div class="label">Precision (Very Happy)</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="value">{report["1"]["recall"]:.2f}</div><div class="label">Recall (Very Happy)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="value">{report["1"]["f1-score"]:.2f}</div><div class="label">F1 Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Feature Importance (Odds Ratios)")
        odds_ratios = np.exp(model_lr.coef_[0])
        feat_df = pd.DataFrame({'Feature': X_lr.columns, 'Odds Ratio': odds_ratios}).sort_values('Odds Ratio', ascending=False)
        top_feats = feat_df.head(10)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = ['#0f3460' if v > 1 else '#e8c5a0' for v in top_feats['Odds Ratio']]
        ax.barh(top_feats['Feature'], top_feats['Odds Ratio'], color=colors_bar, edgecolor='white')
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Odds Ratio = 1 (no effect)')
        ax.set_xlabel("Odds Ratio (exponentiated coefficient)")
        ax.set_title("Top 10 Features by Odds Ratio")
        ax.invert_yaxis()
        ax.legend()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
        **Interpretation:** Features with odds ratio > 1 (blue) *increase* the odds of being Very Happy.
        Features with odds ratio < 1 (tan) *decrease* the odds.
        Marital happiness (`hapmar`) and life excitement (`life_new`) dominate — consistent with happiness literature.
        """)

    # ── Random Forest ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Random Forest: Predicting 'Very Happy' (Binary)")
        st.markdown("A Random Forest classifier using engineered features including `age × health` interaction and `years_married`.")

        with st.spinner("Training Random Forest (this may take a moment)..."):
            rf_df = pd.read_csv("data/ACTUAL_qol.csv")
            rf_df['age_health_interaction'] = rf_df['age'] * rf_df['health_new']
            rf_df['years_married'] = (rf_df['year'] - rf_df['agewed']).fillna(0)
            rf_df['is_very_happy'] = (rf_df['happy_new'] == 2).astype(int)

            drop_cols = ['happy_new', 'is_very_happy', 'id', 'divorce', 'widowed']
            drop_cols = [c for c in drop_cols if c in rf_df.columns]
            X_rf = rf_df.drop(columns=drop_cols).fillna(-1)
            y_rf = rf_df['is_very_happy']

            X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
            rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_rf, y_train_rf)
            rf_preds = rf_model.predict(X_test_rf)
            rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]

        rf_report = classification_report(y_test_rf, rf_preds, output_dict=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="value">{rf_report["accuracy"]:.2f}</div><div class="label">Accuracy</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="value">{rf_report["1"]["precision"]:.2f}</div><div class="label">Precision</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="value">{rf_report["1"]["recall"]:.2f}</div><div class="label">Recall</div></div>', unsafe_allow_html=True)
        with col4:
            fpr, tpr, _ = roc_curve(y_test_rf, rf_probs)
            roc_auc = auc(fpr, tpr)
            st.markdown(f'<div class="metric-card"><div class="value">{roc_auc:.2f}</div><div class="label">AUC-ROC</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test_rf, rf_preds)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Others', 'Very Happy'],
                        yticklabels=['Others', 'Very Happy'], ax=ax)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_right:
            st.markdown("#### ROC Curve")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color='#0f3460', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc='lower right')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Top Feature Importances (Permutation)")
        result = permutation_importance(rf_model, X_test_rf, y_test_rf, n_repeats=5, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({'Feature': X_rf.columns, 'Importance': result.importances_mean}).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='#0f3460', edgecolor='white')
        ax.set_xlabel("Mean Decrease in Accuracy (Permutation Importance)")
        ax.set_title("Top 10 Features — Random Forest")
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: K-Means Clustering
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "K-Means Clustering":
    st.markdown('<div class="section-header"><h2>K-Means Clustering</h2><p>Discovering latent wellbeing profiles in the GSS data</p></div>', unsafe_allow_html=True)
 
    with st.spinner("Running K-Means clustering..."):
        km_df = pd.read_csv("data/ACTUAL_qol.csv")
        km_df['age_health_interaction'] = km_df['age'] * km_df['health_new']
        km_df['years_married'] = (km_df['age'] - km_df['agewed']).fillna(0)
 
        features = ['happy_new', 'health_new', 'life_new', 'educ', 'age', 'age_health_interaction', 'years_married']
        X_km = km_df[features].dropna()
 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_km)
 
        kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
        X_km = X_km.copy()
        X_km['cluster'] = kmeans.fit_predict(X_scaled)
 
    st.markdown("#### Elbow Method — Choosing K")
    wcss = []
    for i in range(1, 11):
        km_temp = KMeans(n_clusters=i, init='k-means++', random_state=42)
        km_temp.fit(X_scaled)
        wcss.append(km_temp.inertia_)
 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 11), wcss, marker='o', color='#0f3460', linewidth=2)
    ax.axvline(x=4, color='#e8c5a0', linestyle='--', label='Chosen K=4')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    ax.set_title("Elbow Method for Optimal K")
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
    st.markdown("---")
    st.markdown("#### Cluster Profiles — Mean Feature Values")
    cluster_summary = X_km.groupby('cluster').mean().round(2)
    cluster_summary.index = [f"Cluster {i}" for i in cluster_summary.index]
 
    st.dataframe(cluster_summary, use_container_width=True)
 
    st.markdown("---")
    st.markdown("#### t-SNE Visualization — K=4 vs K=6 Clusters")
    st.markdown("t-SNE compresses the 7 feature dimensions down to 2D so we can visually inspect how well the clusters separate. We sample 3,000 respondents for speed.")
 
    with st.spinner("Running t-SNE (this takes ~20 seconds)..."):
        # Fit K=4 and K=6 on the full scaled data
        kmeans4 = KMeans(n_clusters=4, random_state=42).fit(X_scaled)
        kmeans6 = KMeans(n_clusters=6, random_state=42).fit(X_scaled)
 
        # Sample for t-SNE speed
        sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=min(3000, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_idx]
        labels4_sample = kmeans4.labels_[sample_idx]
        labels6_sample = kmeans6.labels_[sample_idx]
 
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X_sample)
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
 
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels4_sample,
                    palette='viridis', alpha=0.6, s=15, ax=axes[0], legend='full')
    axes[0].set_title("K=4 Clusters", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].get_legend().set_title("Cluster")
 
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels6_sample,
                    palette='tab10', alpha=0.6, s=15, ax=axes[1], legend='full')
    axes[1].set_title("K=6 Clusters", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].get_legend().set_title("Cluster")
 
    plt.suptitle("t-SNE: Visualizing Cluster Structure in 2D", fontsize=14, y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
 
    st.markdown("---")
    st.markdown("#### Cluster Sizes & Mean Happiness")
    palette = ['#0f3460', '#e8c5a0', '#a0b4c8', '#1a1a2e']
    col1, col2 = st.columns(2)
 
    with col1:
        cluster_counts = X_km['cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar([f'Cluster {i}' for i in cluster_counts.index], cluster_counts.values,
               color=palette, edgecolor='white')
        ax.set_ylabel("Number of Respondents")
        ax.set_title("Respondents per Cluster")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        cluster_happy = X_km.groupby('cluster')['happy_new'].mean()
        ax.bar([f'Cluster {i}' for i in cluster_happy.index], cluster_happy.values,
               color=palette, edgecolor='white')
        ax.set_ylabel("Mean Happiness Score")
        ax.set_title("Mean Happiness by Cluster")
        ax.set_ylim(0, 2.2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
 
    st.markdown("""
    **Key takeaways from clustering:**
    - The t-SNE plots show that K=4 produces cleaner, more separated clusters than K=6 — supporting our elbow choice
    - The 4 clusters reveal distinct wellbeing profiles not reducible to a single dimension
    - Higher-happiness clusters tend to co-occur with better health and more exciting lives (`life_new`)
    - Age and marital duration (`years_married`) also differentiate clusters — pointing to life-stage effects
    - These profiles set up the next phase: **anomaly detection** to find respondents whose happiness
      is statistically unexpected given their cluster profile
    """)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#9ca3af; font-size:0.85rem;">
GSS 1972–2024 · NORC at the University of Chicago · NSF-funded ·
Binary sex variable used (pre-2021 data); findings reflect recorded categories, not the full spectrum of gender identity.
</p>
""", unsafe_allow_html=True)
