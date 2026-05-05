import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import shap

st.set_page_config(page_title="Clasificare", page_icon="🌫️", layout="wide")

# ─── DATA & MODEL TRAINING (cached) ───
@st.cache_resource
def load_and_train_classification():
    df = pd.read_csv('classification/germany_air_quality_2014_2025.csv')

    # Create binary target
    df['Is_Unhealthy'] = (df['AQI_Bucket'] == 'Unhealthy for Sensitive Groups').astype(int)
    df.drop(['AQI', 'AQI_Bucket'], axis=1, inplace=True)

    # Parse date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)

    # Encode categoricals
    le_state = LabelEncoder()
    le_city = LabelEncoder()
    df['State'] = le_state.fit_transform(df['State'])
    df['City'] = le_city.fit_transform(df['City'])

    features = [c for c in df.columns if c != 'Is_Unhealthy']
    X = df[features]
    y = df['Is_Unhealthy']

    imbalance_ratio = (y == 0).sum() / max((y == 1).sum(), 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    scaled_models = {'SVM', 'KNN', 'Logistic Regression'}

    model_defs = {
        'Naive Bayes':        GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'),
        'Decision Tree':      DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest':      RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1),
        'SVM':                SVC(probability=True, random_state=42, class_weight='balanced'),
        'KNN':                KNeighborsClassifier(),
        'XGBoost':            XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss',
                                           scale_pos_weight=imbalance_ratio),
        'CatBoost':           CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False,
                                                 auto_class_weights='Balanced'),
        'Explainable Boosting': ExplainableBoostingClassifier(random_state=42),
    }

    results = []
    trained = {}
    for name, model in model_defs.items():
        Xtr = X_train_sc if name in scaled_models else X_train
        Xte = X_test_sc if name in scaled_models else X_test
        model.fit(Xtr, y_train)
        trained[name] = model
        yp = model.predict(Xte)
        ypr = model.predict_proba(Xte)[:, 1] if hasattr(model, 'predict_proba') else None
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp, zero_division=0),
            'Recall': recall_score(y_test, yp, zero_division=0),
            'F1': f1_score(y_test, yp, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, ypr) if ypr is not None else None,
        })

    results_df = pd.DataFrame(results).sort_values('F1', ascending=False).reset_index(drop=True)
    results_df.index += 1
    top5 = results_df.head(5)['Model'].tolist()

    # Keep raw df for EDA
    df_raw = pd.read_csv('classification/germany_air_quality_2014_2025.csv')

    return {
        'df_raw': df_raw, 'features': features,
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'scaled_models': scaled_models,
        'trained': trained, 'results_df': results_df, 'top5': top5,
        'le_state': le_state, 'le_city': le_city,
        'imbalance_ratio': imbalance_ratio,
    }

data = load_and_train_classification()

# ─── PAGE CONTENT ───
st.title("🌫️ Clasificare — Detectarea Aerului Periculos")

# ─── 1. PROBLEM DESCRIPTION ───
with st.expander("📋 1. Prezentarea Problemei", expanded=False):
    st.markdown(f"""
    **Problemă**: Clasificare binară pe un set de date **sever dezechilibrat** — detectarea zilelor
    cu aer periculos pentru grupurile sensibile din Germania.

    **Dataset**: `germany_air_quality_2014_2025.csv` — 10,512 observații.

    **Target**: `Is_Unhealthy` — **0** (Aer Acceptabil, ~99.5%) / **1** (Aer Periculos, ~0.5%)

    **Raport dezechilibru**: 1:{data['imbalance_ratio']:.0f}
    (doar 50 de zile periculoase din 10,512)

    **Metrici prioritare**: Recall și F1 pentru clasa 1 (nu Accuracy, care este înșelătoare).

    **Caracteristici**: Poluanți (PM2.5, PM10, NO2, CO, O3, SO2, Benzene, etc.) +
    factori de mediu (viteză vânt, umiditate, defrișare, emisii CO2).
    """)

# ─── 2. EDA ───
with st.expander("📊 2. Analiza Exploratorie (EDA)", expanded=False):
    df_raw = data['df_raw']

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        df_raw['AQI_Bucket'].value_counts().plot(kind='barh', ax=ax,
            color=['steelblue', 'steelblue', 'crimson'])
        ax.set_title('Distribuția AQI_Bucket')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df_raw['AQI'].hist(bins=40, ax=ax, color='steelblue', edgecolor='white')
        ax.axvline(x=100, color='red', linestyle='--', label='Prag Unhealthy')
        ax.set_title('Distribuția AQI')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    pollutants = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
                  'O3 (ug/m3)', 'CO (mg/m3)', 'SO2 (ug/m3)']
    df_raw['Is_Unhealthy'] = (df_raw['AQI_Bucket'] == 'Unhealthy for Sensitive Groups').astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i, col in enumerate(pollutants):
        ax = axes[i // 3][i % 3]
        for label, color in [(0, 'steelblue'), (1, 'crimson')]:
            subset = df_raw[df_raw['Is_Unhealthy'] == label][col]
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=f'Cls {label}', density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle('Distribuția poluanților per clasă', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── 3. MODEL COMPARISON ───
st.subheader("🏆 3. Compararea celor 9 Modele")
st.dataframe(data['results_df'], use_container_width=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sorted_df = data['results_df'].sort_values('F1', ascending=True)
colors = ['gold' if m in data['top5'] else 'lightgray' for m in sorted_df['Model']]
axes[0].barh(sorted_df['Model'], sorted_df['F1'], color=colors)
axes[0].set_xlabel('F1 (cls 1)')
axes[0].set_title('F1-Score (🟡 = Top 5)')

colors2 = ['gold' if m in data['top5'] else 'lightgray' for m in sorted_df['Model']]
axes[1].barh(sorted_df['Model'], sorted_df['Recall'], color=colors2)
axes[1].set_xlabel('Recall (cls 1)')
axes[1].set_title('Recall (🟡 = Top 5)')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(f"**Top 5 modele**: {', '.join(data['top5'])}")

# ─── 4. MODEL SELECTION & PREDICTION ───
st.subheader("🎯 4. Selectează un Model și Realizează o Predicție")

selected_model = st.selectbox("Alege modelul:", data['top5'])

st.markdown("**Introdu valorile poluanților și factorilor de mediu:**")
input_vals = {}
col1, col2, col3 = st.columns(3)

with col1:
    input_vals['PM2.5 (ug/m3)'] = st.number_input("PM2.5 (ug/m3)", 0.0, 100.0, 13.0, 0.5)
    input_vals['PM10 (ug/m3)'] = st.number_input("PM10 (ug/m3)", 0.0, 150.0, 22.0, 0.5)
    input_vals['NO (ug/m3)'] = st.number_input("NO (ug/m3)", 0.0, 50.0, 6.5, 0.5)
    input_vals['NO2 (ug/m3)'] = st.number_input("NO2 (ug/m3)", 0.0, 100.0, 16.0, 0.5)
    input_vals['NOx (ppb)'] = st.number_input("NOx (ppb)", 0.0, 60.0, 12.0, 0.5)
with col2:
    input_vals['NH3 (ug/m3)'] = st.number_input("NH3 (ug/m3)", 0.0, 25.0, 8.0, 0.5)
    input_vals['CO (mg/m3)'] = st.number_input("CO (mg/m3)", 0.0, 3.0, 0.33, 0.01)
    input_vals['SO2 (ug/m3)'] = st.number_input("SO2 (ug/m3)", 0.0, 20.0, 3.3, 0.5)
    input_vals['O3 (ug/m3)'] = st.number_input("O3 (ug/m3)", 0.0, 120.0, 45.0, 1.0)
    input_vals['Benzene (ug/m3)'] = st.number_input("Benzene (ug/m3)", 0.0, 2.0, 0.27, 0.01)
with col3:
    input_vals['Toluene (ug/m3)'] = st.number_input("Toluene (ug/m3)", 0.0, 5.0, 0.6, 0.01)
    input_vals['Xylene (ug/m3)'] = st.number_input("Xylene (ug/m3)", 0.0, 2.0, 0.28, 0.01)
    input_vals['Wind_Speed (km/h)'] = st.number_input("Wind Speed (km/h)", 0.0, 40.0, 14.0, 0.5)
    input_vals['Humidity (%)'] = st.slider("Humidity (%)", 20.0, 100.0, 65.0, 1.0)
    input_vals['Population_Density_per_SqKm'] = st.number_input("Pop. Density", 900.0, 1800.0, 1300.0, 10.0)

col_extra1, col_extra2 = st.columns(2)
with col_extra1:
    input_vals['Deforestation_Rate_%'] = st.number_input("Deforestation Rate %", 0.0, 1.0, 0.55, 0.01)
    input_vals['Industry_Growth_%'] = st.number_input("Industry Growth %", 1.0, 6.0, 3.5, 0.1)
    input_vals['CO2_Emission_MT'] = st.number_input("CO2 Emission (MT)", 0.0, 15.0, 4.0, 0.1)
with col_extra2:
    input_vals['State'] = st.number_input("State (encoded)", 0, 15, 5)
    input_vals['City'] = st.number_input("City (encoded)", 0, 20, 5)
    input_vals['Year'] = st.number_input("Year", 2014, 2025, 2023)
    input_vals['Month'] = st.slider("Month", 1, 12, 6)

if st.button("🔮 Clasifică", type="primary"):
    input_df = pd.DataFrame([input_vals])[data['features']]
    model = data['trained'][selected_model]
    if selected_model in data['scaled_models']:
        input_df = pd.DataFrame(data['scaler'].transform(input_df), columns=data['features'])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

    if pred == 1:
        st.error(f"### ⚠️ Aer PERICULOS pentru grupuri sensibile")
    else:
        st.success(f"### ✅ Aer ACCEPTABIL")

    if proba is not None:
        st.write(f"Probabilitate clasa 0 (Acceptabil): **{proba[0]:.4f}** | "
                 f"Clasa 1 (Periculos): **{proba[1]:.4f}**")

# ─── 5. METRICS ───
st.subheader(f"📈 5. Metrici — {selected_model}")
model = data['trained'][selected_model]
Xte = data['X_test_sc'] if selected_model in data['scaled_models'] else data['X_test']
y_pred = model.predict(Xte)
y_test = data['y_test']

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
mc2.metric("Precision (cls 1)", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
mc3.metric("Recall (cls 1)", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
mc4.metric("F1 (cls 1)", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
    display_labels=['Acceptabil', 'Periculos'], ax=ax, cmap='Blues')
ax.set_title(f'Matricea de Confuzie — {selected_model}')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─── 6. LEARNING CURVES ───
st.subheader(f"📉 6. Curba de Învățare — {selected_model}")
with st.spinner("Se calculează curba de învățare..."):
    Xtr = data['X_train_sc'] if selected_model in data['scaled_models'] else data['X_train']
    train_sizes, train_scores, val_scores = learning_curve(
        model, Xtr, data['y_train'], cv=5, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    ax.plot(train_sizes, val_scores.mean(axis=1), 's-', label='Validare')
    ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                     train_scores.mean(1)+train_scores.std(1), alpha=0.1)
    ax.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                     val_scores.mean(1)+val_scores.std(1), alpha=0.1)
    ax.set_xlabel('Dimensiune set antrenare')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Curba de învățare — {selected_model}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── 7. HYPERPARAMETERS ───
st.subheader(f"⚙️ 7. Hiperparametri — {selected_model}")
params = model.get_params() if hasattr(model, 'get_params') else {}
params_clean = {k: v for k, v in params.items() if v is not None and k not in ['verbose', 'verbosity']}
st.json(params_clean)

# ─── 8. SHAP ───
st.subheader(f"🔍 8. Explicabilitate SHAP — {selected_model}")
with st.spinner("Se calculează valorile SHAP..."):
    Xte_shap = data['X_test_sc'] if selected_model in data['scaled_models'] else data['X_test']

    try:
        if selected_model in ['Random Forest', 'Decision Tree', 'XGBoost', 'CatBoost', 'Explainable Boosting']:
            explainer = shap.TreeExplainer(model)
        else:
            bg = shap.sample(data['X_train_sc'] if selected_model in data['scaled_models'] else data['X_train'], 50)
            explainer = shap.KernelExplainer(model.predict_proba, bg)

        sv = explainer(Xte_shap.iloc[:200])

        # Handle binary classification SHAP (might be 3D)
        if sv.values.ndim == 3:
            sv_cls1 = shap.Explanation(
                values=sv.values[:, :, 1],
                base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
                data=sv.data, feature_names=sv.feature_names)
        else:
            sv_cls1 = sv

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Bar Plot — Importanța globală**")
            fig, ax = plt.subplots(figsize=(6, 5))
            shap.plots.bar(sv_cls1, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown("**Summary Plot**")
            fig, ax = plt.subplots(figsize=(6, 5))
            shap.plots.beeswarm(sv_cls1, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Local: find a True Positive if possible
        y_pred_shap = model.predict(Xte_shap.iloc[:200])
        y_test_shap = data['y_test'].iloc[:200].values
        tp_mask = (y_pred_shap == 1) & (y_test_shap == 1)
        tp_idx = np.where(tp_mask)[0]
        idx = tp_idx[0] if len(tp_idx) > 0 else 0
        label = "True Positive ✓" if len(tp_idx) > 0 else "Obs. 0"

        st.markdown(f"**Waterfall — {label}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(sv_cls1[idx], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP nu este disponibil pentru {selected_model}: {e}")
