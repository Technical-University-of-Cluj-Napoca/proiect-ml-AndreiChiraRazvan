import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
import shap

st.set_page_config(page_title="Regresie", page_icon="🏠", layout="wide")

# ─── DATA & MODEL TRAINING (cached) ───
@st.cache_resource
def load_and_train_regression():
    df = pd.read_csv('regression/house_price_regression_dataset.csv')
    features = [c for c in df.columns if c != 'House_Price']
    X = df[features]
    y = df['House_Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    scaled_models = {'SVR', 'KNN'}

    model_defs = {
        'Linear Regression':  LinearRegression(),
        'Decision Tree':      DecisionTreeRegressor(random_state=42),
        'Random Forest':      RandomForestRegressor(random_state=42, n_jobs=1),
        'SVR':                SVR(),
        'KNN':                KNeighborsRegressor(),
        'Gaussian Process':   GaussianProcessRegressor(random_state=42),
        'XGBoost':            XGBRegressor(random_state=42, verbosity=0),
        'CatBoost':           CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),
        'Explainable Boosting': ExplainableBoostingRegressor(random_state=42),
    }

    results = []
    trained = {}
    for name, model in model_defs.items():
        Xtr = X_train_sc if name in scaled_models else X_train
        Xte = X_test_sc if name in scaled_models else X_test
        model.fit(Xtr, y_train)
        trained[name] = model
        yp = model.predict(Xte)
        results.append({
            'Model': name,
            'R²': r2_score(y_test, yp),
            'RMSE': np.sqrt(mean_squared_error(y_test, yp)),
            'MAE': mean_absolute_error(y_test, yp),
        })

    results_df = pd.DataFrame(results).sort_values('R²', ascending=False).reset_index(drop=True)
    results_df.index += 1
    top5 = results_df.head(5)['Model'].tolist()

    return {
        'df': df, 'features': features,
        'X_train': X_train, 'X_test': X_test,
        'X_train_sc': X_train_sc, 'X_test_sc': X_test_sc,
        'y_train': y_train, 'y_test': y_test,
        'scaler': scaler, 'scaled_models': scaled_models,
        'trained': trained, 'results_df': results_df, 'top5': top5,
    }

data = load_and_train_regression()

# ─── PAGE CONTENT ───
st.title("🏠 Regresie — Predicția Prețului Locuințelor")

# ─── 1. PROBLEM DESCRIPTION ───
with st.expander("📋 1. Prezentarea Problemei", expanded=False):
    st.markdown("""
    **Problemă**: Regresie supervisată — predicția prețului unei locuințe.

    **Dataset**: `house_price_regression_dataset.csv` — 1000 observații, 7 caracteristici numerice.

    **Target**: `House_Price` — prețul de vânzare (USD).

    **Caracteristici**: Square_Footage, Num_Bedrooms, Num_Bathrooms, Year_Built,
    Lot_Size, Garage_Size, Neighborhood_Quality.
    """)

# ─── 2. EDA ───
with st.expander("📊 2. Analiza Exploratorie (EDA)", expanded=False):
    df = data['df']

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        df['House_Price'].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
        ax.set_title('Distribuția House_Price')
        ax.set_xlabel('Preț')
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    annot_kws={'size': 8}, vmin=-1, vmax=1)
        ax.set_title('Matricea de corelație')
        st.pyplot(fig)
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, feat in enumerate(['Square_Footage', 'Num_Bedrooms', 'Neighborhood_Quality']):
        axes[i].scatter(df[feat], df['House_Price'], alpha=0.3, s=10, c='steelblue')
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel('House_Price')
        axes[i].set_title(f'{feat} vs Preț')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── 3. MODEL COMPARISON ───
st.subheader("🏆 3. Compararea celor 9 Modele de Bază")
st.dataframe(data['results_df'], use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))
sorted_df = data['results_df'].sort_values('R²', ascending=True)
colors = ['gold' if m in data['top5'] else 'lightgray' for m in sorted_df['Model']]
ax.barh(sorted_df['Model'], sorted_df['R²'], color=colors)
ax.set_xlabel('R²')
ax.set_title('R² per Model (🟡 = Top 5)')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info(f"**Top 5 modele selectate**: {', '.join(data['top5'])}")

# ─── 4. MODEL SELECTION & PREDICTION ───
st.subheader("🎯 4. Selectează un Model și Realizează o Predicție")

selected_model = st.selectbox("Alege modelul:", data['top5'])

st.markdown("**Introdu valorile caracteristicilor:**")
col1, col2, col3 = st.columns(3)
input_vals = {}
feats = data['features']
df = data['df']
with col1:
    input_vals['Square_Footage'] = st.number_input("Square_Footage", int(df['Square_Footage'].min()),
                                                    int(df['Square_Footage'].max()), int(df['Square_Footage'].median()))
    input_vals['Num_Bedrooms'] = st.slider("Num_Bedrooms", int(df['Num_Bedrooms'].min()),
                                           int(df['Num_Bedrooms'].max()), int(df['Num_Bedrooms'].median()))
with col2:
    input_vals['Num_Bathrooms'] = st.slider("Num_Bathrooms", int(df['Num_Bathrooms'].min()),
                                            int(df['Num_Bathrooms'].max()), int(df['Num_Bathrooms'].median()))
    input_vals['Year_Built'] = st.number_input("Year_Built", int(df['Year_Built'].min()),
                                               int(df['Year_Built'].max()), int(df['Year_Built'].median()))
with col3:
    input_vals['Lot_Size'] = st.number_input("Lot_Size", float(df['Lot_Size'].min()),
                                             float(df['Lot_Size'].max()), float(df['Lot_Size'].median()), format="%.2f")
    input_vals['Garage_Size'] = st.slider("Garage_Size", int(df['Garage_Size'].min()),
                                          int(df['Garage_Size'].max()), int(df['Garage_Size'].median()))
input_vals['Neighborhood_Quality'] = st.slider("Neighborhood_Quality", int(df['Neighborhood_Quality'].min()),
                                               int(df['Neighborhood_Quality'].max()), int(df['Neighborhood_Quality'].median()))

if st.button("🔮 Prezice Prețul", type="primary"):
    input_df = pd.DataFrame([input_vals])[feats]
    model = data['trained'][selected_model]
    if selected_model in data['scaled_models']:
        input_df = pd.DataFrame(data['scaler'].transform(input_df), columns=feats)
    pred = model.predict(input_df)[0]
    st.success(f"### 💰 Preț estimat: **${pred:,.2f}**")

# ─── 5. METRICS FOR SELECTED MODEL ───
st.subheader(f"📈 5. Metrici — {selected_model}")
model = data['trained'][selected_model]
Xte = data['X_test_sc'] if selected_model in data['scaled_models'] else data['X_test']
y_pred = model.predict(Xte)
y_test = data['y_test']

mc1, mc2, mc3 = st.columns(3)
mc1.metric("R²", f"{r2_score(y_test, y_pred):.4f}")
mc2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
mc3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, c='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7)
    ax.set_xlabel('Valori reale')
    ax.set_ylabel('Predicții')
    ax.set_title('Real vs Predicție')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(residuals, bins=25, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title('Distribuția Reziduurilor')
    ax.set_xlabel('Reziduu')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── 6. LEARNING CURVES ───
st.subheader(f"📉 6. Curba de Învățare — {selected_model}")
with st.spinner("Se calculează curba de învățare..."):
    Xtr = data['X_train_sc'] if selected_model in data['scaled_models'] else data['X_train']
    train_sizes, train_scores, val_scores = learning_curve(
        model, Xtr, data['y_train'], cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    ax.plot(train_sizes, val_scores.mean(axis=1), 's-', label='Validare')
    ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                     train_scores.mean(1)+train_scores.std(1), alpha=0.1)
    ax.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                     val_scores.mean(1)+val_scores.std(1), alpha=0.1)
    ax.set_xlabel('Dimensiune set antrenare')
    ax.set_ylabel('R²')
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
            explainer = shap.KernelExplainer(model.predict, bg)

        sv = explainer(Xte_shap.iloc[:200]) if selected_model not in ['Gaussian Process'] else None

        if sv is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Bar Plot — Importanța globală**")
                fig, ax = plt.subplots(figsize=(6, 5))
                shap.plots.bar(sv, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Summary Plot**")
                fig, ax = plt.subplots(figsize=(6, 5))
                shap.plots.beeswarm(sv, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("**Waterfall — prima observație**")
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    except Exception as e:
        st.warning(f"SHAP nu este disponibil pentru {selected_model}: {e}")
