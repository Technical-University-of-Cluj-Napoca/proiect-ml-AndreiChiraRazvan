import streamlit as st

st.set_page_config(
    page_title="ML Pipeline — Regresie & Clasificare",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Machine Learning Pipeline")
st.markdown("### Compararea a 9 algoritmi pe două probleme diferite")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## 🏠 Regresie — Prețul Locuințelor
    **Dataset**: `house_price_regression_dataset.csv`
    - **1000** observații, **7** caracteristici numerice
    - **Target**: `House_Price`
    - **9 modele** antrenate și comparate
    - **Top 5** optimizate cu GridSearchCV / BayesSearchCV

    👈 Navighează din sidebar la **Regresie**
    """)

with col2:
    st.markdown("""
    ## 🌫️ Clasificare — Calitatea Aerului
    **Dataset**: `germany_air_quality_2014_2025.csv`
    - **10,512** observații, **21** caracteristici
    - **Target**: `Is_Unhealthy` (binar, dezechilibrat ~0.5%)
    - **9 modele** antrenate și comparate
    - **Top 5** optimizate cu accent pe **class imbalance**

    👈 Navighează din sidebar la **Clasificare**
    """)

st.divider()

st.markdown("""
### 📋 Pipeline ML complet
Fiecare pagină conține:
1. **Prezentarea problemei** și a setului de date
2. **EDA** — grafice exploratorii relevante
3. **Compararea a 9 modele** de bază cu metrici detaliate
4. **Selecție interactivă** — alege un model din Top 5
5. **Predicție live** — introdu valori și obține rezultatul
6. **Metrici și grafice** ale modelului selectat
7. **Curbe de învățare** — diagnoză overfitting/underfitting
8. **Hiperparametri** — valorile optimizate
9. **Analiza SHAP** — explicabilitatea predicției

---
*Proiect realizat cu Python, scikit-learn, XGBoost, CatBoost, SHAP și Streamlit.*
""")
