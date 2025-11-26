import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuration & File Paths (Use in-memory data when possible for single file) ---
MODEL_PATH = "trained_model.pkl" # Save model in the same directory for simplicity
TEMP_DATA_PATH = "temp_data.csv" # Temporary file to pass data between steps

st.set_page_config(layout="wide", page_title="Bhutan Healthcare Data Science Workflow")

## --- 1. Data Generation (Simulating Data Acquisition) ---
def create_synthetic_data():
    """Creates a synthetic dataset to simulate Bhutan health indicators."""
    st.subheader("1️⃣ Data Generation (Synthetic)")
    
    dzongkhags = ['Thimphu', 'Paro', 'Chukha', 'Samdrup Jongkhar', 'Bumthang']
    years = np.arange(2018, 2023)
    
    data = []
    for year in years:
        for dzongkhag in dzongkhags:
            data.append({
                'Dzongkhag': dzongkhag,
                'Year': year,
                # Simulate lower malaria in highly urban Thimphu
                'Malaria_Cases': np.random.randint(0, 50) if dzongkhag != 'Thimphu' else np.random.randint(0, 5),
                'TB_Cases': np.random.randint(10, 100),
                'Hospital_Admissions': np.random.randint(500, 3000),
                'Vaccination_Rate': np.random.uniform(0.8, 0.99) * 100
            })

    df = pd.DataFrame(data)
    
    # Introduce missing values for preprocessing test
    df.loc[df.sample(frac=0.05, random_state=42).index, 'Malaria_Cases'] = np.nan
    df.loc[df.sample(frac=0.02, random_state=42).index, 'Hospital_Admissions'] = np.nan
    
    st.success(f"Generated synthetic data: {df.shape[0]} records, {df['Dzongkhag'].nunique()} Dzongkhags.")
    return df

## --- 2. Data Preprocessing & Feature Engineering ---
def preprocess_and_engineer(df_raw: pd.DataFrame):
    """Performs cleaning, feature engineering, and scaling."""
    st.subheader("2️⃣ Preprocessing & Feature Engineering")
    df = df_raw.copy()

    # --- Feature Engineering (Step 5.4) ---
    st.info("Applying Feature Engineering: Calculating Risk Score and Admission Rate.")
    df['Disease_Risk_Score'] = df['Malaria_Cases'] * 2 + df['TB_Cases'] * 1.5
    
    # Per Capita Proxy (using a synthetic population size for Dzongkhags)
    dzongkhag_pop = {
        'Thimphu': 130000, 'Paro': 45000, 'Chukha': 90000, 
        'Samdrup Jongkhar': 40000, 'Bumthang': 18000
    }
    df['Population'] = df['Dzongkhag'].map(dzongkhag_pop)
    df['Admission_Rate_Per_1000'] = (df['Hospital_Admissions'] / df['Population']) * 1000
    df_eda = df.drop(columns=['Population']) # Data for EDA (unscaled)

    # --- Preprocessing (Step 5.2) ---
    numerical_features = ['Malaria_Cases', 'TB_Cases', 'Hospital_Admissions', 'Vaccination_Rate', 
                          'Disease_Risk_Score', 'Admission_Rate_Per_1000']
    categorical_features = ['Dzongkhag']
    
    # 1. Numerical Pipeline (Imputation and Scaling)
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline (Encoding)
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Save the fitted pipeline/preprocessor for model inference later
    df_for_fit = df.drop(columns=['Population'])
    df_for_fit.dropna(subset=['Year'], inplace=True) # Ensure 'Year' is clean for pass-through

    X_processed = preprocessor.fit_transform(df_for_fit)
    
    # Get feature names
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    
    # Create the final scaled DataFrame
    df_final = pd.DataFrame(X_processed, columns=feature_names + ['Year'])
    df_final['Year'] = df_for_fit['Year'].values
    
    st.success(f"Preprocessing complete. Final scaled data shape: {df_final.shape}")
    
    # Save the preprocessor object for use in the prediction widget
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    return df_final, df_eda, preprocessor

## --- 3. Machine Learning Modeling ---
def train_model(df_final: pd.DataFrame):
    """Trains a Ridge Regression model to predict Admission Rate."""
    st.subheader("3️⃣ Machine Learning Modeling")
    
    TARGET = 'Admission_Rate_Per_1000'
    
    X = df_final.drop(columns=[TARGET, 'Year'])
    y = df_final[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("Training Ridge Regression Model (Predicting Admission Rate)...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    st.markdown(f"**Evaluation Metric:** Root Mean Squared Error (RMSE): **{rmse:.4f}**")
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    st.success(f"Model saved to {MODEL_PATH}")
    return model

## --- 4. Streamlit App/Interactive Visualization ---
def run_app(df_final, df_eda, model, preprocessor):
    """Renders the Streamlit dashboard components."""
    
    st.title("🇧🇹 Bhutan Healthcare Data Science Dashboard")
    st.markdown("---")
    
    # --- Data Prep for EDA ---
    df_plot = df_eda.copy()
    latest_year = df_plot['Year'].max()
    latest_data = df_plot[df_plot['Year'] == latest_year]

    # --- 4.1 Key Health Indicators Dashboard ---
    st.header("4️⃣ Key Health Indicators Overview")

    col1, col2, col3, col4 = st.columns(4)

    avg_risk_score = latest_data['Disease_Risk_Score'].mean()
    total_admissions = latest_data['Hospital_Admissions'].sum()
    max_vaccine = latest_data['Vaccination_Rate'].max()
    
    col1.metric("Latest Year", str(latest_year))
    col2.metric("Avg. Disease Risk Score", f"{avg_risk_score:.2f}")
    col3.metric("Total Admissions (Latest)", f"{int(total_admissions):,}")
    col4.metric("Max. Vaccination Rate", f"{max_vaccine:.1f}%")

    st.markdown("---")

    # --- 4.2 Time Trend & Dzongkhag Comparison (EDA) ---
    
    # Trend Chart
    st.header("5️⃣ Time Trend of Cases")
    trend_data = df_plot.groupby('Year')[['Malaria_Cases', 'TB_Cases']].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    trend_data.set_index('Year')[['Malaria_Cases', 'TB_Cases']].plot(kind='line', ax=ax, marker='o')
    ax.set_title('Malaria & TB Cases Trend Over Time')
    ax.set_ylabel('Total Cases')
    ax.grid(axis='y', linestyle='--')
    st.pyplot(fig)
    
    # Dzongkhag Comparison
    st.header("6️⃣ District-wise Comparison")
    
    indicator_choice = st.selectbox(
        'Select Indicator for Comparison:',
        ['Admission_Rate_Per_1000', 'Disease_Risk_Score', 'Vaccination_Rate']
    )
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=latest_data, 
        x='Dzongkhag', 
        y=indicator_choice, 
        ax=ax_bar, 
        palette='viridis'
    )
    ax_bar.set_title(f'{indicator_choice} by Dzongkhag ({latest_year})')
    ax_bar.set_ylabel(indicator_choice)
    st.pyplot(fig_bar)
    
    st.markdown("---")
    
    # --- 4.3 Model Prediction Widget ---
    st.header("7️⃣ Model Inference: Predict Admission Rate")
    st.caption("Use the trained model to predict the Admission Rate Per 1000 based on health metrics.")
    
    dzongkhag_options = df_plot['Dzongkhag'].unique()
    dzongkhag_pop = {
        'Thimphu': 130000, 'Paro': 45000, 'Chukha': 90000, 
        'Samdrup Jongkhar': 40000, 'Bumthang': 18000
    }
    
    with st.form("prediction_form"):
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            p_malaria = st.slider("Malaria Cases", 0, 100, 10)
            p_tb = st.slider("TB Cases", 0, 200, 50)
            p_admissions = st.slider("Hospital Admissions (Placeholder for rate calculation)", 100, 5000, 1500)
            
        with col_p2:
            p_vaccine = st.slider("Vaccination Rate (%)", 50.0, 100.0, 95.0)
            p_dzongkhag = st.selectbox("Dzongkhag", dzongkhag_options)
            
            # The Admission Rate is the target, so we calculate the input Admission_Rate_Per_1000
            # feature value *before* scaling, which the model uses to predict the scaled target.
            p_risk_score = p_malaria * 2 + p_tb * 1.5
            p_pop = dzongkhag_pop.get(p_dzongkhag, 10000)
            p_admission_rate_per_1000 = (p_admissions / p_pop) * 1000 # Use placeholder admissions to calculate the feature value

        submitted = st.form_submit_button("Predict Admission Rate")
        
        if submitted:
            # Create input DataFrame (must match the structure of the training data features before scaling)
            input_data_raw = pd.DataFrame([{
                'Malaria_Cases': p_malaria,
                'TB_Cases': p_tb,
                'Hospital_Admissions': p_admissions,
                'Vaccination_Rate': p_vaccine,
                'Disease_Risk_Score': p_risk_score,
                'Admission_Rate_Per_1000': p_admission_rate_per_1000, # This is the feature input, NOT the target
                'Dzongkhag': p_dzongkhag
            }])

            # Apply the saved preprocessor pipeline
            try:
                X_new_scaled = preprocessor.transform(input_data_raw)
                
                # Align the input array with the feature column names used during training
                feature_cols_all = df_final.drop(columns=['Admission_Rate_Per_1000', 'Year']).columns
                X_new_df = pd.DataFrame(X_new_scaled, columns=feature_cols_all)
                
                # Drop the target feature from the prediction input (it's the last numeric feature column)
                X_new_df.drop(columns=['Admission_Rate_Per_1000'], inplace=True) 

                prediction_scaled = model.predict(X_new_df)
                
                # Interpretation of the result
                st.success(f"**Predicted Scaled Admission Rate:** **{prediction_scaled[0]:.2f}**")
                st.info("Note: The model predicts a *scaled* rate. This indicates whether the conditions (Malaria, TB, Vaccine, etc.) lead to a high (positive scaled value) or low (negative scaled value) admission rate compared to the average.")
                
            except Exception as e:
                st.error(f"Prediction Error. Ensure the model/preprocessor objects are available. Error: {e}")

# --- Main Execution Flow ---

if __name__ == "__main__":
    
    # 0. Check and set up
    st.markdown("## ⚙️ Data Science Workflow Status")

    # 1. Data Generation
    df_raw = create_synthetic_data()
    
    # 2. Preprocessing & Feature Engineering
    if df_raw is not None:
        # Use st.cache_data to run this only once
        @st.cache_data
        def run_preprocess_and_engineer(data):
            return preprocess_and_engineer(data)
            
        df_final, df_eda, preprocessor = run_preprocess_and_engineer(df_raw)
        
    # 3. Model Training
    if 'df_final' in locals() and df_final is not None:
        # Use st.cache_resource to run training only once and cache the model/preprocessor
        @st.cache_resource
        def run_train_model(data):
            return train_model(data)
            
        model = run_train_model(df_final)
    
    # 4. Run App
    if 'model' in locals() and model is not None:
        st.markdown("---")
        run_app(df_final, df_eda, model, preprocessor)
    else:
        st.error("Workflow not fully completed. Check the status messages above.")
