import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

MODEL_PATH = 'models/logistic_regression_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
RAW_DATA_PATH = "road_safety_indicators_btn.csv"
TARGET_COLUMN = 'Is_Law'
FEATURE_COLUMNS = ['GHO_DISPLAY', 'DIMENSION_NAME']

def create_and_train_model(df_features, features, target):
    """Creates, trains, and saves the full ML pipeline."""
    st.info("Starting model training: Logistic Regression (Target: Is_Law)")
    
    # 1. Feature Selection and Split
    df_model = df_features[features + [target]].copy().dropna(subset=features + [target])
    
    if len(df_model) < 5 or df_model[target].nunique() < 2:
        st.error("Insufficient data for training after cleaning (need at least 2 classes or 5 rows).")
        return None
        
    X = df_model[features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 2. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='drop'
    )

    # 3. Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    # 4. Training
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model Trained! Test Accuracy: {accuracy:.2f}")

    # 6. Save Model and Preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    joblib.dump(model_pipeline.named_steps['preprocessor'], PREPROCESSOR_PATH)
    st.success(f"Model saved to {MODEL_PATH}")
    
    return model_pipeline

@st.cache_data
def load_and_clean_data(file_path):
    """Loads and performs initial cleaning on the WHO road safety data."""
    try:
        # Load the CSV, skipping the metadata row (assuming header is row 0, metadata is row 1)
        df = pd.read_csv(file_path, header=0)
        df = df.iloc[1:].reset_index(drop=True)
        
        # Clean up column names
        df.columns = df.columns.str.replace(r'[\(\)]', '', regex=True).str.replace(' ', '_')
        
        # Fill NaN for categorical columns used as features
        df['DIMENSION_NAME'].fillna('N/A', inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Check if '{file_path}' exists and is correctly formatted.")
        st.stop()


st.set_page_config(page_title="Bhutan Healthcare Analytics", layout="wide")

st.title("🇧🇹 Bhutan Road Safety/Public Health Analytics")
st.write("This application analyzes WHO Road Safety Indicators for Bhutan and predicts the existence of related national laws.")
st.markdown("---")


st.header("1. Load Healthcare Dataset")

# Directly load the uploaded file for a seamless runnable example
try:
    df_raw = load_and_clean_data(RAW_DATA_PATH)
    st.success(f"Dataset '{RAW_DATA_PATH}' loaded successfully.")
    st.write("Preview of cleaned raw dataset:")
    st.dataframe(df_raw.head())
except:
    st.warning(f"Could not load required file '{RAW_DATA_PATH}'.")
    st.stop()


st.header("2. Basic Data Cleaning")

df_clean = df_raw.copy()

# Custom Cleaning Step (based on initial inspection)
# Create a binary target: Is a law/mandate 'Yes'?
df_clean[TARGET_COLUMN] = (df_clean['Value'] == 'Yes').astype(int)

st.write("Cleaned dataset (Target `Is_Law` created):")
st.dataframe(df_clean[[*FEATURE_COLUMNS, TARGET_COLUMN]].head())


st.header("3. Exploratory Data Analysis (EDA)")

if st.checkbox("Show target variable distribution"):
    fig, ax = plt.subplots()
    df_clean[TARGET_COLUMN].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f"Distribution of Target: {TARGET_COLUMN} (1=Yes, 0=No/N/A)")
    ax.set_xlabel("Is_Law")
    ax.set_ylabel("Count")
    st.pyplot(fig)

if st.checkbox("Show Indicator counts"):
    st.bar_chart(df_clean['GHO_DISPLAY'].value_counts())


st.header("4. Feature Engineering")

st.write("Features selected: **Indicator Name (`GHO_DISPLAY`)** and **Dimension Name (`DIMENSION_NAME`)**.")

df_features = df_clean.copy()

st.write("Feature-engineered data preview:")
st.dataframe(df_features[[*FEATURE_COLUMNS, TARGET_COLUMN]].head())


st.header("5. Machine Learning Model")

mode = st.radio("Choose model mode:", ["Train New Model", "Load Existing Model"], index=0)

model = None
preprocessor = None

if mode == "Train New Model":
    model = create_and_train_model(df_features, FEATURE_COLUMNS, TARGET_COLUMN)
    if model:
        # Load the preprocessor that was saved during training for prediction
        preprocessor = joblib.load(PREPROCESSOR_PATH)
elif mode == "Load Existing Model":
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        st.success("Model and preprocessor loaded successfully from saved files.")
    except FileNotFoundError:
        st.warning("Model files not found. Please train a new model first.")


st.header("6. Prediction Interface")

if model is not None:
    st.subheader("Predict the Existence of a National Law")
    st.caption("Inputs are based on the available Road Safety Indicators.")

    # Get unique categories for inputs from the data
    indicator_options = df_features['GHO_DISPLAY'].unique().tolist()
    dimension_options = df_features['DIMENSION_NAME'].unique().tolist()
    
    with st.form("prediction_form"):
        # Prediction Input Widgets
        col1, col2 = st.columns(2)
        
        with col1:
            p_indicator = st.selectbox(
                "Road Safety Indicator:",
                indicator_options
            )
        with col2:
            p_dimension = st.selectbox(
                "Specific Dimension:",
                dimension_options
            )

        if st.form_submit_button("Predict Law Existence"):
            try:
                # 1. Create raw input DataFrame
                X_input_raw = pd.DataFrame([{
                    'GHO_DISPLAY': p_indicator,
                    'DIMENSION_NAME': p_dimension
                }])
                
                # --- FIX IS HERE ---
                # Do NOT manually transform. Pass raw data directly to the pipeline.
                
                # 2. Predict probability
                # The pipeline automatically runs the preprocessor, then the classifier.
                pred_proba = model.predict_proba(X_input_raw)[:, 1][0]
                
                # 3. Predict class (0 or 1)
                pred_class = (pred_proba > 0.5).astype(int)

                st.success(f"Prediction Complete!")
                
                st.markdown(f"**Predicted Outcome:** {'**Law Exists (Yes)**' if pred_class == 1 else '**Law Does Not Exist (No/N/A)**'}")
                st.info(f"Probability of Law Existence (Is_Law=1): **{pred_proba:.2f}**")
                
            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
else:
    st.warning("Model not available. Please train or upload a model in Section 5.")


st.header("7. Export Processed Data")

if st.button("Download cleaned dataset"):
    cleaned_csv = df_features.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, "cleaned_features_data.csv", "text/csv")
