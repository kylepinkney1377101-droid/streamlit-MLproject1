import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Battery Material Predictor",
    page_icon="🔋",
    layout="centered"
)

# ─── Data Loading & Model Training ────────────────────────────────────────────
# Cache the model training so it doesn't retrain on every slider click
@st.cache_resource
def train_model():
    try:
        # Read the data directly from your CSV file
        df = pd.read_csv("dataset0.csv")
        
        # Define our input features (X) and target variable (y)
        feature_cols = [
            'Formation Energy (eV)', 
            'E Above Hull (eV)', 
            'Band Gap (eV)', 
            'Nsites', 
            'Density (gm/cc)', 
            'Volume'
        ]
        
        X = df[feature_cols]
        y = df['Crystal System']
        
        # Train a Random Forest Classifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        return model, feature_cols, df
        
    except FileNotFoundError:
        return None, None, None

# Load the trained model and data
rf_model, features, raw_data = train_model()

# ─── Dashboard UI ─────────────────────────────────────────────────────────────
st.title("🔋 Battery Material Crystal Predictor")
st.markdown("""
Enter the thermodynamic and physical properties of your theoretical battery material below. 
Our Machine Learning model will analyze these metrics to predict the material's underlying **Crystal System**.
""")

st.divider()

if rf_model is None:
    st.error("⚠️ **Dataset not found!** Please ensure `dataset0.csv` is uploaded to your GitHub repository in the same folder as this app.")
else:
    # Organize inputs into columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        form_energy = st.number_input("Formation Energy (eV)", value=-2.700, format="%.3f")
        band_gap = st.number_input("Band Gap (eV)", value=3.000, format="%.3f")
        density = st.number_input("Density (gm/cc)", value=3.000, format="%.3f")
    
    with col2:
        e_above_hull = st.number_input("E Above Hull (eV)", value=0.015, format="%.3f")
        nsites = st.number_input("Number of Sites (Nsites)", value=28, step=1)
        volume = st.number_input("Volume (Å³)", value=300.000, format="%.3f")
    
    st.divider()
    
    # ─── Prediction Logic ─────────────────────────────────────────────────────────
    if st.button("Predict Crystal Structure", type="primary", use_container_width=True):
        # Assemble the user inputs into a DataFrame matching the training features
        user_input_data = pd.DataFrame([[
            form_energy, 
            e_above_hull, 
            band_gap, 
            nsites, 
            density, 
            volume
        ]], columns=features)
        
        # Run the prediction
        prediction = rf_model.predict(user_input_data)[0]
        
        # Display the result
        st.success("Analysis Complete!")
        st.metric(label="Predicted Crystal System", value=prediction.capitalize())
        
        st.info("Note: The current prototype is trained on a limited dataset consisting solely of monoclinic structures. To predict other systems like cubic or triclinic, a larger training dataset is required.")
    
    # ─── Data Preview (Optional) ──────────────────────────────────────────────────
    with st.expander("View Training Data (dataset0.csv)"):
        st.dataframe(raw_data)