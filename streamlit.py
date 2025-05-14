import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras # For loading Keras models
import joblib # For loading scikit-learn scalers
import altair as alt # For plotting
from sklearn.manifold import TSNE # For t-SNE
from sklearn.decomposition import PCA # For PCA


# Classifier model and its scaler
CLASSIFIER_MODEL_PATH = "feed_forward_nn_original_data.keras"
CLASSIFIER_SCALER_PATH = "scaler_for_feed_forward_nn_original_data.joblib"
CATEGORIES_PATH = "categories.npy"
FEATURE_NAMES_PATH = "training_feature_names.npy" # Genes used by the classifier

# Autoencoder's encoder model and its specific scaler
AUTOENCODER_ENCODER_PATH = "AE_Latent64_NoL1_L2_Dropout.keras" 
AUTOENCODER_SCALER_PATH = "AE_Latent64_NoL1_L2_Dropout.joblib" 

#Caching for Model and Scaler Loading
@st.cache_resource
def load_all_artifacts():
    """Loads all models, scalers, categories, and feature names."""
    artifacts = {"classifier_model": None, "classifier_scaler": None, "categories": None, 
                 "training_feature_names": None, "autoencoder_encoder": None, "autoencoder_scaler": None}
    try:
        artifacts["classifier_model"] = keras.models.load_model(CLASSIFIER_MODEL_PATH)
        artifacts["classifier_scaler"] = joblib.load(CLASSIFIER_SCALER_PATH)
        artifacts["categories"] = np.load(CATEGORIES_PATH, allow_pickle=True)
        artifacts["training_feature_names"] = np.load(FEATURE_NAMES_PATH, allow_pickle=True)
    except Exception as e:
        st.error(f"Error loading classifier model or its artifacts: {e}")
    
    try:
        artifacts["autoencoder_encoder"] = keras.models.load_model(AUTOENCODER_ENCODER_PATH)
        artifacts["autoencoder_scaler"] = joblib.load(AUTOENCODER_SCALER_PATH)
    except Exception as e:
        st.warning(f"Could not load autoencoder artifacts (for AE+PCA visualization): {e}")
    return artifacts

# Helper Function for Data Processing
def preprocess_and_align_data(df_input_original, training_features_list):
    df_processed = df_input_original.copy()
    possible_id_colnames = ['cell', 'barcode', 'id', 'unnamed: 0', 'index']
    cell_ids = None
    df_features = None

    if df_processed.index.name is not None and df_processed.index.name.lower() not in ['gene', 'genes'] and df_processed.index.nunique() == len(df_processed):
        cell_ids = df_processed.index.astype(str)
        df_features = df_processed.copy()
    else:
        found_id_col = None
        for col_name in df_processed.columns:
            if col_name.lower() in possible_id_colnames:
                if df_processed[col_name].nunique() == len(df_processed[col_name]) or pd.api.types.is_string_dtype(df_processed[col_name]):
                    found_id_col = col_name
                    break
        if found_id_col:
            cell_ids = df_processed[found_id_col].astype(str)
            df_features = df_processed.drop(columns=[found_id_col]).copy()
        else:
            cell_ids = pd.Series([f"Cell_{i+1}" for i in range(len(df_processed))])
            df_features = df_processed.copy()

    uploaded_features_set = set(df_features.columns)
    training_features_set = set(training_features_list)
    missing_in_upload = list(training_features_set - uploaded_features_set)
    if missing_in_upload:
        st.warning(f"The following {len(missing_in_upload)} gene(s) required by the model were missing and imputed with zeros: "
                   f"{', '.join(missing_in_upload[:5])}{'...' if len(missing_in_upload) > 5 else ''}")
        for gene in missing_in_upload:
            df_features[gene] = 0
    extra_in_upload = list(uploaded_features_set - training_features_set)
    if extra_in_upload:
        st.info(f"The following {len(extra_in_upload)} gene(s) were present but not used by the classifier and were ignored: "
                f"{', '.join(extra_in_upload[:5])}{'...' if len(extra_in_upload) > 5 else ''}")
    try:
        df_features_aligned = df_features[training_features_list].copy()
        for col in df_features_aligned.columns:
            df_features_aligned[col] = pd.to_numeric(df_features_aligned[col], errors='coerce')
        if df_features_aligned.isnull().values.any():
            st.warning("Non-numeric values found and imputed with 0 before scaling.")
            df_features_aligned = df_features_aligned.fillna(0)
        return cell_ids, df_features_aligned
    except KeyError as e:
        st.error(f"Error aligning features: {e}.")
        return None, None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Initialize Session State
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'dim_reduction_df' not in st.session_state:
    st.session_state.dim_reduction_df = None
if 'df_features_aligned_for_dim_reduction' not in st.session_state:
    st.session_state.df_features_aligned_for_dim_reduction = None
if 'current_dim_reduction_method' not in st.session_state:
    st.session_state.current_dim_reduction_method = None
if 'current_dim_reduction_plot_coords' not in st.session_state:
    st.session_state.current_dim_reduction_plot_coords = ['Dim_1', 'Dim_2'] 

# Main Application UI
st.set_page_config(layout="wide", page_title="scRNA-seq Cell Type Classifier")
st.title("🔬 scRNA-seq Cell Type Classifier & Visualizer")

st.markdown("""
**Welcome! This tool allows you to classify cell types from scRNA-seq count data and visualize the results.**

**How to Use:**
1.  **Upload your CSV file:**
    * The file should contain cells as rows and genes as columns.
    * The application will try to identify cell identifiers from the first column or the row index. If it can't, default IDs will be assigned.
    * Gene names in your CSV column headers should generally match those used for training the model.
        * *Missing required genes?* They'll be automatically filled with zeros (you'll be notified).
        * *Extra genes not in the training set?* They'll be ignored (you'll be notified).
2.  **Classify Cell Types:** Click the "Classify Cell Types" button. Results and a summary bar chart will appear.
3.  **Visualize Data:**
    * Choose a dimensionality reduction method (t-SNE or Autoencoder+PCA).
    * Click "Generate Dimensionality Reduction Plot".
    * Use the slider to filter points by prediction confidence.
    * Check the box to color the plot by predicted cell type.
4.  **Download:** Get your prediction table and a summary report.
---
""")

artifacts = load_all_artifacts()
classifier_model = artifacts["classifier_model"]
classifier_scaler = artifacts["classifier_scaler"]
categories = artifacts["categories"]
training_feature_names = artifacts["training_feature_names"]
autoencoder_encoder = artifacts["autoencoder_encoder"]
autoencoder_scaler = artifacts["autoencoder_scaler"]

if not (classifier_model and classifier_scaler and categories is not None and training_feature_names is not None):
    st.error("Crucial classifier model or its artifacts could not be loaded. The application cannot proceed with classification.")
    st.stop() 

st.success(f"Classifier model loaded. Ready to classify based on {len(training_feature_names)} gene features for {len(categories)} cell types.")
if autoencoder_encoder and autoencoder_scaler:
    st.info("Autoencoder artifacts for 'PCA on Autoencoder Latent Space' visualization option are also available.")
else:
    st.warning("Autoencoder artifacts not loaded. The 'PCA on Autoencoder Latent Space' visualization option will not be available.")

uploaded_file = st.file_uploader("📤 Upload your scRNA-seq CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df_original = pd.read_csv(uploaded_file)
        st.info("Successfully read CSV.")
        with st.expander("View Uploaded Data Preview (First 5 Rows)"):
            st.dataframe(input_df_original.head())

        if st.button("🚀 Classify Cell Types", type="primary", key="classify_button"):
            st.write("Debug: Classify button pressed.") # Debug
            with st.spinner("Preprocessing data..."):
                cell_ids, df_features_aligned = preprocess_and_align_data(input_df_original, training_feature_names)
            
            if cell_ids is not None and df_features_aligned is not None:
                st.session_state.df_features_aligned_for_dim_reduction = df_features_aligned
                with st.spinner("Scaling data and making predictions..."):
                    try:
                        scaled_data_for_classifier = classifier_scaler.transform(df_features_aligned)
                        predictions_proba = classifier_model.predict(scaled_data_for_classifier)
                        predicted_indices = np.argmax(predictions_proba, axis=1)
                        predicted_labels = categories[predicted_indices]
                        predicted_confidences = np.max(predictions_proba, axis=1)
                        st.session_state.predictions_df = pd.DataFrame({
                            'Cell_ID': cell_ids.values,
                            'Predicted_Cell_Type': predicted_labels,
                            'Confidence': predicted_confidences
                        })
                        st.session_state.dim_reduction_df = None 
                        st.session_state.current_dim_reduction_method = None
                        st.write("Debug: Classification complete. predictions_df set.") # Debug
                    except Exception as e:
                        st.error(f"Error during prediction or scaling: {e}")
                        st.session_state.predictions_df = None
                        st.session_state.df_features_aligned_for_dim_reduction = None
            else:
                st.error("Data preprocessing failed. Cannot proceed with classification.")
        
        if st.session_state.get('predictions_df') is not None:
            st.write("Debug: predictions_df exists. Displaying results and viz UI.") # Debug
            st.subheader("📊 Prediction Results")
            st.dataframe(st.session_state.predictions_df)

            st.subheader("📈 Dimensionality Reduction & Visualizations")
            
            dim_reduction_method_options = ["t-SNE on Original Scaled Data"]
            if autoencoder_encoder and autoencoder_scaler: 
                dim_reduction_method_options.append("PCA on Autoencoder Latent Space")

            dim_reduction_method = st.radio(
                "Choose dimensionality reduction method for visualization:",
                options=dim_reduction_method_options,
                key="dim_reduction_choice"
            )

            if st.button("🧬 Generate Dimensionality Reduction Plot", key="generate_dim_red_plot"):
                st.write(f"Debug: Generate plot button pressed. Method: {dim_reduction_method}") # Debug
                if st.session_state.current_dim_reduction_method != dim_reduction_method or st.session_state.dim_reduction_df is None:
                    st.session_state.dim_reduction_df = None # Clear previous plot data if method changed
                st.session_state.current_dim_reduction_method = dim_reduction_method 
                
                if st.session_state.get('df_features_aligned_for_dim_reduction') is not None:
                    base_data_for_dim_reduction = st.session_state.df_features_aligned_for_dim_reduction
                    plot_coord_names = ['Dim_1', 'Dim_2'] 
                    dim_red_results = np.zeros((len(base_data_for_dim_reduction), 2)) 
                    
                    if dim_reduction_method == "t-SNE on Original Scaled Data":
                        with st.spinner("Scaling for t-SNE & generating t-SNE embedding... This may take a moment."):
                            data_scaled_for_tsne = classifier_scaler.transform(base_data_for_dim_reduction)
                            n_samples = len(data_scaled_for_tsne)
                            if n_samples > 1:
                                perplexity_value = min(30.0, float(n_samples - 1))
                                if perplexity_value < 1.0: perplexity_value = 1.0
                                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, 
                                            n_iter=300, init='pca', learning_rate='auto')
                                dim_red_results = tsne.fit_transform(data_scaled_for_tsne)
                            else:
                                st.warning("Not enough data points (>1) for t-SNE. Displaying placeholder.")
                            plot_coord_names = ['tSNE_1', 'tSNE_2']
                    
                    elif dim_reduction_method == "PCA on Autoencoder Latent Space":
                        if autoencoder_encoder and autoencoder_scaler:
                            with st.spinner("Scaling for AE, generating latent space & PCA embedding..."):
                                data_scaled_for_ae = autoencoder_scaler.transform(base_data_for_dim_reduction)
                                latent_space = autoencoder_encoder.predict(data_scaled_for_ae)
                                pca = PCA(n_components=2, random_state=42)
                                dim_red_results = pca.fit_transform(latent_space)
                                plot_coord_names = ['AE_PCA_1', 'AE_PCA_2']
                        else:
                            st.error("Autoencoder artifacts not loaded. Cannot perform AE+PCA.")
                    
                    # Ensure predictions_df is still valid before copying
                    if st.session_state.get('predictions_df') is not None:
                        temp_df = st.session_state.predictions_df.copy()
                        temp_df[plot_coord_names[0]] = dim_red_results[:, 0]
                        temp_df[plot_coord_names[1]] = dim_red_results[:, 1]
                        st.session_state.dim_reduction_df = temp_df
                        st.session_state.current_dim_reduction_plot_coords = plot_coord_names
                        st.write("Debug: dim_reduction_df generated.") # Debug
                    else:
                        st.warning("Prediction data is missing. Cannot generate full dimensionality reduction plot.")
                else:
                    st.warning("No preprocessed data available (df_features_aligned_for_dim_reduction is missing). Please classify cells first.")


            col1_viz, col2_viz = st.columns([1, 2]) 

            with col1_viz:
                st.markdown("#### Distribution of Predicted Cell Types")
                if st.session_state.get('predictions_df') is not None:
                    cell_type_counts = st.session_state.predictions_df['Predicted_Cell_Type'].value_counts().reset_index()
                    cell_type_counts.columns = ['Predicted_Cell_Type', 'Count']
                    bar_chart = alt.Chart(cell_type_counts).mark_bar().encode(
                        x=alt.X('Predicted_Cell_Type:N', sort='-y', title="Predicted Cell Type"),
                        y=alt.Y('Count:Q', title="Number of Cells"),
                        tooltip=['Predicted_Cell_Type', 'Count']
                    ).properties(height=350) 
                    st.altair_chart(bar_chart, use_container_width=True)
                else:
                    st.info("Classify data to see distribution.")
            
            with col2_viz:
                plot_title_display = st.session_state.get('current_dim_reduction_method', "Not Generated")
                st.markdown(f"#### Scatter Plot: {plot_title_display}")
                if st.session_state.get('dim_reduction_df') is not None:
                    min_confidence_scatter = st.slider("Minimum Prediction Confidence for Scatter Plot", 0.0, 1.0, 0.5, 0.05, key="min_conf_scatter")
                    color_by_cell_type_checkbox = st.checkbox("Color by Predicted Cell Type", False, key="color_scatter_by_type")

                    filtered_scatter_df = st.session_state.dim_reduction_df[st.session_state.dim_reduction_df['Confidence'] >= min_confidence_scatter]
                    
                    coord_x, coord_y = st.session_state.get('current_dim_reduction_plot_coords', ['Dim_1', 'Dim_2'])

                    if not filtered_scatter_df.empty:
                        color_encoding = alt.value('steelblue') 
                        if color_by_cell_type_checkbox:
                            color_encoding = alt.Color('Predicted_Cell_Type:N', legend=alt.Legend(title="Predicted Cell Type"))

                        scatter_chart_final = alt.Chart(filtered_scatter_df).mark_circle(size=60).encode(
                            x=alt.X(coord_x, title=f"{coord_x.replace('_', ' ')}"),
                            y=alt.Y(coord_y, title=f"{coord_y.replace('_', ' ')}"),
                            color=color_encoding,
                            tooltip=['Cell_ID', 'Predicted_Cell_Type', 'Confidence', coord_x, coord_y]
                        ).interactive().properties(height=350) 
                        
                        st.altair_chart(scatter_chart_final, use_container_width=True)
                    else:
                        st.info("No data points meet the current confidence threshold for the scatter plot.")
                else:
                    st.info("Click 'Generate Dimensionality Reduction Plot' to visualize the data.")

            if st.session_state.get('predictions_df') is not None:
                st.subheader("📥 Download Results")
                csv_predictions = convert_df_to_csv(st.session_state.predictions_df)
                st.download_button(label="Download Predictions as CSV", data=csv_predictions, file_name="cell_type_predictions.csv", mime="text/csv")
                summary_text = "Cell Type Prediction Summary:\n\n"
                summary_text += f"Total cells processed: {len(st.session_state.predictions_df)}\n\nPredicted Cell Type Counts:\n"
                cell_type_counts_download = st.session_state.predictions_df['Predicted_Cell_Type'].value_counts()
                for cell_type, count_val in cell_type_counts_download.items(): 
                    summary_text += f"- {cell_type}: {count_val}\n"
                st.download_button(label="Download Summary Report (TXT)", data=summary_text.encode('utf-8'), file_name="prediction_summary.txt", mime="text/plain")
    
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
        st.exception(e) 
        st.error("Please ensure your CSV file is correctly formatted.")

st.markdown("---")
st.markdown("Developed with Streamlit and TensorFlow/Keras.")
