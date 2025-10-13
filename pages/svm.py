"""
Support Vector Machine (SVM) Simulator Page
Author: Akshit
Date: October 13, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from models.svm import SVMModel
from utils.plot_helpers import (plot_confusion_matrix, plot_roc_curve,
                                plot_actual_vs_predicted, plot_residuals)
from utils.data_helpers import get_dataset_by_name
from sklearn.datasets import load_breast_cancer, load_diabetes

# Page configuration
st.set_page_config(page_title="SVM Simulator", layout="wide", page_icon="🎯")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 Support Vector Machine Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>🚀 About Support Vector Machines</h3>
    <p>SVMs are powerful supervised learning models that find the optimal hyperplane to separate classes
    or fit data. They work exceptionally well with high-dimensional data and are effective in cases where
    the number of dimensions exceeds the number of samples.</p>
    <p><b>Key Feature:</b> SVMs use support vectors (data points closest to the decision boundary) to define the margin.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ SVM Configuration")

# Task selection
task = st.sidebar.radio("Select Task:", ["Classification", "Regression"])

# Data source
data_source = st.sidebar.radio("Data Source:", ["Upload CSV", "Use Sample Dataset"])

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded!")
else:
    if task == "Classification":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.sidebar.info("📊 Using Breast Cancer Dataset (Classification)")
    else:
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.sidebar.info("📊 Using Diabetes Dataset (Regression)")

if df is not None:
    # Dataset Overview
    st.markdown("### 📁 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("👀 View Dataset"):
        st.dataframe(df.head(10))
    
    # Feature and Target Selection
    st.markdown("### 🎯 Feature & Target Selection")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        target_column = st.selectbox("Select Target Column:", df.columns)
    with col2:
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    
    available_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Select Features:",
        available_features,
        default=available_features[:min(5, len(available_features))]
    )
    
    # SVM Parameters
    st.markdown("### 🔧 SVM Parameters")
    
    st.markdown("""
    <div class="warning-box">
        <b>⚠️ Important:</b> Feature scaling is <b>highly recommended</b> for SVM. 
        The model will automatically scale features using StandardScaler.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"],
                             help="Kernel function for the algorithm")
    with col2:
        C = st.slider("C (Regularization)", 0.01, 100.0, 1.0, step=0.01,
                     help="Smaller C = more regularization")
    with col3:
        gamma = st.selectbox("Gamma:", ["scale", "auto"],
                            help="Kernel coefficient")
    
    # Additional parameters based on kernel
    if kernel == "poly":
        degree = st.slider("Polynomial Degree:", 2, 5, 3)
    else:
        degree = 3
    
    if task == "Regression":
        epsilon = st.slider("Epsilon:", 0.01, 1.0, 0.1,
                           help="Epsilon in epsilon-SVR")
    else:
        epsilon = 0.1
    
    scale_features = st.checkbox("Scale Features (Recommended)", value=True,
                                 help="Apply StandardScaler to features")
    
    # Train Model
    if len(selected_features) > 0 and st.button("🚀 Train SVM Model"):
        with st.spinner('🎯 Training SVM...'):
            # Prepare data
            X = df[selected_features].fillna(df[selected_features].mean())
            y = df[target_column]
            
            # Initialize model
            svm_model = SVMModel(
                task=task.lower(),
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                epsilon=epsilon,
                random_state=42
            )
            
            # Train
            results = svm_model.train(X, y, test_size=test_size, scale_features=scale_features)
            
            st.success("✅ SVM trained successfully!")
            
            # ==================== RESULTS ====================
            st.markdown("### 📊 Training Results")
            
            if task == "Classification":
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Accuracy</h3>
                        <h2>{results['test_accuracy']:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <h2>{results['precision']:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <h2>{results['recall']:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <h2>{results['f1_score']:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Support Vectors Info
                st.markdown("### 🎯 Support Vectors Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Support Vectors", int(np.sum(results['n_support_vectors'])))
                with col2:
                    support_per_class = results['n_support_per_class']
                    st.write("**Support Vectors per Class:**")
                    for class_label, count in support_per_class.items():
                        st.write(f"- Class {class_label}: {count} vectors")
                
                # Confusion Matrix
                st.markdown("### 🎯 Confusion Matrix")
                cm_fig = plot_confusion_matrix(results['y_test'], results['y_pred'])
                st.pyplot(cm_fig)
                
                # ROC Curve (for binary classification)
                if len(np.unique(results['y_test'])) == 2 and results['y_pred_proba'] is not None:
                    st.markdown("### 📈 ROC Curve")
                    roc_fig = plot_roc_curve(results['y_test'], results['y_pred_proba'])
                    st.pyplot(roc_fig)
                
                # Classification Report
                st.markdown("### 📋 Classification Report")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Blues'))
                
                # Decision Boundary (for 2D data)
                if len(selected_features) >= 2:
                    st.markdown("### 🎨 Decision Boundary Visualization")
                    st.info("Showing decision boundary for first 2 features")
                    try:
                        db_fig = svm_model.plot_decision_boundary(feature_idx=[0, 1])
                        st.pyplot(db_fig)
                    except Exception as e:
                        st.warning(f"Could not plot decision boundary: {str(e)}")
                    
                    # Margin plot for linear kernel
                    if kernel == "linear":
                        st.markdown("### 📏 Hyperplane and Margins")
                        try:
                            margin_fig = svm_model.plot_support_vectors_detail(feature_idx=[0, 1])
                            st.pyplot(margin_fig)
                        except Exception as e:
                            st.warning(f"Could not plot margins: {str(e)}")
            
            else:
                # Regression metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>R² Score</h3>
                        <h2>{results['test_r2']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MSE</h3>
                        <h2>{results['mse']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>RMSE</h3>
                        <h2>{results['rmse']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>MAE</h3>
                        <h2>{results['mae']:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Support Vectors
                st.markdown("### 🎯 Support Vectors")
                st.metric("Number of Support Vectors", int(np.sum(results['n_support_vectors'])))
                
                # Actual vs Predicted
                st.markdown("### 📈 Actual vs Predicted")
                avp_fig = plot_actual_vs_predicted(results['y_test'], results['y_pred'])
                st.pyplot(avp_fig)
                
                # Residuals
                st.markdown("### 📉 Residual Analysis")
                res_fig = plot_residuals(results['y_test'], results['y_pred'])
                st.pyplot(res_fig)
            
            # Model Information
            st.markdown("### ℹ️ Model Information")
            model_info = svm_model.get_model_info()
            info_df = pd.DataFrame([model_info]).T
            info_df.columns = ['Value']
            st.dataframe(info_df)

else:
    st.info("👆 Please upload a dataset or select sample dataset to get started!")
    
    st.markdown("""
    ### 📋 How to Use:
    1. Choose task type (Classification or Regression)
    2. Upload CSV or use sample dataset
    3. Select target column and features
    4. Adjust SVM parameters (kernel, C, gamma)
    5. Click **Train SVM Model**
    6. View results, decision boundaries, and support vectors
    
    ### 🌟 Key SVM Parameters:
    - **Kernel**: Function to transform data ('rbf' is most versatile)
    - **C**: Regularization (lower = more regularization, smoother boundary)
    - **Gamma**: Influence of single training examples ('scale' is usually best)
    - **Degree**: For polynomial kernel only
    
    ### 💡 Tips:
    - Always scale features for SVM (enabled by default)
    - Start with RBF kernel and C=1.0
    - Lower C if overfitting, higher C if underfitting
    - Linear kernel is fastest for large datasets
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🎃 Hacktoberfest 2025 | Built by Akshit | SVM Simulator</p>
</div>
""", unsafe_allow_html=True)
