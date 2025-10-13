"""
Decision Tree Simulator Page
Author: Akshit
Date: October 13, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
from models.decision_tree import DecisionTreeModel
from utils.plot_helpers import (plot_confusion_matrix, plot_feature_importance,
                                plot_prediction_distribution, plot_actual_vs_predicted,
                                plot_residuals)
from sklearn.datasets import load_iris, load_diabetes

# Page configuration
st.set_page_config(page_title="Decision Tree Simulator", layout="wide", page_icon="🌳")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #d5f4e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌳 Decision Tree Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>🎯 About Decision Trees</h3>
    <p>Decision Trees are intuitive models that make predictions by learning simple decision rules 
    from data features. They're easy to interpret and visualize, making them perfect for understanding 
    how predictions are made.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Model Configuration")

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
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.sidebar.info("📊 Using Iris Dataset (Classification)")
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
    
    # Model Parameters
    st.markdown("### 🔧 Model Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_depth = st.slider("Max Depth", 1, 20, 5, 
                             help="Maximum depth of the tree")
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2,
                                     help="Minimum samples to split a node")
    with col3:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1,
                                    help="Minimum samples in leaf node")
    
    if task == "Classification":
        criterion = st.selectbox("Criterion:", ["gini", "entropy"],
                                help="Function to measure split quality")
    else:
        criterion = st.selectbox("Criterion:", ["squared_error", "absolute_error"],
                                help="Function to measure split quality")
    
    # Train Model
    if len(selected_features) > 0 and st.button("🚀 Train Decision Tree"):
        with st.spinner('🌳 Growing the tree...'):
            # Prepare data
            X = df[selected_features].fillna(df[selected_features].mean())
            y = df[target_column]
            
            # Initialize model
            dt_model = DecisionTreeModel(
                task=task.lower(),
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42
            )
            
            # Train
            results = dt_model.train(X, y, test_size=test_size)
            
            st.success("✅ Decision Tree trained successfully!")
            
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
                
                # Confusion Matrix
                st.markdown("### 🎯 Confusion Matrix")
                cm_fig = plot_confusion_matrix(results['y_test'], results['y_pred'])
                st.pyplot(cm_fig)
                
                # Classification Report
                st.markdown("### 📋 Classification Report")
                report_df = pd.DataFrame(results['classification_report']).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='Greens'))
                
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
                
                # Actual vs Predicted
                st.markdown("### 📈 Actual vs Predicted")
                avp_fig = plot_actual_vs_predicted(results['y_test'], results['y_pred'])
                st.pyplot(avp_fig)
                
                # Residuals
                st.markdown("### 📉 Residual Analysis")
                res_fig = plot_residuals(results['y_test'], results['y_pred'])
                st.pyplot(res_fig)
            
            # Tree Structure
            st.markdown("### 🌳 Decision Tree Structure")
            st.info("Displaying top 3 levels of the tree for readability")
            tree_fig = dt_model.plot_tree_structure(
                feature_names=selected_features,
                max_depth_display=3
            )
            st.pyplot(tree_fig)
            
            # Tree Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", results['n_nodes'])
            with col2:
                st.metric("Leaf Nodes", results['n_leaves'])
            with col3:
                st.metric("Tree Depth", results['max_depth_achieved'])
            
            # Feature Importance
            st.markdown("### ⭐ Feature Importance")
            fi_fig = plot_feature_importance(selected_features, results['feature_importance'])
            st.pyplot(fi_fig)
            
            # Feature importance table
            importance_df = dt_model.get_feature_importance(selected_features)
            st.dataframe(importance_df, use_container_width=True)

else:
    st.info("👆 Please upload a dataset or select sample dataset to get started!")
    
    st.markdown("""
    ### 📋 How to Use:
    1. Choose task type (Classification or Regression)
    2. Upload CSV or use sample dataset
    3. Select target column and features
    4. Adjust tree parameters (max_depth, min_samples, etc.)
    5. Click **Train Decision Tree**
    6. View results, tree structure, and feature importance
    
    ### 🌟 Key Parameters:
    - **Max Depth**: Limits tree depth to prevent overfitting
    - **Min Samples Split**: Minimum samples needed to split a node
    - **Min Samples Leaf**: Minimum samples in each leaf node
    - **Criterion**: Gini/Entropy (classification) or MSE/MAE (regression)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🎃 Hacktoberfest 2025 | Built by Akshit | Decision Tree Simulator</p>
</div>
""", unsafe_allow_html=True)
