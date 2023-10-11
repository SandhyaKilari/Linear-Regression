from multiprocessing import Value
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rbf_nn(x, center1, center2, L, weight):
    phi1 = np.exp(-((x - center1) ** 2) / (L ** 2))
    phi2 = np.exp(-((x - center2) ** 2) / (L ** 2))
    prediction = phi1 * weight[0] + phi2 * weight[1]
    return prediction

# Select the dataset of your choice
mpg_dataset = sns.load_dataset("mpg").dropna()

# Excluded categorical features
mpg_dataset = mpg_dataset.select_dtypes(exclude='category')

data = [
    {'x':0.5, 'y':2},
    {'x': 1, 'y':1}
]
dataset_2 = pd.DataFrame(data)

data_button = st.selectbox('Please select one dataset from the following:', ['mpg_dataset', 'dataset_2'])

if data_button == 'mpg_dataset':
    # Choose the variable from the dataset
    st.title("Select X and Y Variables for the 'mpg' Dataset")
    x_variable = st.selectbox("X Variable", mpg_dataset.columns)
    y_variable = st.selectbox("Y Variable", mpg_dataset.columns)
    st.subheader("Scatter Plot of Selected Variables:")
    st.write(f"X Variable: {x_variable}")
    st.write(f"Y Variable: {y_variable}")
    # Create a scatter plot
    scatter_plot = sns.scatterplot(data=mpg_dataset, x=x_variable, y=y_variable)
    st.pyplot(scatter_plot.figure)

elif data_button == 'dataset_2':
    st.title("Select X and Y Variables for 'dataset_2'")
    x_variable = st.selectbox("X Variable", dataset_2.columns)
    y_variable = st.selectbox("Y Variable", dataset_2.columns)
    
    fig, ax = plt.subplots()
    ax.scatter(dataset_2[x_variable], dataset_2[y_variable])
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    st.pyplot(fig)

# Add a choice for RBF-NN model
model_choice = st.selectbox('Select a model:', ['Line', 'RBF-NN'])

if model_choice == 'RBF-NN':
    if data_button == 'mpg_dataset':
        df = mpg_dataset
    elif data_button == 'dataset_2':
        df = dataset_2
    st.subheader("RBF-NN Model Configuration")
    center1 = st.slider("Center 1", min_value=df[x_variable].min(), max_value=df[x_variable].max(), value=df[x_variable].min())
    center2 = st.slider("Center 2", min_value=df[x_variable].min(), max_value=df[x_variable].max(), value=df[x_variable].max())
    L = st.slider("Bandwidth (L)", min_value=0.1, max_value=5.0, value=1.0)
    weights = st.slider("Weights", min_value=-5.0, max_value=5.0, value=1.0)

    # Calculate RBF-NN predictions
    rbf_predictions = [rbf_nn(value, center1, center2, L, [weights, weights]) for value in df[x_variable]]

    st.subheader("RBF-NN Model")
    st.line_chart(pd.DataFrame({'x': df[x_variable], 'y': rbf_predictions}))

    # Calculate MAE and MSE for the RBF-NN model
    rbf_mae = mean_absolute_error(df[y_variable], rbf_predictions)
    rbf_mse = mean_squared_error(df[y_variable], rbf_predictions)

    st.subheader("Metrics for RBF-NN Model:")
    st.write(f"Mean Absolute Error (MAE): {rbf_mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {rbf_mse:.2f}")

elif model_choice == 'Line':
    if data_button == 'mpg_dataset':
        df = mpg_dataset
    elif data_button == 'dataset_2':
        df = dataset_2

    # Slider widgets for selecting slope and intercept
    slope = st.slider("Slope", -10.0, 10.0, 1.0)
    intercept = st.slider("Intercept", -10.0, 10.0, 0.0)

    # Generate some sample data points for visualization
    x = df[x_variable]
    y = slope * x + intercept

    st.title("Linear Regression Model:")
    st.line_chart(pd.DataFrame({'x': x, 'y': y}))

    mae = mean_absolute_error(df[y_variable], y)
    mse = mean_squared_error(df[y_variable], y)
    st.title("Metrics for Linear Model:")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    
