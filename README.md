# DSML
# Crop Yield Prediction Project

## Overview

This project focuses on building a machine learning model to predict crop yield based on various environmental factors and crop-specific data. By leveraging historical crop production, soil conditions, weather data, and location information, the project aims to provide insights and predictions that could potentially help farmers and agricultural stakeholders make informed decisions.

## Project Goal

The primary goal is to develop a robust regression model capable of accurately predicting the expected yield (tons per hectare) for different crops under varying conditions.

## Dataset

The dataset used in this project is the "Crop Production in India" dataset. It contains information about:
- State Name
- District Name
- Crop Year
- Season
- Crop
- Area (in hectares)
- Production (in tons)
- Weather parameters (N, P, K, pH, rainfall, temperature)

*(Note: You might want to specify the source of your weather parameters if they were merged from a separate dataset. If N, P, K, pH, rainfall, and temperature were already part of the "Crop Production in India" dataset you used, you can mention that.)*

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration:** Loading the dataset and understanding its structure, data types, and initial statistics.
2.  **Data Preprocessing:** Handling missing values (dropping rows), encoding categorical features (State Name, Crop Type, Crop), and preparing the data for modeling.
3.  **Exploratory Data Analysis (EDA):** Visualizing data distributions, relationships between features, and correlations to gain insights.
4.  **Feature Engineering/Selection:** Preparing features for the models, including dropping columns that are not needed for prediction ('Unnamed: 0', original categorical columns).
5.  **Model Selection:** Experimenting with different regression models suitable for predicting a continuous numerical value.
6.  **Model Training:** Splitting the data into training and testing sets and training the selected models on the training data.
7.  **Model Evaluation:** Evaluating the performance of the trained models using appropriate regression metrics (Mean Absolute Error, Root Mean Squared Error).
8.  **Prediction Interface (Optional but included):** Building a simple interactive interface using `ipywidgets` to allow users to input parameters and get a predicted yield.

## Libraries Used

-   `pandas` for data manipulation and analysis
-   `numpy` for numerical operations
-   `matplotlib` and `seaborn` for data visualization
-   `plotly` for interactive plots
-   `scikit-learn` for data preprocessing, model selection, training, and evaluation
    -   `StandardScaler`
    -   `OrdinalEncoder`
    -   `train_test_split`, `KFold`, `GridSearchCV`
    -   `LinearRegression`, `ElasticNet`, `Ridge`, `KNeighborsRegressor`, `RandomForestRegressor`
    -   `mean_absolute_error`, `mean_squared_error`, `mean_squared_log_error`
-   `xgboost` for the XGBoost Regressor model
-   `ipywidgets` for the interactive prediction interface

## Installation

1.  Clone the repository: git clone
2.  Navigate to the project directory: cd crop-yield-prediction
3.  Install the required libraries: pip install -r requirements.txt
4.  *(Note: You will need to create a `requirements.txt` file containing all the libraries listed above. You can generate this automatically using `pip freeze > requirements.txt` after installing all libraries in your environment.)*

## Usage

1.  Ensure you have the dataset (`Crop_production.csv`) in the correct path (`/content/Crop_production.csv` as per your code, or update the path in the notebook).
2.  Open and run the Jupyter Notebook (`your_notebook_name.ipynb`).
3.  Follow the cells sequentially to load data, preprocess, explore, train models, and evaluate.
4.  Use the interactive `ipywidgets` interface at the end of the notebook to input values and get a yield prediction.

## Code Explanation (Important Functions/Sections)

*(This section will be generated below)*

## Results

*(Describe the performance of your models based on the MAE and RMSE scores you printed. Mention which model performed best and why you think that might be the case.)*

## Future Work

*(Suggest potential improvements, such as trying more advanced models, incorporating more data sources (e.g., detailed soil data, satellite imagery), performing more in-depth feature engineering, hyperparameter tuning, etc.)*

## Contributing

Feel free to fork the repository and contribute.

## License

*(Choose an open-source license like MIT or Apache 2.0 and mention it here. You should also include a LICENSE file in your repository.)*

## Contact

*(Add your contact information, like your GitHub profile or email, if you'd like people to reach out.)*
