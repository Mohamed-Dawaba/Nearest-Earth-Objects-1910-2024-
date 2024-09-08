# Nearest Earth Objects (NEOs) Hazard Prediction

This repository contains a project aimed at predicting whether a Near-Earth Object (NEO) is hazardous based on various features such as absolute magnitude, estimated diameter, relative velocity, and miss distance. The project includes data preprocessing, model training, and evaluation using multiple machine learning algorithms.

## Project Overview

This project uses a dataset of Near-Earth Objects (NEOs) to classify them into hazardous and non-hazardous categories. The primary goal is to create an accurate model to predict whether an NEO poses a threat to Earth. The following steps were taken:

1. **Data Preprocessing**: Cleaning the data, handling missing values, scaling features, and dealing with class imbalance.
2. **Model Training and Evaluation**: Training multiple machine learning models, comparing their performance, and selecting the best-performing model.
3. **Insights and Key Findings**: Drawing insights from data exploration and model evaluation.

## Dataset

The dataset contains information on NEOs between 1910 and 2024. The key features include:
- **absolute_magnitude**: The brightness of the NEO.
- **estimated_diameter_min**: The minimum estimated diameter of the NEO (in km).
- **estimated_diameter_max**: The maximum estimated diameter of the NEO (in km).
- **relative_velocity**: The velocity of the NEO relative to Earth (in km/h).
- **miss_distance**: The distance between Earth and the NEO at its closest approach (in km).
- **is_hazardous**: The target variable indicating if the NEO is hazardous (1 for hazardous, 0 for non-hazardous).

## Approach

### 1. Exploratory Data Analysis (EDA)

- **Visualization**: We used histograms, scatter plots, and correlation matrices to identify trends and relationships in the data. 
- **Outlier Detection**: Boxplots were used to identify and handle outliers in the continuous features.
- **Feature Engineering**: Categorical and continuous features were identified, but categorical features were not present in this dataset after initial cleaning.

### 2. Data Preprocessing

- **Handling Missing Values**: Missing values in continuous features were filled using the mean.
- **Outlier Handling**: We used the Interquartile Range (IQR) method to replace outliers with boundary values.
- **Scaling**: Continuous features were standardized using `StandardScaler` to ensure all features are on a similar scale.
- **Class Imbalance**: The target variable `is_hazardous` was highly imbalanced (fewer hazardous NEOs). We addressed this using:
  - **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class.
  - **Class Weight Adjustment** to penalize misclassification of hazardous objects.

### 3. Model Training and Evaluation

The following models were trained on the preprocessed dataset:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

#### Evaluation Metrics:
To evaluate model performance, the following metrics were used:
- **Accuracy**: Overall accuracy of the predictions.
- **Precision**: Percentage of correctly predicted hazardous NEOs out of all predicted hazardous NEOs.
- **Recall**: Percentage of actual hazardous NEOs that were correctly predicted.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC**: Measures the performance of the model by plotting the True Positive Rate against the False Positive Rate.

#### Key Findings:
- **Random Forest** emerged as the best-performing model, with high **F1-Score** and **AUC-ROC**.
- **Precision** and **Recall** were balanced in the Random Forest model, making it a reliable choice for identifying hazardous NEOs.

### 4. Results

| Model                 | Accuracy | Precision | Recall  | F1-Score | AUC-ROC |
|-----------------------|----------|-----------|---------|----------|---------|
| Logistic Regression    | 0.92     | 0.85      | 0.80    | 0.82     | 0.89    |
| Random Forest          | 0.94     | 0.88      | 0.86    | 0.87     | 0.93    |
| Support Vector Machine | 0.91     | 0.83      | 0.78    | 0.80     | 0.87    |

- The **Random Forest** classifier had the highest **AUC-ROC** of 0.93, indicating excellent performance in distinguishing between hazardous and non-hazardous NEOs.

### Conclusion

- The project successfully predicted whether an NEO is hazardous or not based on various features.
- The **Random Forest** model was the most effective for this classification task.
- Class imbalance was handled using SMOTE, improving the model's ability to identify hazardous NEOs without bias toward the majority class.

## How to Run This Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neo-hazard-prediction.git


## Required Dependencies

To run this project, you'll need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`

You can install all required dependencies by running:

```bash
pip install -r requirements.txt


## Future Work

There are several ways to improve this project in the future:

1. **Model Optimization**: 
   - Perform hyperparameter tuning using techniques like Grid Search or Random Search to optimize model performance.
   - Experiment with more advanced algorithms like XGBoost, LightGBM, or Neural Networks for potentially better results.
   
2. **Time-Based Analysis**:
   - Incorporate temporal features to predict hazardous NEOs over time, which could be useful for forecasting future threats.
   
3. **Feature Engineering**:
   - Explore new features such as object composition or orbital characteristics to improve classification performance.
   - Investigate feature interactions that might provide better predictive power.

4. **Handling Imbalanced Data with More Techniques**:
   - Test alternative strategies like undersampling, class-weight adjustments, or ensemble methods that are robust to class imbalance.


## Contact

For any questions or feedback, please reach out to [mohameddawaba3@gmail.com].
