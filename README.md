# **Loan Approval Prediction Using Machine Learning**  

## **Overview**  
This project aims to predict the approval status of loan applications using several machine learning models. It involves data preprocessing, outlier detection, exploratory visualizations, and model evaluations to ensure high-quality predictions. The insights generated will aid financial institutions in making data-driven decisions efficiently.  

---

## **Key Highlights**  
- **Data Preprocessing**:  
   - Handling missing values using mean (for numerical) and mode (for categorical) imputation.  
   - Identifying outliers with **Z-Score analysis**.  

- **Machine Learning Models Implemented**:  
   - Logistic Regression  
   - Linear Discriminant Analysis (LDA)  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  

- **Evaluation Metrics**:  
   - Accuracy Score  
   - Confusion Matrices  
   - Classification Report (Precision, Recall, F1-Score)  
   - Visual comparison of model performance  

---

## **Project Workflow**  

1. **Loading the Data**  
   - Dataset: `Loan_data.csv`  
   - Missing value treatment for robust predictions.  

2. **Exploratory Data Analysis (EDA)**  
   - **Box Plot** to visualize outliers across numerical columns.  
   - **Histogram** for the distribution of loan amounts with thresholds highlighting outliers.  

3. **Model Building & Training**  
   - Data split into **80% training** and **20% testing** sets.  
   - Standardization of features using `StandardScaler`.  
   - Training and evaluation of the following models:  
     - Logistic Regression  
     - LDA  
     - KNN  
     - Decision Tree  

4. **Model Evaluation**  
   - Confusion matrix heatmaps to analyze performance.  
   - Bar chart visualization for **accuracy comparison** of all models.  

---

## **Tech Stack**  
- **Language**: Python  
- **Libraries**:  
   - `Pandas`, `Seaborn`, `Matplotlib`: Data processing and visualization  
   - `Scikit-Learn`: Machine learning models  
   - `SciPy`: Statistical analysis tools  
   - `Joblib`: Model saving and loading  

---

## **Directory Structure**  
```
loan-prediction-project/
│
├── data/                      # Contains the dataset
├── loan_analysis.py           # Main script for the project
├── loan_model.pkl             # Serialized trained model
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## **Setup Instructions**  
1. **Clone the repository**:  
   ```bash
   git clone <repository-url>
   cd loan-prediction-project
   ```

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**:  
   ```bash
   python loan_analysis.py
   ```

4. **Make Predictions with the Saved Model**:  
   ```python
   import joblib

   # Load the trained model
   model = joblib.load('loan_model.pkl')

   # Predict with new data
   new_data = [[5000, 2000, 150, 1, ...]]  # Example input
   result = model.predict(new_data)

   status = 'Approved' if result[0] == 1 else 'Rejected'
   print(f"Loan Approval Status: {status}")
   ```

---

## **Results**  
Below are the accuracy scores for the models tested:  
| **Model**                 | **Accuracy** |
|---------------------------|--------------|
| Logistic Regression       | 0.xx         |
| Linear Discriminant Analysis (LDA) | 0.xx |
| K-Nearest Neighbors       | 0.xx         |
| Decision Tree             | 0.xx         |

---

## **Visualizations Included**  
- **Box Plot**:  
   Displays the distribution and outliers across all numerical features.  

- **Loan Amount Distribution Histogram**:  
   Shows the loan amount's distribution with the mean and outlier threshold lines.  

- **Model Comparison Bar Chart**:  
   Highlights the performance (accuracy) of each machine learning model.  

---

## **Contributing**  
We welcome contributions!  
- **Fork** the repository.  
- Create a **new branch**.  
- Submit a **pull request** for improvements or bug fixes.  

---

## **License**  
This project is licensed under the **MIT License**. See the `LICENSE` file for more information.  

---

## **Acknowledgments**  
- **Kaggle**: For providing the dataset.  
- **Scikit-Learn and SciPy**: For machine learning and statistical tools.  
- **Matplotlib and Seaborn**: For powerful visualizations.
