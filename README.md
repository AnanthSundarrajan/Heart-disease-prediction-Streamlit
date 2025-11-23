# Heart Disease Prediction

**by Ananth Sundarrajan**

#### Executive summary
This project aims to predict the likelihood of a person having heart disease using their biometric and preliminary medical markers, such as Electrocardiogram (ECG) results, fasting blood sugar levels, and reported chest pain, thereby guiding the appropriate use of costly and invasive advanced diagnostics.

The prediction model uses the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, utilizing the Heart Failure Prediction Dataset from Kaggle.

Based on the analysis, a user facing application was built using “Streamlit” that can be accessed here. The application obtains the user’s information and runs a predictive model to assess the likelihood of the person having heart disease. 

#### Rationale
The significance of this project lies in addressing heart disease, the principal cause of mortality globally. Unlike acute trauma, such as bone fractures or contusions, heart disease progresses slowly and is often referred to as the "Silent Killer." Optimal management necessitates early detection and intervention. Current direct diagnostic methods, such as angiograms, are both costly and subject patients to potentially unnecessary radiation exposure. This project proposes utilizing non-invasive measures, specifically a person's biometric markers and preliminary tests, including Electrocardiogram (ECG) results, fasting blood sugar levels, and reported chest pain, to predict the probability of underlying heart disease.

Should an individual be predicted to have underlying heart disease, more advanced diagnostics, including both invasive procedures and medical imaging, can then be appropriately performed. Furthermore, this project aims to minimize unnecessary medical interventions for individuals who are not determined to be at risk for cardiac disease.

Furthermore, the prediction tool facilitates the early detection of cardiac disease, enabling primary intervention through proactive and personalized care, which can result in reduced costs for both the patient and insurance providers.
 

#### Research Question
Can we predict the likelihood of a person having heart disease based on their personal and medical parameters?

#### Data Sources
[Heart Failure Prediction Dataset from Kaggle] (https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

#### Methodology
Using the CRISP-DM Methodology, the predictive model uses the below methods:
- Data Pre-processing:
  * One-Hot Encoding
  * Standard Scaling
  * Data Splitting
- Model Training and Tuning:
  * Logistic Regression
  * Random Forest Classifier
  * Decision Tree Classifier
  * Support Vector Machine (SVC)
  * Gradient Boosting Classifier
  * Stacking Classifier (an ensemble method that combines the predictions of the other models)
- Hyperparameter Tuning: GridSearchCV
- Model Evaluation: F1 Score
- Model Persistence: joblib
- Front-end Application: A User facing application is created using “Streamlit” libraries and hosted on “Streamlit Community”. 

#### Results
- Exploratory Data Analysis
  * The DataFrame df comprises 918 entries and 12 columns
  * There are no missing entries
  * There are no duplicates
  * The Data types are distributed among 6 int64 columns (Age, RestingBP, Cholesterol, FastingBS, MaxHR, HeartDisease), 5 object columns (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope), and 1 float64 column (Oldpeak).
  * Based on the outlier analysis of the features, the following observations were made regarding outliers:
    - Age: No significant outliers were detected.
    - RestingBP: 28 outliers were identified, with values falling below 90 or exceeding 170.
    - Cholesterol: 183 outliers were found, corresponding to values below 32 or above 407.
    - MaxHR: 2 outliers were noted, with values below 66 or above 210.
    - Oldpeak: 16 were identified, with values below -2.25 or above 3.75.
    - Next Steps: Given that the objective is to determine the probability of heart diseases, it is expected that some patients in advanced stages of the condition will exhibit elevated values for Resting Blood Pressure (RestingBP), Cholesterol, and Oldpeak. **While these values may appear to be statistical outliers, they constitute valid recordings within the context of the problem statement and, consequently, should not be removed from the dataset.**
- Numerical feature distributions (visualized using histograms with KDEs), show varying patterns:
  * 'Age' and 'RestingBP' appear somewhat normally distributed
  * 'Cholesterol' shows a bimodal-like distribution with a significant number of zero values
  * 'Oldpeak' has a notable peak at zero.
  * 'MaxHR' shows a bell curve.
- Plots for categorical features show
  * Data set has more information about Male than Female that could lead to some bias in the prediction
  * 'FastingBS' shows that the majority of patients have a fasting blood sugar less than 120 mg/dl.
  * 'HeartDisease' target variable appears relatively balanced, suggesting an even distribution between positive and negative cases, which is good for prediction
- The correlation matrix of numerical features (visualized using a heatmap), indicates:
  * 'Age' and 'MaxHR' have a moderate negative correlation (Logical as age increases, the person's maxHR will decrease)
  * 'RestingBP' and 'Oldpeak' have weak positive correlations
- Data Pre-Processing
  * One-Hot Encoding: Categorical features like 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', and 'ST_Slope' are converted into a numerical format using pd.get_dummies.
  * Standard Scaling: Numerical features such as 'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', and 'Oldpeak' are normalized using StandardScaler from sklearn.preprocessing.
  * Data Splitting: The dataset is divided into features (X) and target (y) using standard pandas operations, and then further split into training and testing sets using train_test_split from sklearn.model_selection.
- Evaluation Metrics
  * The model's performance is assessed using F1-Score. A high F1-score indicates that the model has good performance in both identifying actual heart disease cases (high recall) and making accurate positive predictions (high precision). This balance is often desirable in medical applications where both false positives and false negatives have significant costs.
- Model Training
  * Both "Random Forest" and "Stacking Classifier" have relatively equal scores with a "high f1-score" score of 0.8920
  * But given Random Forest has lesser computation complication compared to Stacking classifier - needs to compute 5 methods and then run Logistic Regression on the result leading to a total of 6 methods - **Random Forest would be a better model for this data set.**
- Front-end Interface
  * [Front-end interface]() created using Streamlit and hosted on Streamlit community

#### Next steps
While this model does a good job of predicting heart disease, additional steps can be taken to improve the real-life usability
- Additional data sources can be added to the dataset to further improve the prediction metrics 
- Additional advance metrics - derived from simple blood test -  associated with heart diseases can be added to provide a better prediction, such as:
  * HDL (High Density Lipoprotein) value
  * LDL (Low Density Lipoprotein) value
  * VLDL (Very Low Density Lipoprotein) value
  * Apolipoprotein A
  * Apolipoprotein B
  * high-sensitivity C-reactive protein
- Since Gradient Boosting derived good results, advance methods such as XGBoosting can be used to assess if better scores are available 
Test the model on external or newer datasets to verify generalizability and robustness across populations.
- This prediction model can be utilized in conjunction with the "hotspotter" model to facilitate the early identification of "vulnerable patients." This proactive approach will enable the delivery of personalized care, thereby aiming to reduce overall healthcare expenditures for both patients and insurance providers.
 

#### Outline of project

- [Link to front end application]()
- [Streamlit - Link to .py file]()
- [Streamlit - Link to requirements file]()
- [Link to jupyter notebook]()
- [Link to project report]()
- [Link to dataset]()

#### Contact and Further Information
- For technical implementation details or model deployment guidance, please take a look at the complete analysis in the Jupyter Notebook.
- Author: Ananth Sundarrajan
- Program: UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence
- GitHub: https://github.com/AnanthSundarrajan
