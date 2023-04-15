# AK_Teams
**Introduction**:-

The goal of the "Diabetes Prediction using SVM Machine Learning Model" project is to build 
a predictive model using Support Vector Machine (SVM) algorithm that can accurately predict 
whether or not a person has diabetes based on certain features.
The project can be broken down into the following steps:
1. Data Collection: The first step is to gather a dataset of patients with and without 
diabetes. The dataset should contain various features such as age, BMI, blood pressure, 
insulin level, etc.
2. Data Preprocessing: The dataset needs to be preprocessed to remove any missing 
values, normalize or standardize the data, and split it into training and testing sets.
3. Feature Selection: The features with the most impact on the prediction need to be 
selected. This can be done using statistical techniques like correlation analysis or 
feature importance ranking.
4. Model Training: The SVM model needs to be trained using the training dataset. The 
SVM algorithm tries to find the hyperplane that best separates the two classes of data. 
The SVM model can be optimized by tuning the hyperparameters using techniques like 
cross-validation.
5. Model Evaluation: The performance of the model needs to be evaluated using the 
testing dataset. The evaluation metrics that can be used include accuracy, precision, 
recall, F1-score, and confusion matrix.
6. Deployment: Once the model is trained and evaluated, it can be deployed to make 
predictions on new data.
Overall, the project aims to build a robust SVM-based diabetes prediction model that can be 
used to assist medical professionals in diagnosing diabetes and predicting its onset.

**Objective**:-

The main objective of the "Diabetes Prediction using SVM Machine Learning Model" project 
is to develop a predictive model that can accurately predict whether a person has diabetes or 
not based on certain features such as age, BMI, blood pressure, insulin level, etc. The project 
aims to achieve the following objectives:
1. To collect and preprocess the diabetes dataset.
2. To perform feature selection and identify the most important features that impact the 
prediction.
3. To train an SVM model on the dataset, and optimize the hyperparameters to improve 
the model's performance.
4. To evaluate the model's performance using various evaluation metrics such as accuracy, 
precision, recall, F1-score, and confusion matrix.
5. To deploy the model to make predictions on new data.
By achieving these objectives, the project can help medical professionals in diagnosing 
diabetes and predicting its onset. This can assist in providing early intervention and treatment, 
ultimately leading to better health outcomes for patients. Additionally, the project can also help 
researchers in understanding the various factors that contribute to diabetes and identifying potential risk factors.

**Technologies**:- 

1) Jupyter Notebook
2) Python Programming Language
3) SVM (Support Vectore Machine)

**Methodology**:- 

<img width="470" alt="Screenshot 2023-04-01 123531" src="https://user-images.githubusercontent.com/82876237/232225602-97eacbad-324a-4653-85c1-9c038868aa88.png">

1) Import Dependencies:- First we import some usefull libraries for our model likes, pandas, numpy,matplotlib,etc..
2) Import Dataset:- We use "Pima Indian Diabetes Dataset" for our model which is available on Kaggle.
3) Data Preprocessing:- In this, we remove the null & Duplicate Values from our dataset.
4) Splitting Data:- For model,we split data into two parts; one is training dataset which is 80% of our dataset and testing dataset which is 20% of dataset.
5) Model Training:- We use SVM for our model.We also use Confusion Matrix.
<img width="331" alt="Screenshot 2023-04-06 154342" src="https://user-images.githubusercontent.com/82876237/232225640-3f31433a-58e3-45cb-a808-b29114a9a708.png">
6) UI:- We Use Gradio to make the UI of the model.In this, You can give input like BMI,Insulin level,Glucose level, etc..
<img width="499" alt="Diabetes Prediction UI" src="https://user-images.githubusercontent.com/82876237/232226000-638e69e7-4a23-4139-a4bc-31494779cd11.png">



**Conclusion**:- 

The objective of the project was to develop a model which could identify patients with diabetes 
who are at high risk of hospital admission. Prediction of risk of hospital admission is a fairly 
complex task. Many factors influence this process and the outcome. There is presently a serious 
need for methods that can increase healthcare institution’s understanding of what is important 
in predicting the hospital admission risk. This project is a small contribution to the present 
existing methods of diabetes detection by proposing a system that can be used as an assistive 
tool in identifying the patients at greater risk of being diabetic. This project achieves this by 
analysing many key factors like the patient’s blood glucose level, body mass index, etc., using 
various machine learning models and through retrospective analysis of patients’ medical 
records. The project predicts the onset of diabetes in a person based on the relevant medical 
details that are collected using a desktop application. When the user enters all the relevant 
medical data required in the online Web application, this data is then passed on to the trained 
model for it to make predictions whether the person is diabetic or nondiabetic. The model is 
developed using artificial neural network consists of total of six dense layers. Each of these 
layers is responsible for the efficient working of the model. The model makes the prediction 
with an accuracy of 78%, which is fairly good and reliable.
