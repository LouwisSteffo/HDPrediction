import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import Linear Discriminant Analysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_curve

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

def read_data():
    data = pd.read_csv('cleveland.csv', names=columns)
    # Replace missing values with mode group by their target
    # data['ca'] = data['ca'].fillna(data.groupby('target')['ca'].transform(lambda x: x.mode()[0]))
    # data['thal'] = data['thal'].fillna(data.groupby('target')['thal'].transform(lambda x: x.mode()[0]))
    
    data['ca'] = data['ca'].fillna(data['ca'].mode()[0])
    data['thal'] = data['thal'].fillna(data['thal'].mode()[0])

    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
    return data

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Heart Disease Prediction App")
    st.write("This app predicts the presence of heart disease based on various medical attributes.")
    
    data = read_data()
    # st.write(data.head())
    
    # st.write("Data Types:", data.dtypes)
    
    # st.write("Target Size: " , data.groupby('target').size())
    
    st.title('Correlation Matrix')
    plot = plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=2)
    st.pyplot(plot)
    
    #feature importance
    st.title('Feature Importance')
    model = ExtraTreesClassifier()
    model.fit(data.drop('target', axis=1), data['target'])
    feat_importances = pd.Series(model.feature_importances_, index = data.drop('target', axis=1).columns)
    plot = plt.figure(figsize=(20, 20))
    feat_importances.nlargest(13).plot(kind='barh')
    st.pyplot(plot)
    # st.write("")
    
    # st.write('Null Data Before Null Data Handling: ', data.isnull().sum())
    
 
    
    # st.write('Null Data After Null Data Handling: ', data.isnull().sum())
    
    # st.write("")
    
    # st.write('Shape of the data: ', data.shape)
    
    # st.write("")
    
    X = data.drop('target', axis=1) 
    y = data['target']
    
    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    model_list = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "Gaussian Discriminant Analysis": LinearDiscriminantAnalysis()  # Change to Linear Discriminant Analysis
    }

    model_name = st.sidebar.selectbox("Select Model", list(model_list.keys()))

    model = model_list[model_name]
    model.fit(X, y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    
    print('Model Name: ', model_name)
    #print f1 score
    print('F1 Score: ', f1_score(y_test, y_pred))
    
    #print accuracy score by dividing true positive + true negative by total
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    
    st.sidebar.write('Accuracy: ', accuracy_score(y_test, y_pred))
    
    # ROC Curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    # st.line_chart({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    
    # Sidebar for user input features
    st.sidebar.title('User Input Features')
    age = st.sidebar.slider('Age', 29, 77, 55)
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    cp = st.sidebar.slider('Chest Pain Type', 1, 4, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 250)
    fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.radio('Exercise Induced Angina', ['No', 'Yes'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    ca = st.sidebar.slider('Number of Major Vessels Colored by Flourosopy', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    sex = 1 if sex == 'Male' else 0
    fbs = 1 if fbs == 'True' else 0
    exang = 1 if exang == 'Yes' else 0
    restecg_mapping = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
    restecg = restecg_mapping[restecg]
    slope_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
    slope = slope_mapping[slope]
    thal_mapping = {'Normal': 3, 'Fixed Defect': 6, 'Reversible Defect': 7}
    thal = thal_mapping[thal]
    
    # Scale user input features
    input_data = sc.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    
    st.sidebar.write('Prediction: ', model.predict(input_data)[0])
    st.sidebar.write('Prediction Probability: ', model.predict_proba(input_data)[0][1])
    
if __name__ == '__main__':
    main()
