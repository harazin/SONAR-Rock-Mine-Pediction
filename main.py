import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading The Dataset To A Pandas Dataframe
SONAR_DATA_PATH = '/content/sonar-data.csv'
sonar_data = pd.read_csv(SONAR_DATA_PATH, header=None)

#Seperating Data & Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

#Segregate Training & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

#Training The Logistic Regression Model With Training data
model = LogisticRegression()
model.fit(X_train, Y_train)

def predict_result(input_data):
  #Convert DataType Of input_data To Numpy Array
  input_data_as_numpy_array = np.asarray(input_data)

  #Reshape Numpy Array As We Are Predicting For Only 1 Instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

  #Return Prediction
  prediction = model.predict(input_data_reshaped)
  if prediction[0]=='R':
    return 'The object is a Rock.'
  else:
    return 'The object is a Mine.'
 
