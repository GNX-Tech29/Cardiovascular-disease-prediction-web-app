import pandas as pd
import numpy as np
import pickle
import joblib
data = pd.read_csv('heart data.csv')

x = np.array(data.iloc[:,0:4])
y = np.array(data.iloc[:,4:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train,y_train)


joblib.dump(gaussian,'model_joblib_heart')

