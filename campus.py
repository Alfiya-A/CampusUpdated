import numpy as np 
import pandas as pd 

df = pd.read_csv("/Users/alfia/Desktop/Campus/data.csv")

df["salary"]=df["salary"].fillna(0)

from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
ssc_b = LabelEncoder()
hsc_b = LabelEncoder()
hsc_s = LabelEncoder()
degree_t = LabelEncoder()
workex = LabelEncoder()
spec = LabelEncoder()
status= LabelEncoder()

df["Gender"] = le_gender.fit_transform(df.gender)
df["SSC_Board"] = ssc_b.fit_transform(df.ssc_b)
df["HSC_Board"] = hsc_b.fit_transform(df.hsc_b)
df["HSC_Stream"] = hsc_s.fit_transform(df.hsc_s)
df["Degree"] = degree_t.fit_transform(df.degree_t)
df["Experience"] = workex.fit_transform(df.workex)
df["Specialization"] = spec.fit_transform(df.specialisation)
df["Status"] = status.fit_transform(df.status)


df.drop(['sl_no', 'gender', 'ssc_b','hsc_b', 'hsc_s',
         'degree_t', 'workex', 'specialisation',
         'status'],axis=1,inplace=True)

df.rename(columns={'ssc_p':'SSC_P',
                  'hsc_p':'HSC_P',
                  'degree_p':'Degree_P',
                  'etest_p':'Etest_P',
                  'mba_p':'MBA_P',
                  'salary':'Salary'},inplace=True)


df = df[['Gender','SSC_P','SSC_Board','HSC_P','HSC_Board',
         'HSC_Stream','Degree','Degree_P','Experience',
         "Etest_P","Specialization","Status","Salary"]]


X = df.drop("Status",axis=1)
Y = df.Status

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.2)


import tensorflow
from tensorflow import keras
import keras_tuner
from keras import layers
from kerastuner.tuners import RandomSearch


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu,sigmoid


# Initialising the ANN
ann = Sequential()

ann.add(Dense(units = 12,
                     kernel_initializer = 'he_uniform',
                     activation='relu',
                     input_dim = 12))
ann.add(Dropout(0.2))

ann.add(Dense(units = 10,
                     kernel_initializer='he_uniform',
                     activation='relu'))
ann.add(Dropout(0.2))

ann.add(Dense(units = 8,
                     kernel_initializer='he_uniform',
                     activation='relu'))

ann.add(Dense(units = 1,
                     kernel_initializer = 'glorot_uniform',
                     activation = 'sigmoid'))


ann.compile(optimizer = 'Adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


ann.fit(x_train,y_train,validation_split=0.33,
        batch_size = 10,epochs = 100)


import pickle

with open('model_ann','wb') as f:
    pickle.dump(ann,f)


with open('model_ann','rb') as f:
    mp = pickle.load(f)


y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print(score)