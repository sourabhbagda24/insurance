import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv(r"C:\Users\Sourabh Sharma\OneDrive\Desktop\insurance_data\insurance - insurance.csv")
print(df.head(3))


print(df.columns)

from sklearn.model_selection import train_test_split
x=df.drop(columns=['charges'])
y=df['charges']

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
print(x_train.shape)


# ohe = OneHotEncoder(drop = 'first' , sparse_output = False )
# x_train_sex_smoker_region = ohe.fit_transform(x_train[['gender' , 'city' ]])

 
# x_test_gender_city = ohe.fit_transform(x_test[['gender' , 'city' ]])

# print(x_train_gender_city.shape) 








