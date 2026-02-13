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

ohe = OneHotEncoder(drop = 'first' , sparse_output = False )
x_train_sex_smoker_region = ohe.fit_transform(x_train[['sex' , 'smoker','region']])
x_test_sex_smoker_region = ohe.fit_transform(x_test[['sex' , 'smoker','region']])
# print(x_train_sex_smoker_region.shape)


x_train_age_bmi_children = x_train.drop(columns =['smoker', 'region','sex']).values
x_test_age_bmi_children = x_test.drop(columns =['smoker', 'region','sex']).values
# print(x_train_age_bmi_children.shape)

x_train_transformed = np.concatenate((x_train_age_bmi_children ,x_train_sex_smoker_region) , axis = 1)
print(x_train_transformed.shape)











