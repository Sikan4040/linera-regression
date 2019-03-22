import numpy as np
import pandas as pd
from sklearn import linear_model
df=pd.read_csv('homeprices.csv')
median=df.bedrooms.median()
df.bedrooms=df.bedrooms.fillna(median)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
reg.predict(df.price)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[3000,3,40]]))
print(reg.predict([[2500,4,5]]))





