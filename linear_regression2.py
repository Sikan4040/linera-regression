import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv('canada_per_capita_income.csv')
print(df.head())
#plt.scatter(df.year,df.income,color='blue',marker='+')
#plt.show()
print(type(df.year))
print(type(df.income))
x = np.array(df['year'])
y=np.array(df['income'])
x= x.reshape(-1,1)
reg=LinearRegression()
reg.fit(x,y)
print(reg.predict([[2020]]))
'''#print(type(x))
#y=df.income.astype(np.float)
#print(y)
#x=list(map(float,df.year))
reg=LinearRegression()
reg.fit(z,y)
#print(reg.predict([['2016']]))'''
