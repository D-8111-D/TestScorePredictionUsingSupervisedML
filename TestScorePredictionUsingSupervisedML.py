Deepshikha Prajapati

# Importing Libraries

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import linear_model 
%matplotlib inline

url="http://bit.ly/w-data"
df=pd.read_csv(url)
df

plt.title("Hours vs Percentage")
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.scatter(df['Hours'],df['Scores'])

# Preparing the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Hours']],df.Scores,train_size=0.8)

#  Training the Algorithm

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

plt.title("Hours vs Percentage")
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.scatter(df['Hours'],df['Scores'])
plt.plot(df.Hours,reg.predict(df[['Hours']]),color='blue')

# Making predictions

y_pred= reg.predict(X_test)
y_pred

reg.predict([[9.25]])

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

reg.score(X_test,y_test)

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))




