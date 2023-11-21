#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")


# In[2]:


# CSV 파일 로드
train = pd.read_csv('~/data/data/bike-sharing-demand/train.csv')


# In[3]:


train['datetime'] = train['datetime'].astype('str')
train['datetime'] = pd.to_datetime(train['datetime'])


# In[4]:


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second


# In[5]:


train.head()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1) # 2행 3열의 첫 번째 위치
sns.countplot(x='year', data=train)

# month 데이터 시각화
plt.subplot(2, 3, 2) # 2행 3열의 두 번째 위치
sns.countplot(x='month', data=train)

# day 데이터 시각화
plt.subplot(2, 3, 3) # 2행 3열의 세 번째 위치
sns.countplot(x='day', data=train)

# hour 데이터 시각화
plt.subplot(2, 3, 4) # 2행 3열의 네 번째 위치
sns.countplot(x='hour', data=train)

# minute 데이터 시각화
plt.subplot(2, 3, 5) # 2행 3열의 다섯 번째 위치
sns.countplot(x='minute', data=train)

# second 데이터 시각화
plt.subplot(2, 3, 6) # 2행 3열의 여섯 번째 위치
sns.countplot(x='second', data=train)


# In[7]:


X = train[['year','month','day','hour','minute','second','temp','humidity']].values
y = train['count'].values


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


model = LinearRegression()


# In[10]:


model.fit(X_train, y_train)


# In[11]:


predictions = model.predict(X_test)
predictions


# In[12]:


mse = mean_squared_error(y_test, predictions)
mse


# In[13]:


rmse = mse**0.5


# In[14]:


print(rmse)


# In[16]:


#temp
plt.scatter(X_test[:, 6], y_test, label="true")
plt.scatter(X_test[:, 6], predictions, label="pred")
plt.legend()
plt.show()


# In[17]:


#humidity
plt.scatter(X_test[:, 7], y_test, label="true")
plt.scatter(X_test[:, 7], predictions, label="pred")
plt.legend()
plt.show()


# In[ ]:




