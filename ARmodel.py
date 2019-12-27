#!/usr/bin/env python
# coding: utf-8

# # AR model estimation to any real data
# I used ”ISTANBUL STOCK EXCHANGE Data Set” in UCI Machine Learning Repository. ”The data is collected from imkb.gov.tr and finance.yahoo.com. Data is organized with regard to working days in Istanbul Stock Exchange.”
# I apply AR(2) model to ”NIKKEI” of the data set. The training set is the first 70% of the whole data. The test set is the last 30%. The following are my python code , the graph of real data, and the prediction graph.
# 

# In[41]:



# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import the data and the data matrix
df = pd.read_excel("/Users/daigofujiwara/Documents/授業資料/パターン認識特論/data_akbilgic.xlsx", header=1)
y = df["NIKKEI"].values[2:]
length = y.shape[0]
x = df["NIKKEI"].values[1:-1]
pre_x = df["NIKKEI"].values[:-2]
x = x.reshape([length, 1])
pre_x = pre_x.reshape([length, 1])
X = np.hstack((x, pre_x))

# split the real data to train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.1, shuffle=False)

# estimate the coefficients of AR model and predict in test set
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
n_train = len(X_train)
n_test = len(X_test)

# estimate the variance
sigma = np.std(y_test - y_pred)

lr.coef_


# In[36]:


# plot the whole real data
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(111)
ax.plot(df["date"], df["NIKKEI"].values, label="real")
ax.legend()
ax.set_xlabel("date")
ax.set_ylabel("NIKKEI")
plt.show()


# Figure 1: The whole real data. The red line is real data.

# In[40]:


# plot the prediction
fig = plt.figure(figsize=(13, 3))
ax = fig.add_subplot(111)
ax.plot(df["date"][n_train+2:], y_test, label="real")
ax.plot(df["date"][n_train+2:], y_pred+np.random.normal(0, sigma, n_test), label="AR(2) model")
ax.legend()
ax.set_xlabel("date")
ax.set_ylabel("NIKKEI")
plt.show()


# Figure 2: Prediction in test set. The blue line is the prediction. The red line is real data.

# In[13]:


y_pred


# In[ ]:




