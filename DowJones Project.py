#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns',None)


# In[3]:


df=pd.read_csv('DowJonesIndex.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


#Check the null values in dataset
df.isnull().sum()


# In[9]:


#Lets fill the missing values
df['percent_change_volume_over_last_wk'].fillna('Missing', inplace=True)  
df['previous_weeks_volume'].fillna('Missing', inplace=True) 


# In[10]:


#Lets check the missing values again
df.isnull().sum()


# AS we see there is no missing value in the dataset

# In[11]:


#Lets do the heatmap on dataset
sns.heatmap(df.isnull())


# In[12]:


#Lets plot the graph of stocks
df['volume'].plot(figsize=(10,6))


# In[13]:


#Lets plot the graph of stocks
df['percent_change_price'].plot(figsize=(10,6))


# In[14]:


df['percent_change_next_weeks_price'].plot(figsize=(10,6))


# In[15]:


#Lets clean the dataset 
get_ipython().run_line_magic('timeit', "df.open.str.replace('$','')")


# In[16]:


get_ipython().run_line_magic('timeit', "df.close.str.replace('$','')")


# In[17]:


get_ipython().run_line_magic('timeit', "df.high.str.replace('$','')")


# In[18]:


get_ipython().run_line_magic('timeit', "df.low.str.replace('$','')")


# In[19]:


#Lets see the dataset
df.head()


# In[20]:


df.info()


# In[21]:


df.dtypes


# In[22]:


# convert column of a dataframe
df.open.str.split('-').str[0][0]


# In[25]:


df['open']=df['open'].apply(lambda x:x.split('$')[1])


# In[26]:


df['open'].apply(lambda x:x.replace('$',''))


# In[27]:


df['close']=df['close'].apply(lambda x:x.split('$')[1])


# In[28]:


df['close'].apply(lambda x:x.replace('$',''))


# In[32]:


df['high']=df['high'].apply(lambda x:x.split('$')[1])


# In[33]:


df['high'].apply(lambda x:x.replace('$',''))


# In[34]:


df['low']=df['low'].apply(lambda x:x.split('$')[1])


# In[35]:


df['low'].apply(lambda x:x.replace('$',''))


# In[36]:


df.head()


# In[37]:


df.dtypes


# In[38]:


#convert column of a dataframe
df['open']=df['open'].replace(',','').astype(float)


# In[41]:


df['close']=df['close'].replace(',','').astype(float)


# In[42]:


df['high']=df['high'].replace(',','').astype(float)


# In[43]:


df['low']=df['low'].replace(',','').astype(float)


# In[53]:


#Lets visualize the closing price history
plt.figure(figsize=(10,6))
plt.title('closing price history')
plt.plot(df['close'])
plt.xlabel('date',fontsize=10)
plt.ylabel('close price USD ($)', fontsize=10)
plt.show()


# In[54]:


df.describe


# In[56]:


import math
#Create the dataframe with only close column
data=df.filter(['close'])
#convert the dataframe to a numpy array
dataset=data.values
#Get the number of rows to train the model on
training_data_len = math.ceil( len(dataset) * .8)

training_data_len


# In[59]:


from sklearn.preprocessing import MinMaxScaler
#Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


# In[60]:


#Create the training dataset 
#create the scaled training dataset
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train datasets
x_train = []
y_train = []
for i in range (60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[61]:


#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[64]:


#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[74]:


from keras.models import Sequential
from keras.layers import Dense,LSTM
#Build the LSTM model
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[77]:


#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[78]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[79]:


#create the testing data
#create a new array containing scaled values from index
test_data = scaled_data[training_data_len - 60: :]
#Create the datasets x_test and y_test
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[80]:


#convert the data to a numpy array
x_test = np.array(x_test)


# In[81]:


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[83]:


#Get the models predicted price values 
predictions  = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[84]:


#Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean( predictions - y_test )**2 )
rmse


# In[86]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions
#visualize the data
plt.figure(figsize=(10,6))
plt.title('model')
plt.xlabel('Date',fontsize=10)
plt.ylabel('close price USD $', fontsize=10)
plt.plot(train['close'])
plt.plot(valid[['close','predictions']])
plt.legend(['train','val','predictions'], loc='lower right')
plt.show()


# In[87]:


#show the valid and predicted price
valid


# In[ ]:


#Get the quote

