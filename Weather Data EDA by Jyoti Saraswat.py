#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[118]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# # Load the data

# In[119]:


df=pd.read_csv(r'C:\Users\DELL\Downloads\file.csv')


# # Size of Data

# In[120]:


df.head()


# In[121]:


df.shape


# In[122]:


df.index


# In[123]:


df.columns


# In[124]:


df.dtypes


# In[125]:


df.count ()                        # it shows the total number of non-null  values in each column.


# In[126]:


df.info()


# In[127]:


df.describe()


# # Checking null values

# In[128]:


df.isnull().sum()                                         


# #### No null values present

# In[129]:


df.nunique()                # All the Unique values in the dataset 


# # Type of Weather

# In[130]:


df.Weather.value_counts()                               # checking values for weather column only


# In[131]:


df.Weather.unique()


# In[132]:


df.Weather.nunique()


# # Check for duplicates

# In[133]:


df.duplicated().sum()


# # Correlation among the features

# In[134]:


cols=['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']


# In[135]:


cor_matrix = df[cols].corr()
cor_matrix


# # Heat Map

# In[136]:


sns.heatmap(cor_matrix, annot = True)


# # Histogram

# In[137]:


df['Temp_C'].plot(kind='hist')


# In[138]:


df['Dew Point Temp_C'].plot(kind='hist')


# In[139]:


df['Rel Hum_%'].plot(kind='hist')


# In[140]:


df['Wind Speed_km/h'].plot(kind='hist')


# In[141]:


df['Visibility_km'].plot(kind='hist')


# In[142]:


df['Press_kPa'].plot(kind='hist')


# In[143]:


df['Temp_C'].plot(kind='box')                            # no outliers present


# In[144]:


df['Dew Point Temp_C'].plot(kind='box')                         # no outliers present


# In[145]:


df['Rel Hum_%'].plot(kind='box')


# In[146]:


df['Wind Speed_km/h'].plot(kind='box')


# In[147]:


df['Visibility_km'].plot(kind='box')


# In[148]:


df['Press_kPa'].plot(kind='box')


# # Label Encoding

# #### Converting Weather column (categorical column) into numeric.

# In[149]:


from sklearn.preprocessing import LabelEncoder


# In[150]:


label_encoder= LabelEncoder()


# In[151]:


df['Weather']= label_encoder.fit_transform(df['Weather'])


# In[152]:


label_encoder.classes_


# In[153]:


df.head()


# #### Weather column is converted into numeric

# In[154]:


# dropping Date/Time column
df.drop(["Date/Time"], axis=1,inplace=True)


# In[155]:


df.head()


# # X,y variables

# In[156]:


# Independent variable
x=df.drop(['Weather'] , axis=1)
x


# In[157]:


# Target variable
y=df['Weather']
y


# # Feature Scaling

# In[158]:


from sklearn.preprocessing import StandardScaler


# In[159]:


std_scaler= StandardScaler()


# In[160]:


x_std = std_scaler.fit_transform(x)
x_std                           


# # Splitting Data into training and testing:

# In[161]:


from sklearn.model_selection import train_test_split


# In[162]:


x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42)      
#test_size=0.2 means 20% data for testing & 80% data for training


# In[163]:


x_train.shape, x_test.shape


# # Model Building

# In[164]:


from sklearn.tree import DecisionTreeClassifier
# Create a decision tree classifier
decision_tree_model = DecisionTreeClassifier()


# # Model Training

# In[165]:


decision_tree_model.fit(x_train, y_train)


# # Model Predictions

# In[166]:


y_pred_dt= decision_tree_model.predict(x_test)


# # Model Evaluation

# In[167]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# # Accuracy

# In[168]:


accuracy_score(y_test, y_pred_dt)


# # Classification Report

# In[169]:


print(classification_report(y_test, y_pred_dt))


# # Confusion Matrix

# cm = confusion_matrix(y_test, y_pred_dt)
# sns.heatmap(cm, annot= True, fmt='d')

# # Regression algorithm

# #### Algorithm to predict visibility based on other collected measurement data 

# In[170]:


from sklearn.linear_model import LinearRegression


# In[171]:


model = LinearRegression()
model.fit(x_train, y_train)


# # Model Prediction

# In[172]:


predictions = model.predict(x_test)


# # Model Evaluation

# In[174]:


from sklearn.metrics import mean_squared_error


# In[175]:


mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# # Visualizing Results

# In[176]:


plt.scatter(y_test, predictions)
plt.xlabel("Actual Weather")
plt.ylabel("Predicted Weather")
plt.title("Actual vs. Predicted Weather")
plt.show()


# In[ ]:




