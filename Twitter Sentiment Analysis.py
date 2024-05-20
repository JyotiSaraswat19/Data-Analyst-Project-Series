#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re                                                 # re is regular expression
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[12]:


import nltk
nltk.download('stopwords')


# In[13]:


print(stopwords.words('english'))            # printing the stopwords in english


# # Loading the data

# In[14]:


df = pd.read_csv(r'C:\Users\DELL\Downloads\archive (10)\twitter.csv',encoding = 'latin',header=None)


# # Preprocessing the Data

# In[15]:


df.shape


# In[16]:


df.head()


# #### Naming the columns

# In[17]:


columns = ['target', 'ids', 'date', 'flag', 'user', 'text']                    #reading the dataset again
df = pd.read_csv(r'C:\Users\DELL\Downloads\archive (10)\twitter.csv',names=columns,encoding = 'latin') 


# In[18]:


df.head()


# In[19]:


df.shape


# # Checking for Missing Values

# In[20]:


df.isnull().sum()


# #### No missing values in the dataset

# # Checking the distribution of target columns

# In[21]:


df['target'].value_counts()


# #### Data is equally distributed

# ## Converting the target level "4" to "1"

# In[22]:


df.replace({'target': {4: 1}}, inplace=True)


# In[23]:


df['target'].value_counts()


# # Stemming
# ####  It is a process to reduce words to their word stems or roots.

# In[24]:


from nltk.stem.porter import PorterStemmer
port_stem = PorterStemmer()


# In[25]:


def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', '' , content)  
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ''.join(stemmed_content)
  return stemmed_content


# In[26]:


df['stemmed_content'] = df['text'].apply(stemming)


# In[27]:


df.head()


# In[28]:


print(df['stemmed_content'])


# In[29]:


print(df['target'])


# In[30]:


# Separating the data and label
X= df['stemmed_content'].values
Y= df['target'].values


# In[31]:


print(X)


# In[32]:


print(Y)


# # Splitting the data to training data and test data

# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[34]:


print(X.shape, X_train.shape, X_test.shape)


# In[35]:


print(X_train)


# In[36]:


print(X_test)


# # Converting the textual data to numerical data

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train= vectorizer.fit_transform(X_train)
X_test= vectorizer.transform(X_test)


# In[38]:


print(X_train)


# In[39]:


print(X_test)


# # Training the logistic regression model

# In[40]:


model= LogisticRegression(max_iter=1000)


# In[41]:


model.fit(X_train, Y_train)


# # Model Evaluation

# In[42]:


# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[43]:


print('Accuracy', training_data_accuracy)


# In[44]:


# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[45]:


print('Accuracy', test_data_accuracy)


# # Saving the trained model

# In[46]:


import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[48]:


# Using the save model for prediction
loaded_model=pickle.load(open('trained_model.sav', 'rb'))


# In[49]:


X_new= X_test[200]
print(Y_test[200])
prediction=model.predict(X_new)
print(prediction)

if (prediction[0]== 0) :
    print('Negative Tweet')
else:
    print('Positive Tweet')


# In[ ]:




