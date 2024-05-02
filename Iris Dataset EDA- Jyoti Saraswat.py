#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\DELL\Downloads\iris.csv', encoding='unicode_escape')  #to avoid encoding error, use unicode_escape


# In[4]:


df.shape                  


# In[5]:


df.head()


# In[6]:


df.info()


# In[9]:


df["species"].value_counts()


# ### Iris is a balanced dataset as the number of data points for every class is same
# 

# In[5]:


df.describe()


# # 2-D Scatter Plot

# In[11]:


df.plot(kind='scatter', x='sepal_length' , y='sepal_width');
plt.show()


# In[19]:


sns.set_style('whitegrid')
sns.FacetGrid(df, hue="species", height=5) \
    .map(plt.scatter, 'sepal_length', 'sepal_width') \
    .add_legend()

plt.show()


# #### Using sepal_length vs sepal_width plot we can distinguish between Setosa flowers with other flowers (i.e Versicolor & Virginica)

# # Heatmap

# In[6]:


# Heatmap can give correlation between different features
fig= plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), cmap='Blues' , annot=True)


# ## Observations
# #### Petal_length and petal_width- Corr 0.96: Having highest correlation
# #### Petal_length and sepal_length - Corr 0.87: Second Highest correlation
# #### Sepal_length and petal_width - Corr 0.82: Third Highest correlation

# # Distribution Plot

# In[3]:


#Distribution plot
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
fig = plt.figure(figsize=(20,4))
i = 1

for col in columns: 
    plt.subplot(1,4, i)
    sns.distplot(df[col])
    i = i + 1

plt.show()


# ## Observation
# #### SepalLength : Maximum distribution is around 5 to 6
# #### SepalWidth : Maximum distribution is around 3 
# #### PetalLength : Maximum distribution is around 4.5
# #### PetalWidth : Maximum distribution is around 1.5

# # Univariate analysis of features

# In[9]:


sns.FacetGrid(df, hue='species', height=4).map(sns.distplot, 'sepal_length').add_legend()


# #### We can not separate flowers just using sepal_length

# In[11]:


sns.FacetGrid(df, hue='species', height=4).map(sns.distplot, 'sepal_width').add_legend()


# #### We can not separate flowers just using sepal_width

# In[13]:


sns.FacetGrid(df, hue='species', height=4).map(sns.distplot, 'petal_length').add_legend()


# #### We can separate Setosa flowers easily just using petal_length.
# #### Versicolor and virginica can also partially seperated

# In[16]:


sns.FacetGrid(df, hue='species', height=4).map(sns.distplot, 'petal_width').add_legend()


# #### Setosa is well separable using petal_width
# #### Versicolor and virginica are partially separable using petal_width

# # Boxplot

# In[17]:


fig, axis = plt.subplots(2, 2, figsize=(20,10))

sns.boxplot(x='species', y='sepal_length', data=df, ax=axis[0,0])
sns.boxplot(x='species', y='sepal_width', data=df, ax=axis[0,1])
sns.boxplot(x='species', y='petal_length', data=df, ax=axis[1,0])
sns.boxplot(x='species', y='petal_width', data=df, ax=axis[1,1])

plt.show()


# #### Setosa : It's usually having smaller features except sepal_width
# #### Versicolor : It's having average features
# #### Virginica : It's having bigger features except sepal_width

# # Conclusion
# ### Data is balanced
# ### Strong correlation between petal_length and petal_width
# ### Setos is easily separable from other flowers, even using single feature
# ### Versicolor and virginica are difficult to seperate just using single feature
