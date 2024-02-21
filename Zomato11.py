#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os

# Use double backslashes or forward slashes without single quotes
new_directory = r'C:\Users\bluepal\Desktop\Python'

# Change the current directory
os.chdir(new_directory)

# Check the current directory
print(os.getcwd())


# In[3]:


Zomato_data=pd.read_csv('zomato.csv')


# In[4]:


Zomato_data.head()


# In[5]:


Zomato_data.tail()


# In[6]:


Zomato_data.info()


# In[7]:


Zomato_data.shape


# In[8]:


Zomato_data.columns


# In[9]:


Zomato_data.isnull().sum().sort_values(ascending=False)


# In[19]:


Zomato_data.dtypes


# In[10]:


# Redundant columns in a dataset refer to those columns that do not provide additional or valuable information for the analysis or modeling task at hand.


# In[11]:


##dropping columns 
Zomato_data = Zomato_data.drop(columns = ['url', 'address', 
       'phone', 'name', 'reviews_list', 'menu_item',
       ])


# In[12]:


Zomato_data.duplicated().sum()


# In[98]:


# inplace=True: This parameter, when set to True, specifies that the operation should be done in-place, meaning it modifies the original DataFrame. If inplace is not specified or set to False (the default), a new DataFrame with duplicates removed is returned, and the original DataFrame remains unchanged.


# In[13]:


Zomato_data.drop_duplicates(inplace=True)


# In[14]:


Zomato_data.isnull().sum().sort_values(ascending=False)


# In[15]:


# Impute with word 'NotAvailable'
Zomato_data.cuisines=Zomato_data.cuisines.fillna('NotAvailable')


# In[16]:


# Impute with word 'NotAvailable'
Zomato_data.rest_type=Zomato_data.rest_type.fillna('NotAvailable')


# In[17]:


# Impute with word 'NotAvailable'
Zomato_data.location=Zomato_data.location.fillna('NotAvailable')


# In[21]:


Zomato_data['votes']=Zomato_data['votes'].astype('int32')


# In[23]:


Zomato_data['rate'].isna().sum()


# In[24]:


#unique rating in rates columns
Zomato_data['rate'].unique()


# In[96]:


# Zomato_data['rate']=Zomato_data['rate'].str.replace(" ",""): This line removes spaces from the 'rate' column. It uses the str.replace method to replace all occurrences of a space (" ") with an empty string. 
# Zomato_data['rate']=Zomato_data['rate'].str.replace('-','NaN'): This line replaces hyphens ('-') in the 'rate' column with the string 'NaN'. This is likely done to handle missing or undefined values.
# Zomato_data['rate']=Zomato_data['rate'].fillna('NaN'): This line fills any remaining NaN (Not a Number) values in the 'rate' column with the string 'NaN'. 
# This line prints the unique values in the 'rate' column after the above transformations.


# In[26]:


#df.rates.str.replace('-',np.NaN)
Zomato_data['rate']=Zomato_data['rate'].str.replace(" ","")
Zomato_data['rate']=Zomato_data['rate'].str.replace('-','NaN')
Zomato_data['rate']=Zomato_data['rate'].fillna('NaN')
Zomato_data['rate'].unique()


# In[97]:


# Zomato_data['approx_cost(for two people)']=Zomato_data['approx_cost(for two people)'].str.replace(",",""): This line removes commas from the 'approx_cost(for two people)' column. It uses the str.replace method to replace all occurrences of a comma (",") with an empty string. This operation is common when dealing with currency values or numbers that use commas as thousands separators, as removing commas allows for proper numerical interpretation.
# Zomato_data['approx_cost(for two people)'].unique(): This line prints the unique values in the 'approx_cost(for two people)' column after the comma removal. 


# In[27]:


Zomato_data['approx_cost(for two people)']=Zomato_data['approx_cost(for two people)'].str.replace(",","")
Zomato_data['approx_cost(for two people)'].unique()


# In[28]:


Zomato_data.isnull().sum().sort_values(ascending=False)


# In[29]:


Zomato_data=Zomato_data.rename(columns={"approx_cost(for two people)" : "avg_cost",
                      "listed_in(type)" : "meal_type", 
                      "listed_in(city)" : "city"})


# In[30]:


Zomato_data.head()


# In[32]:


Zomato_data.dropna(subset=['rate','avg_cost','rest_type'],inplace=True)


# In[37]:


Zomato_data['avg_cost']=Zomato_data['avg_cost'].astype(int)


# In[95]:


Zomato_data.isnull().sum().sort_values(ascending=False)


# In[40]:


Zomato_data['rate']=Zomato_data['rate'].astype(float)


# In[41]:


##dropping columns 
Zomato_data = Zomato_data.drop(columns = ['location'
       ])


# In[42]:


import seaborn as sns


# In[43]:


plt.figure(figsize=(10, 10))

# Convert 'online_order' column to categorical data
Zomato_data['online_order'] = pd.Categorical(Zomato_data['online_order'])

# Plot the countplot
sns.countplot(x='online_order', data=Zomato_data)

plt.title("Number of restaurants that take online orders", fontsize=25, color='blue')
plt.ylabel("Count", fontsize=20)


# In[46]:


plt.figure(figsize=(10, 10))

# Convert 'book_table' column to categorical data
Zomato_data['book_table'] = pd.Categorical(Zomato_data['book_table'])

# Plot the countplot
sns.countplot(x='book_table', data=Zomato_data)

plt.title("Number of restaurants that have the option to book a table", fontsize=25, color='purple')
plt.ylabel("Count", fontsize=20)


# In[47]:


plt.figure(figsize=(10,10))
ax=Zomato_data.city.value_counts()[:10]
ax.plot(kind='bar')


# In[48]:


plt.figure(figsize=(16,16))

ax=Zomato_data.city.value_counts()
ax.plot(kind='pie',fontsize=20)

plt.title('number of restaurants in each area of banglore',fontsize=30,color='darkblue')
plt.show()


# In[49]:


plt.figure(figsize=(10,10))

ax=Zomato_data.rest_type.value_counts()[:10]
label=Zomato_data['rest_type'].value_counts()[:10].index
ax.plot(kind='pie',labels=label,autopct='%.2f')

plt.title("Type of Restaurant in City",fontsize=20,color='darkgreen')
plt.show()


# In[50]:


plt.figure(figsize=(10,10))

ax=Zomato_data.rest_type.value_counts()[:10]
ax.plot(kind='bar')

plt.title("Number of Type of Restaurant in City",fontsize=25)
plt.xlabel("Type of Restaurants",fontsize=15)
plt.show()


# In[52]:


plt.figure(figsize=(16,16))

sns.countplot(x=Zomato_data['rate'],hue=Zomato_data['online_order'])
plt.title("Rate VS Online order",fontsize=25)
plt.ylabel("Restaurants that Accept/Not Accepting online orders",fontsize=20)
plt.xlabel("rate",fontsize=20)
plt.show()


# In[56]:


Zomato_data=Zomato_data.drop(['dish_liked','meal_type'],axis=1)


# In[57]:


from sklearn.preprocessing import LabelEncoder


# In[58]:


le=LabelEncoder()


# In[59]:


list1=['online_order','book_table','rest_type','cuisines','city']


# In[60]:


for i in list1:
    Zomato_data[i]=le.fit_transform(Zomato_data[i])


# In[61]:


Zomato_data.shape


# In[63]:


#Drop null values
Zomato_data.dropna(how='any',inplace=True)
Zomato_data.shape


# In[64]:


# selecing features
features=Zomato_data.drop(['rate'],axis=1)
features.shape


# In[65]:


features.dtypes


# In[66]:


Zomato_data['rate'].unique()


# In[67]:


# selecting Class/ Label
label=Zomato_data['rate'].values
label


# In[71]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(features,label,test_size=0.10,random_state=42)


# In[72]:


# Check the shape of splited data
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[104]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# In[74]:


lin_reg=LinearRegression()
lin_reg


# In[75]:


lin_reg.fit(X_train,y_train)


# In[76]:


print(lin_reg.score(X_train,y_train))
print(lin_reg.score(X_test,y_test))


# In[80]:


# Decision Tree Regressor
decision_reg = DecisionTreeRegressor()
# Fit the model
decision_reg.fit(X_train, y_train)


# In[81]:


print(decision_reg.score(X_train,y_train))
print(decision_reg.score(X_test,y_test))


# In[106]:


# Random Forest Regressor
random_reg = RandomForestRegressor()
# Fit the model
random_reg.fit(X_train, y_train)


# In[107]:


print(random_reg.score(X_train,y_train))
print(random_reg.score(X_test,y_test))


# In[85]:


svr=SVR()
svr.fit(X_train,y_train)


# In[86]:


print(svr.score(X_train,y_train))
print(svr.score(X_test,y_test))


# In[89]:


rr=Ridge()
rr


# In[90]:


rr.fit(X_train,y_train)


# In[91]:


print(rr.score(X_train,y_train))
print(rr.score(X_test,y_test))


# In[92]:


lr=Lasso()
lr


# In[93]:


lr.fit(X_train,y_train) 


# In[94]:


print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))


# In[110]:


sample = pd.DataFrame({"Actual Rating": y_test, "Predicted Rating": np.round(rfr_pred, 2)})


# In[109]:


rfr_pred = random_reg.predict(X_test)


# In[112]:


sample


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




