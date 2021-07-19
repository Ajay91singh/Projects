#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#get_ipython().run_line_magic('matplotlib', 'inline')

pd.pandas.set_option('display.max_columns',None)


# In[3]:


dataset = pd.read_csv('housing.csv')
dataset.head()


# In[10]:


X = dataset.iloc[:,:]
y = dataset.iloc[:,8]


# In[14]:


X=X.drop('median_house_value', axis=1)


# In[15]:


X


# In[17]:


# There can be a possibility of data leakage so we will split first and then apply feature engineering

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# In[19]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Missing Values

# In[20]:



features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 ]
for feature in features_nan:
    print("{}: {}% missing values".format(feature, np.round(dataset[feature].isnull().mean(),4)))


# In[21]:


#Replace the missing nan values

for feature in features_nan:
     ##We will replace by using median
    median_value = dataset[feature].median()
    print(median_value)
    ##create a new feature to capture nan values
    dataset[feature+'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_value, inplace = True)

dataset[features_nan].isnull().sum()


# In[22]:


dataset.head()


# ## Numerical Features

# In[23]:


#list of numerical variables

numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))
dataset[numerical_features].head()


# ## Relation between dependent and independent variable

# In[684]:


# for feature in numerical_features:
#     dataset.groupby(feature)['median_house_value'].median().plot()
#     plt.xlabel(feature)
#     plt.ylabel('Median House Price')
#     plt.title(feature)
#     plt.show()


# ## Discrete Feature

# In[24]:


## Numerical variables are usually of 2 types:
# Continuous variable and Discrete variable

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[25]:


discrete_feature


# ## Continuous Feature

# In[26]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature count {}".format(len(continuous_feature)))


# In[688]:


#Analyse continuous variable by creating histograms

# for feature in continuous_feature:
#     data=dataset.copy()
#     data[feature].hist(bins=25)
#     plt.xlabel(feature)
#     plt.ylabel("count")
#     plt.title(feature)
#     plt.show()


# ## Outliers

# In[27]:


##Only for continuous variable, can't be used for categorical variable
#
# for feature in continuous_feature:
#     data = dataset.copy()
#     if 0 in data[feature].unique():
#         pass
#     else:
#         data[feature]=np.log(data[feature])
#         data.boxplot(column=feature)
#         plt.ylabel(feature)
#         plt.title(feature)
#         plt.show()
#

# ## Handling Rare Categorical Feature

# In[28]:


categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[29]:


categorical_features


# In[30]:


temp=dataset.groupby(categorical_features)['median_house_value'].count()/len(dataset)
print(temp)
temp_df=temp[temp>0.01].index
print(temp_df)
dataset[categorical_features]=np.where(dataset[categorical_features].isin(temp_df),dataset[categorical_features],'Rare_var')


# In[31]:


dataset.head()


# In[32]:


for feature in categorical_features:
    labels_ordered=dataset.groupby(feature)['median_house_value'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# ## Correlation

# In[33]:


dataset.head()
dataset.drop(['total_bedroomsnan'], axis = 1)
dataset.iloc[:,0:].corr()


# In[521]:


# for feature in dataset.columns:
#     data=dataset.copy()
#     data.groupby(feature)['median_house_value'].median().plot.bar(color=['red', 'blue', 'green', 'yellow', 'orange'])
#     plt.xlabel(feature)
#     plt.ylabel('median_house_value')
#     plt.title(feature)
#     plt.show()


# In[ ]:





# ##  Feature Scaling

# In[36]:


feature_scale=[feature for feature in dataset.columns if feature not in ['median_house_value']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[35]:



data=pd.concat([dataset[['median_house_value']].reset_index(drop=True),
                  pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                 axis=1)


# In[21]:


data.head()


# In[22]:


data.to_csv('X_train.csv',index=False)


# In[23]:


# for feature in data.columns:
#     dataset.groupby(feature)['median_house_value'].median().plot()
#     plt.xlabel(feature)
#     plt.ylabel('Median House Price')
#     plt.title(feature)
#     plt.show()
#

# In[ ]:





# In[24]:


data.to_csv('X_train_bkp.csv', index=False)


# In[25]:


data_load_bkp = pd.read_csv('X_train_bkp.csv')


# In[26]:


data_load=data_load_bkp.copy()


# ## Removing Outliers

# In[27]:


from scipy import stats
z=np.abs(stats.zscore(data_load))
print(z)


# In[28]:


print(np.where(z>5))
data_load=data_load[(z<=5).all(axis=1)]


# In[29]:


data_load.shape


# In[30]:


# There can be a possibility of data leakage so we will split first and then apply feature engineering

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_load, data_load['median_house_value'], test_size=0.1, random_state=0)


# In[31]:


X_test=X_test.iloc[:,1:]
X_train=X_train.iloc[:,1:]


# In[32]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[33]:


y=data_load.iloc[:,1]


# In[ ]:





# ## Linear Regression

# In[34]:


data=data_load[['housing_median_age','population', 'total_rooms', 'latitude','longitude' , 'total_bedrooms','households' ,'median_income', 'ocean_proximity', 'median_house_value']]
data.head()

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train, y_test = train_test_split(data, data['median_house_value'], test_size=0.04, random_state=0)
X_train=X_train1.iloc[:,:-1]
X_test=X_test1.iloc[:,:-1]
X_train


# In[35]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

lin_regressor = LinearRegression()
#parameters = {'alpha':[30, 35,36,37,38,39, 40, 45], 'max_iter':[1700]}
#lin_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
mse = cross_val_score(lin_regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
mean_mse = np.mean(mse)
print(mean_mse)
lin_regressor.fit(X_train, y_train)
pickle.dump(lin_regressor, open('model_house_pred.pkl','wb'))
#print(lin_regressor.best_params_) #best_params_ gives which lambda value was more suitable
#print(lin_regressor.best_score_)
# acc=lin_regressor.score(X_test, y_test)
# print('acc: {}'.format(int(round(acc*100))))
# preds = lin_regressor.predict(X_test)
# preds.shape

model = pickle.load(open('model_house_pred.pkl','rb'))
print(model.predict([[2, 9, 6, 5, 7,8,4,3,2]]))

# In[36]:


#y_test


# In[37]:


# g=plt.scatter(y_test, preds)
# g.axes.set_yscale('log')
# g.axes.set_xscale('log')
# g.axes.set_xlabel('True Values ')
# g.axes.set_ylabel('Predictions ')
# g.axes.axis('equal')
# g.axes.axis('square')


# In[38]:


#g=plt.plot(y_test - preds,marker='o',linestyle='')


# # In[39]:


# from sklearn.metrics import mean_squared_error
# error = mean_squared_error(y_test, preds)
# print(error)
# from sklearn.metrics import r2_score
# error_r2 = r2_score(y_test, preds)
# print(error_r2)


# # In[40]:


# import seaborn as sns

# sns.distplot(y_test-preds)


# # ## Lasso Regression

# # In[41]:


# X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # In[42]:


# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
# lasso=Lasso()
# parameters = {'alpha':[0.01]}
# lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
# lasso_regressor.fit(X_train,y_train)
# print(lasso_regressor.best_params_) #best_params_ gives which lambda value was more suitable
# print(lasso_regressor.best_score_)


# # In[43]:


# preds1 = lasso_regressor.predict(X_test)
# sns.distplot(y_test-preds1)


# # In[44]:


# from sklearn.metrics import mean_squared_error
# error1 = mean_squared_error(y_test, preds1)
# print(error1)
# from sklearn.metrics import r2_score
# error_r21 = r2_score(y_test, preds1)
# print(error_r21)


# # ## Ridge Regression

# # In[45]:


# from sklearn.linear_model import Ridge


# ridge = Ridge()
# parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
# ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
# ridge_regressor.fit(X_train,y_train)
# preds2=ridge_regressor.predict(X_test)


# # In[46]:


# sns.distplot(y_test-preds2)


# # In[47]:


# error2 = mean_squared_error(y_test, preds2)
# print(error2)
# error_r22 = r2_score(y_test, preds2)
# print(error_r22)


# # ## Random Forest Regressor

# # In[126]:


# from sklearn.ensemble import RandomForestRegressor
# #from sklearn.datasets import make_regression
# regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)
# regressor.fit(X_train, y_train)


# # In[127]:


# preds3=regressor.predict(X_test)


# # In[128]:


# sns.distplot(y_test-preds3)


# # In[129]:


# error3 = mean_squared_error(y_test, preds3)
# print(error3)
# error_r23 = r2_score(y_test, preds3)
# print(error_r23)


# # ## XGBOOST Regressor

# # In[52]:


# import xgboost as xg
# xgb_r = xg.XGBRegressor(learning_rate=0.15, num_parallel_tree= 2, n_estimators=150, max_depth=15, objective ='reg:linear', reg_lambda= 2, seed = 123)
# xgb_r.fit(X_train, y_train)


# # In[53]:


# preds4 = xgb_r.predict(X_test)


# # In[54]:


# sns.distplot(y_test-preds4)


# # In[55]:


# error4 = mean_squared_error(y_test, preds4)
# print(error4)
# error_r24 = r2_score(y_test, preds4)
# print(error_r24)


# # ## Neural Network Regressor

# # In[130]:


# import tensorflow
# #tensorflow.random.set_seed(1)
# from tensorflow.python.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


# # In[134]:


# model = Sequential()
# model.add(Dense(20, input_dim=9, kernel_initializer='normal', activation='relu'))
# model.add(Dense(2670, activation='relu'))
# model.add(Dense(1, activation='linear'))
# model.summary()


# # In[135]:


# model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
# model.fit(X_train, y_train, epochs=20, batch_size=150, verbose=1, validation_split=0.9)
# predictions = model.predict(X_test)


# # In[136]:


# error_r25 = r2_score(y_test, predictions)
# print(error_r25)


# # In[ ]:




