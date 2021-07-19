#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


# In[95]:


messages.head()


# In[96]:


messages.shape


# In[97]:


# Data Cleaning 

import re
import nltk


# In[98]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[99]:


ps=PorterStemmer()
corpus= []


# In[100]:


for i in range(len(messages)):
    review=re.sub('[^A-Za-z]', ' ', messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(msg) for msg in review if msg not in stopwords.words('english') ]
    review=' '.join(review)
    corpus.append(review)


# In[75]:


corpus


# In[104]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=4000)
X=cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# In[105]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[106]:


y


# In[107]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=0)


# In[108]:


X_train.shape


# In[109]:


X_test.shape


# In[110]:


# Train model using Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB
spam_model=MultinomialNB()
sp_model=spam_model.fit(X_train, y_train)


# In[111]:


y_pred=spam_model.predict(X_test)


# In[112]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[113]:


cm


# In[114]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)


# In[115]:


accuracy


# In[116]:


import pickle


# In[117]:


pickle.dump(spam_model,open('spam_model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




