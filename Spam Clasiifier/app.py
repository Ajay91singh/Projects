#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask,request, jsonify, render_template
import pickle


# In[2]:


app=Flask(__name__)
model=pickle.load(open('spam_model.pkl','rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	prediction = model.predict(vect)
    	return render_template('index.html', prediction_text="message is $ {}".format(prediction))


# In[5]:


if __name__=="__main__":
    app.run(debug=True)


# In[6]:





# In[ ]:




