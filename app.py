#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing essential libraries

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# Load the naive bayes model and CountVectorizer object from disk
classifier = pickle.load(open('modelfile.pkl', 'rb'))
cv = pickle.load(open('transformfile.pkl','rb'))


@app.route('/')
def home():
	return render_template('home_page.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run()


# In[ ]:




