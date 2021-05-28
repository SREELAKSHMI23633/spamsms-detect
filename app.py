# Importing essential libraries
from flask import Flask, render_template, request
pip install sklearn
from sklearn.svm import LinearSVC
import sklearn.svm.classes
import pickle

# Load the SVM model and CountVectorizer object from disk
filename = 'modelfile.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transformfile.pkl','rb'))
app = Flask(__name__)

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
	app.run(debug=True)
