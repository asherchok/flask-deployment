import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)
    if output == 0:
        output_string = "From our predictions, the student would drop out."
    elif output == 1:
        output_string = "From our predictions, the student would be still enrolled."
    elif output == 2:
        output_string = "From our predictions, the student would graduate."
    else:
        output_string = "We cannot predict based on the unusual input values."

    return render_template('index.html', prediction_text=output_string)

if __name__ == "__main__":
    app.run(port=5000,debug=True)