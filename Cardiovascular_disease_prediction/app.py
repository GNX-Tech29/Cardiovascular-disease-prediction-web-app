from flask import Flask, render_template , url_for , request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('C:\BCA Project\.venv\model_joblib_heart')


@app.route("/")
def home():
    # Render the home page template
    return render_template("prediction.html")


@app.route("/predict", methods=["POST"])
def predict():
    

        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        cp =  int(request.form.get('cp'))
        trestbps = int(request.form.get('trestbps'))
        chol = int(request.form.get('chol'))
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form.get('restecg'))
        thalach = int(request.form.get('thalach'))
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))
        # Predict the probability of heart disease
        arr = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        pred = model.predict(arr)
        return render_template('after.html', data =pred)
    
@app.route("/tips")
def tips():
    # Render the home page template
    return render_template("tips.html")
    

    # Run the app
if __name__ == "__main__":
    app.run(debug=True)


