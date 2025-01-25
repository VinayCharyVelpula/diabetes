from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset and train the model
file_path = 'diabetes.csv'
diabetes_data = pd.read_csv(file_path)
X = diabetes_data.drop(columns=['Outcome'])
y = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(request.form[key]) for key in X.columns]
    new_patient = pd.DataFrame([data], columns=X.columns)

    # Predict
    prediction = rf_model.predict(new_patient)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
