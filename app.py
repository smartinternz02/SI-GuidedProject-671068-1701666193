import pickle 
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("knn.pkl", "rb"))

def preprocess_input(input_data):
    
    df = pd.DataFrame(input_data, index=[0])

    
    df['Coping_Mechanisms'] = pd.to_numeric(df['Coping_Mechanisms'], errors='coerce')
    df['stresss'] = pd.to_numeric(df['stresss'], errors='coerce')
    df['Demographics'] = pd.to_numeric(df['Demographics'], errors='coerce')
    df['Family_History'] = pd.to_numeric(df['Family_History'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['Impact_on_Life'] = pd.to_numeric(df['Impact_on_Life'], errors='coerce')
    df['Symptoms'] = pd.to_numeric(df['Symptoms'], errors='coerce')

    
    df = df.fillna(0)  

    return df

@app.route('/')
def indput():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def admin():
    try:
        input_data = preprocess_input(request.form.to_dict())

        input_data_for_prediction = input_data.values.reshape(1, -1)

        preds = model.predict(input_data_for_prediction)

        if preds[0] == 1:
            return render_template("index.html", p='Patient might face panic disorder')
        else:
            return render_template("index.html", p='Patient is normal')
    except ValueError as ve:
        error_message = f"ValueError: {ve}"
        print(error_message)
        return render_template("index.html", p=error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return render_template("index.html", p=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=4000)
