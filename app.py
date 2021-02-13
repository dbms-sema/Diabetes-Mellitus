from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome to the Diabettes model API!"

# inputs
training_data = 'OriginalDataset.csv'
include = ['age', 'sex', 'Systollic Blood Pressure', 'Distollic Blood Pressure']
dependent_variable = include[-1]

model_directory = ''
model_file_name = '%sclassifier_model.pkl' % model_directory
model_columns_file_name = '%smodel_columns.pkl' % model_directory

# These will be populated at training time
model_columns = None
classifier_model = None


@app.route('/predict', methods=['POST'])
def predict():
    if classifier_model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))

            query = query.index(columns=model_columns, fill_value=0)

            prediction = list(classifier_model.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'
     
if __name__ == '__main__':
    #app.run(port=8080)
    try:
        port = int(sys.argv[1])
    except:
        port = 8080 
    classifier_model = joblib.load(model_file_name) # Load "classifier_model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(model_columns_file_name) # Load "model_columns.pkl"
    print ('Model columns loaded')
    print(model_columns)
    app.run(port=port, debug=True)