import numpy as np
from flask import Flask, abort, jsonify, request
import pickle

random_forest = pickle.load(open("leukemia_rfc_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST','GET'])
def predict():
    #all kinds of error checking should go here
    data = request.get_json(force=True)
    #convert json into numpy array
    predict_req = [data['WBC'],data['HGB'],data['NENO'],data['LYMNO'],data['MONO'],data['EONO'],data['BANO'],data['HCT'],data['MCV'],data['PLT']]
    predict_req = np.array(predict_req).reshape(1,-1)
    #np array goes into random forest , prediction comes out
    y_hat = random_forest.predict(predict_req)
    #return our prediction
    output = {'Condition': str(y_hat[0])}
    print(output)
    return jsonify(results=output)

if __name__ == "__main__":
    app.run(port = 5000, debug = True)
    # app.run(debug=True)

