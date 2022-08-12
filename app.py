from cgitb import reset
from crypt import methods
from urllib import request
import numpy as np
import pickle

from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open('model.pkl','rb'))
    result=loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method =='POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if result>0.5:
            prediction="Placed"
        else:
            prediction = "Not Placed"

        return render_template('result.html',prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)