import numpy as np
from flask import Flask, jsonify, request
import traceback
import paddle.v2 as paddle

app = Flask(__name__)


def errorResp(msg):
    return jsonify(code=-1, message=msg)

def successResp(data):
    return jsonify(code=0, message="success", data=data)

@app.route('/', methods=['POST'])
def mnist():
    global inferer
    feeding = {}
    d = []
    for i, key in enumerate(request.json):
        d.append(request.json[key])
        feeding[key] = i

    try:
        r = inferer.infer([d], feeding=feeding)
    except Exception as e:
        return errorResp(traceback.format_exc())
    return successResp(r.tolist())

if __name__ == '__main__':
    global inferer
    paddle.init()
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    predict = paddle.layer.fc(input=images, size=10, act=paddle.activation.Softmax())
    cost = paddle.layer.classification_cost(input=predict, label=label)
    parameters = paddle.parameters.create(cost)
    inferer = paddle.inference.Inference(output_layer=predict, parameters=parameters)
    app.run(host='0.0.0.0', port=80)
