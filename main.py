import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

import paddle.v2 as paddle


def softmax_regression(img):
    predict = paddle.layer.fc(
        input=img, size=10, act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    # The first fully-connected layer
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # The second fully-connected layer and the according activation function
    hidden2 = paddle.layer.fc(
        input=hidden1, size=64, act=paddle.activation.Relu())
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = paddle.layer.fc(
        input=hidden2, size=10, act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def errorResp(msg):
    return jsonify(code=-1, message=msg)

def successResp(data):
    return jsonify(code=0, message="success", data=data)

@app.route('/', methods=['POST'])
def mnist():
    global parameters
    global predict

    d = np.array(request.json)
    if (d.dtype == np.dtype('int64')):
        d = d.astype(np.int32)
    elif (d.dtype == np.dtype('float64')):
        d = d.astype(np.float32)
    else:
        return errorResp("data type not supported: only supports lists of int or float.")

    r = inferer.infer([(d,)])
    return successResp(r.tolist()[0])

if __name__ == '__main__':
    global parameters
    global predict
    global infer
    paddle.init(use_gpu=False, trainer_count=1)
    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))
    # Here we can build the prediction network in different ways. Please
    # choose one by uncomment corresponding line.
    # predict = softmax_regression(images)
    # predict = multilayer_perceptron(images)
    predict = convolutional_neural_network(images)
    cost = paddle.layer.classification_cost(input=predict, label=label)
    parameters = paddle.parameters.create(cost)
    inferer = paddle.inference.Inference(output_layer=predict, parameters=parameters)

    app.run(host='0.0.0.0', port=80)
