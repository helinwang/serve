import os
import traceback

import paddle.v2 as paddle
from flask import Flask, jsonify, request

tarfn = os.getenv('PARAMETER_TAR_PATH', None)

if tarfn is None:
    raise ValueError("parameter tar file is not specific")

topology_filepath = os.getenv('TOPOLOGY_FILE_PATH', None)

if topology_filepath is None:
    raise ValueError("topology file is not specific")

port = int(os.getenv('PORT', '80'))

app = Flask(__name__)


def errorResp(msg):
    return jsonify(code=-1, message=msg)


def successResp(data):
    return jsonify(code=0, message="success", data=data)


@app.route('/', methods=['POST'])
def infer():
    global inferer
    try:
        feeding = {}
        d = []
        for i, key in enumerate(request.json):
            d.append(request.json[key])
            feeding[key] = i
        r = inferer.infer([d], feeding=feeding)
    except:
        trace = traceback.format_exc()
        return errorResp(trace)
    return successResp(r.tolist())


if __name__ == '__main__':
    paddle.init()
    with open(tarfn) as param_f, open(topology_filepath) as topo_f:
        params = paddle.parameters.Parameters.from_tar(param_f)
        inferer = paddle.inference.Inference(parameters=params, fileobj=topo_f)
    print 'serving on port ', port
    app.run(host='0.0.0.0', port=port)
