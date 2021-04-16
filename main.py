import math, logging
from flask import Flask, jsonify, make_response

from src.client import Client

HOST = "127.0.0.1"
PORT = 7001

app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.disabled = True

@app.route("/")
def root():
    raw_pred = bvhListener.predicting_thread.predictions

    if raw_pred != []:
        raw_pred = list(map(math.ceil, raw_pred*100))
    else:
        raw_pred = [0 for i in range(len(bvhListener.predicting_thread.classes))]

    response = make_response(jsonify({"predictions": raw_pred}), 200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.content_type = "application/json"

    return response


if __name__ == "__main__":
    bvhListener = Client(HOST, PORT)
    bvhListener.start()

    app.run()
    bvhListener.kill()
    