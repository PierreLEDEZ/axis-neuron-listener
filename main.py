import math, logging, argparse
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        help="Choose the mode (training or recognition or web)",
        action="store",
        required=True,
        default="web"
    )
    
    args = parser.parse_args()

    if args.mode != "training" and args.mode != "recognition" and args.mode != "web":
        parser.print_help()
        sys.exit(-1)

    bvhListener = Client(HOST, PORT, args.mode)

    # With "web" mode, launch listener in its own thread and launch Flask application in the main thread
    if args.mode == "web":
        bvhListener.start()
        app.run()
        bvhListener.kill()
        exit()

    else:
        bvhListener.run()
        bvhListener.kill()
    