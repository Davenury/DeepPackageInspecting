from flask import Flask, request, abort
from inspection import predict, prepare_models
from schedule import start

app = Flask(__name__)

prepare_models()

start()


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        
        # TODO - hanlde request somehow

        return '', 200

    else:
        abort(405)


if __name__ == '__main__':
    app.run(debug=True)