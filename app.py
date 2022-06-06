from flask import Flask, request, abort, jsonify
# from inspection import predict, prepare_models
# from schedule import start

app = Flask(__name__)
# prepare_models()
# start()


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        country = request.get_json()['Country']
        print(f'New request for {country}')
        return jsonify({"Response": 'OK'})
    else:
        abort(405)


if __name__ == '__main__':
    app.run(debug=True)