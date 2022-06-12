from flask import Flask, request, abort, jsonify
from inspection import prepare_models
from schedule_job import start_scheduling

app = Flask(__name__)
prepare_models()
start_scheduling()


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