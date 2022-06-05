from flask import Flask, request, abort
from inspection import predict

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return "Hello world"
    
    elif request.method == 'POST':
        
        predict(request.json)

        return '', 200

    else:
        abort(405)


if __name__ == '__main__':
    app.run(debug=True)