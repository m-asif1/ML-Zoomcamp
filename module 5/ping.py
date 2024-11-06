#def ping():
#    return "PONG"

# python -m pip install flask
# python -m pip install scikit-learn
# http://localhost:9696/ping

from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])

def ping():
    return "PONG"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)