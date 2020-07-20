# flask_web/app.py

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hey, we have Flask in a Docker container'

@app.route('/hey')
def hello_world_2():
    return 'we might reference a script here..'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')