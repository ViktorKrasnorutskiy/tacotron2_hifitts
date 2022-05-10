# IMPORTS FOR SERVER
from flask import Flask, jsonify, render_template, request, make_response
import time
from speecher import Speecher
import json
from config import conf

server_ip = conf['server_ip']


# SETUP SERVER
app = Flask(__name__, template_folder='./')
app.debug = False


# MAIN LINK
@app.route('/', methods=['post', 'get'])
def home_page():
    data = {'server_ip': server_ip}
    return render_template('index.html', data=data)


# SYNTHESIZE METHOD
@app.route('/synthesize/', methods=['post', 'get'])
def synthesize():
    if request.method == 'POST':

        # PARSE REQUEST
        data = json.loads(request.data)
        text = str(data['text'])
        speaker = int(data['speaker'])
        print(text, speaker)

        # GENERATE AUDIO
        audio = speecher.synthesize(text, speaker)

        # MAKE RESPONSE
        response = make_response(audio)
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'inline; filename=sound.wav'
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response


if __name__ == '__main__':
    # INIT MODEL
    speecher = Speecher()
    # RUN SERVER
    app.run(host = '0.0.0.0', port=5000)
