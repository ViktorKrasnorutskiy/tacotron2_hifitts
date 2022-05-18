from flask import Flask, jsonify, render_template, request, make_response
import time
from speecher import Speecher
import json
from config import conf

server_ip = conf['server_ip']


# Настройки сервера
app = Flask(__name__, template_folder='./')
app.debug = False


# Роутинг запросов по основной ссылке
@app.route('/', methods=['get'])
def home_page():
    data = {'server_ip': server_ip}
    return render_template('index.html', data=data)


# Роутинг post запросов
@app.route('/synthesize/', methods=['post'])
def synthesize():
    if request.method == 'POST':

        # Парсинг данных из запроса
        data = json.loads(request.data)
        text = str(data['text'])
        speaker = int(data['speaker'])

        # Вывод в консоль
        print(text, speaker)

        # Генерация аудио
        audio = speecher.synthesize(text, speaker)

        # Формирование ответа клиенту от сервера с указанием типа отправляемых данных
        response = make_response(audio)
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'inline; filename=sound.wav'
        response.headers.add("Access-Control-Allow-Origin", "*")

        return response


if __name__ == '__main__':

    # Инициализация модели
    speecher = Speecher()

    # Запуск сервера
    app.run(host = '0.0.0.0', port=5000)
