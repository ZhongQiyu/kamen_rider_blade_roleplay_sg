# api.py

from flask import Flask, request, jsonify
import json
from threading import Thread
import time

app = Flask(__name__)

messages = {}
processed_data_store = {}

def async_communication(agent_id, message, callback_url):
    time.sleep(2)  # Simulate delay
    if agent_id not in messages:
        messages[agent_id] = []
    messages[agent_id].append({'from': 'system', 'message': message})
    # Here you would make an HTTP request to the callback URL with the message
    # Example: requests.post(callback_url, json={'confirmation_message': 'Message received'})

def async_data_processing(data_id, raw_data, callback_url):
    time.sleep(5)  # Simulate processing delay
    processed_data_store[data_id] = {'processed_data': raw_data}
    # Here you would make an HTTP request to the callback URL with the processed data
    # Example: requests.post(callback_url, json={'processed_data': raw_data})

@app.route('/agent_comm', methods=['POST'])
def agent_comm():
    data = request.json
    agent_id = data['agent_id']
    message = data['message']
    callback_url = data.get('callback_url')

    if callback_url:
        thread = Thread(target=async_communication, args=(agent_id, message, callback_url))
        thread.start()
        return jsonify({'confirmation_message': 'Asynchronous communication initiated.'})
    else:
        if agent_id not in messages:
            messages[agent_id] = []
        messages[agent_id].append({'from': 'system', 'message': message})
        return jsonify({'confirmation_message': 'Communication initiated.'})

@app.route('/agent_comm', methods=['GET'])
def get_agent_messages():
    agent_id = request.args.get('agent_id')
    return jsonify({'messages': messages.get(agent_id, [])})

@app.route('/data_processor', methods=['POST'])
def data_processor():
    data = request.json
    raw_data = data['raw_data']
    data_id = str(len(processed_data_store) + 1)
    callback_url = data.get('callback_url')

    if callback_url:
        thread = Thread(target=async_data_processing, args=(data_id, raw_data, callback_url))
        thread.start()
        return jsonify({'processing_id': data_id})
    else:
        processed_data_store[data_id] = {'processed_data': raw_data}
        return jsonify({'processing_id': data_id})

@app.route('/data_processor/status', methods=['GET'])
def data_processor_status():
    processing_id = request.args.get('processing_id')
    status = 'completed' if processing_id in processed_data_store else 'processing'
    return jsonify({'status': status})

@app.route('/data_processor/result', methods=['GET'])
def data_processor_result():
    processing_id = request.args.get('processing_id')
    return jsonify(processed_data_store.get(processing_id, {}))

if __name__ == '__main__':
    app.run(debug=True)
