# THIS FILE RETRIEVES THE INFORMATION FROM THE THINGS NETWORK IN REAL TIME

import paho.mqtt.client as mqtt
import json
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt

app = Flask(__name__)
socketio = SocketIO(app)
rows = 0


# MQTT broker details
mqtt_host = "eu1.cloud.thethings.network"
mqtt_port = 1883

mqtt_username = "sensors-openlab@ttn"
mqtt_password = "NNSXS.AKU22YDKMFUNHGKYVJMMQVDLOBYMQBLAACLENZA.2KOWOPCVUXWCLN6SVP4LRZUYKBKMP7XDWB7OJHQENFMTVF4JSLFA"

# Handle connection events and incoming messages.
def on_message(client, userdata, message):
    decoded_data = json.loads(message.payload.decode("utf-8"))
    uplink_message = decoded_data.get("uplink_message", {})
    decoded_payload = uplink_message.get("decoded_payload")
    if decoded_payload is not None:
        #print(f"Received Data: Activity: {decoded_payload.get('activity')}, CO2: {decoded_payload.get('co2')}, Humidity: {decoded_payload.get('humidity')}, Illumination: {decoded_payload.get('illumination')}, Infrared: {decoded_payload.get('infrared')}, Infrared and Visible: {decoded_payload.get('infrared_and_visible')}, Pressure: {decoded_payload.get('pressure')}, Temperature: {decoded_payload.get('temperature')}, TVOC: {decoded_payload.get('tvoc')}")
        save_to_sqlite(decoded_payload)
        display_data()


def save_to_sqlite(data):
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        c.execute('''DROP TABLE received_data''')
        c.execute('''CREATE TABLE IF NOT EXISTS received_data (
                     received_at TIMESTAMP default CURRENT_TIMESTAMP, activity INTEGER default NULL, co2 INTEGER default NULL, humidity REAL default NULL,
                     illumination INTEGER default NULL, infrared INTEGER default NULL, infrared_and_visible INTEGER default NULL, pressure REAL default NULL,
                     temperature REAL default NULL, tvoc INTEGER default NULL);''')
        c.execute('INSERT INTO received_data (activity, co2, humidity, illumination, infrared, infrared_and_visible, pressure, temperature, tvoc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                  (data.get('activity'), data.get('co2'), data.get('humidity'), data.get('illumination'), data.get('infrared'), data.get('infrared_and_visible'), data.get('pressure'), data.get('temperature'), data.get('tvoc')))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error in save_to_sqlite: {e}")

@app.route("/")
def index():
    return render_template("index.html", rows = rows, len = 0)

def display_data():
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        c.execute('SELECT * FROM received_data;')
        #print(c.fetchone()[0])
        rows = c.fetchall()
        for row in rows:
            print(f"Date: {row[0]}, Activity: {row[1]}, CO2: {row[2]}, Humidity: {row[3]}, Illumination: {row[4]}, Infrared: {row[5]}, Infrared and Visible: {row[6]}, Pressure: {row[7]}, Temperature: {row[8]}, TVOC: {row[9]}")
        conn.close()
        with app.app_context():
            return render_template("index.html", rows = rows, len = len(rows))
    except Exception as e:
        print(f"Error in display_data: {e}")


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        client.subscribe("#")

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message

mqttc.user_data_set([])
mqttc.username_pw_set(mqtt_username, mqtt_password)
mqttc.connect(mqtt_host, mqtt_port)
mqttc.loop_forever()

if __name__ == '_main_':
    socketio.run(app, debug=True)
    app.run(use_reloader = True, debug = True, port=3000)