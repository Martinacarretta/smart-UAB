import paho.mqtt.client as mqtt
import json
import sqlite3
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import plotly
import plotly.graph_objs as go

app = Flask(__name__)
socketio = SocketIO(app)

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

    end_device_ids = decoded_data.get("end_device_ids", {})  # Extract device info
    device_ids = end_device_ids.get("device_id")  # Get the device_id

    time = uplink_message.get("received_at")

    if decoded_payload is not None:
        #print(f"Received Data: Activity: {decoded_payload.get('activity')}, CO2: {decoded_payload.get('co2')}, Humidity: {decoded_payload.get('humidity')}, Illumination: {decoded_payload.get('illumination')}, Infrared: {decoded_payload.get('infrared')}, Infrared and Visible: {decoded_payload.get('infrared_and_visible')}, Pressure: {decoded_payload.get('pressure')}, Temperature: {decoded_payload.get('temperature')}, TVOC: {decoded_payload.get('tvoc')}")
        save_to_sqlite(decoded_payload, device_ids, time)
        q = display_data() # Return things after saving them

def save_to_sqlite(data, info, time):
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        #c.execute('''DROP TABLE received_data''')
        c.execute('''CREATE TABLE IF NOT EXISTS received_data (
                     device_id STRING default NULL, received_at TIMESTAMP default NULL, activity INTEGER default NULL, co2 INTEGER default NULL, humidity REAL default NULL,
                     illumination INTEGER default NULL, infrared INTEGER default NULL, infrared_and_visible INTEGER default NULL, pressure REAL default NULL,
                     temperature REAL default NULL, tvoc INTEGER default NULL);''')
        c.execute('INSERT INTO received_data (device_id, received_at, activity, co2, humidity, illumination, infrared, infrared_and_visible, pressure, temperature, tvoc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', 
                  (info, time, data.get('activity'), data.get('co2'), data.get('humidity'), data.get('illumination'), data.get('infrared'), data.get('infrared_and_visible'), data.get('pressure'), data.get('temperature'), data.get('tvoc')))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error in save_to_sqlite: {e}")

@app.route("/")
def index():
    #print(rows)
    #c=mqttc.loop(timeout = 180)
    return render_template("index.html")

def fetch_sensor_data(sensor_id):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT * FROM received_data WHERE device_id = ?", (sensor_id,))
    data = c.fetchall()
    conn.close()
    return data

def create_plots(sensor_data):
    print("Creating Plots")
    plots = []
    figures = []
    activity_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[2] for row in sensor_data], mode='lines', name='Activity')
    co2_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[3] for row in sensor_data], mode='lines', name='Co2')
    humidity_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[4] for row in sensor_data], mode='lines', name='Humidity')
    illumination_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[5] for row in sensor_data], mode='lines', name='Illumination')
    infrared_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[6] for row in sensor_data], mode='lines', name='Infrared')
    infrared_and_visible_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[7] for row in sensor_data], mode='lines', name='Infrared and visible')
    pressure_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[8] for row in sensor_data], mode='lines', name='Pressure')
    temperature_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[9] for row in sensor_data], mode='lines', name='Temperature')
    tvoc_trace = go.Scatter(x=[row[1] for row in sensor_data], y=[row[10] for row in sensor_data], mode='lines', name='Tvoc')

    plots.append([temperature_trace, co2_trace, humidity_trace, activity_trace, illumination_trace, infrared_trace, infrared_and_visible_trace, pressure_trace, tvoc_trace])
    
    for fig in plots:
        figures.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    print(figures)
    return figures

@app.route('/<sensorId>.html')
def sensor_page(sensorId):
    if sensorId == "index":
        return render_template("index.html")
    else: 
        sensors = ["eui-24e124710c408089", "eui-24e124128c147444", "eui-24e124128c147500", "eui-24e124128c147204", "eui-24e124128c147499", "am307-9074", "q4-1003-7456", "eui-24e124128c147446", "eui-24e124128c147470"]
        sensorID = sensors[int(sensorId)-1] #Change from number from 1-9 to actual ID to retrieve info from table
        # Fetch data for the specified sensor_id
        sensor_data = fetch_sensor_data(sensorID)
        # Create plots for the fetched sensor data
        figures = create_plots(sensor_data)
        return render_template(sensorId + '.html',figures=json.dumps(figures))

@app.route('/decoration.css')
def return_deco():
    return send_file('templates/decoration.css')

@app.route('/script.js')
def return_js():
    return send_file('templates/script.js')

def display_data():
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        c.execute('SELECT * FROM received_data;')
        #print(c.fetchone()[0])
        q = c.fetchall()
        for row in q:
            print(f"Id: {row[0]}, Date: {row[1]}, Activity: {row[2]}, CO2: {row[3]}, Humidity: {row[4]}, Illumination: {row[5]}, Infrared: {row[6]}, Infrared and Visible: {row[7]}, Pressure: {row[8]}, Temperature: {row[9]}, TVOC: {row[10]}")
        conn.close()
        return q
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
# mqttc.loop_forever()
mqttc.loop_start()

if __name__ == '_main_':
    socketio.run(app, debug=True)
    app.run(use_reloader = True, debug = True, port=3000)