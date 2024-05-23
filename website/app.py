# Imports
import paho.mqtt.client as mqtt
import json
import sqlite3
from flask import Flask, render_template, send_file, request, jsonify
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import plotly
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import base64
import os
from io import BytesIO
from datetime import datetime
from datetime import timedelta

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)


# MQTT settings:
mqtt_host = "eu1.cloud.thethings.network"
mqtt_port = 1883
mqtt_username = "sensors-openlab@ttn"
# mqtt_password = "NNSXS.AKU22YDKMFUNHGKYVJMMQVDLOBYMQBLAACLENZA.2KOWOPCVUXWCLN6SVP4LRZUYKBKMP7XDWB7OJHQENFMTVF4JSLFA"
mqtt_password = "NNSXS.KMPUFJE3MR5OVSKWJHUQQITO7CVJTI5O6IRVCSI.NTELY7IRGYY3GV6D3WK42WPSKQ3HLMZBOEEZ46MFDHCJPKW7TRUQ"


# MQTT event handlers: Handle connection events and incoming messages.
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

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        client.subscribe("#")

# Utility functions
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

# Routes
@app.route("/")
def index():
    #print(rows)
    #c=mqttc.loop(timeout = 180)
    return render_template("index.html")

@app.route('/<sensorId>.html')
def sensor_page(sensorId):
    if sensorId == "index":
        return render_template("index.html")
    elif sensorId == "predictions":
        return render_template("predictions.html")
    else: 
        sensors = ["eui-24e124710c408089", "eui-24e124128c147444", "eui-24e124128c147500", "eui-24e124128c147204", "eui-24e124128c147499", "am307-9074", "q4-1003-7456", "eui-24e124128c147446", "eui-24e124128c147470"]
        try:
            sensorID = sensors[int(sensorId)-1] #Change from number from 1-9 to actual ID to retrieve info from table
        except ValueError:
            return jsonify({'error': 'Invalid sensorId'}), 400
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

################################## PREDICTIONS ##################################
def predict_energy_occupation(df, result_mode, month, week_day, hour=None):
    if hour is None:
        filtered_df = df[(df['Month'] == month) & (df['Week_day'] == week_day)]
    else:
        filtered_df = df[(df['Month'] == month) & (df['Week_day'] == week_day) & (df['Hour'] == hour)]

    if not filtered_df.empty:
        if result_mode == 'energy-consumption':
            avg_value = filtered_df['Energy consumption [kwh]'].mean()
            avg_value = avg_value * (1 if hour is None else 0.8)
        elif result_mode == 'occupation':
            period_mask = (filtered_df['Date'] >= pd.Timestamp('2020-01-09')) & (filtered_df['Date'] <= pd.Timestamp('2022-01-07'))
            filtered_df.loc[period_mask, 'Occupation'] *= 2
            avg_value = filtered_df['Occupation'].mean()
        else:
            return "No data available for the specified parameters"
        return avg_value
    else:
        return "No data available for the specified parameters"

def predict_until(df, df_mode, result_mode, end_date):
    start_value = df['Date'].max() + pd.Timedelta(days=1) if df_mode == 'daily' else df['Date'].max() + pd.Timedelta(hours=1)
    date_range = pd.date_range(start=start_value, end=end_date, freq='D' if df_mode == 'daily' else 'h')

    future_dates = pd.DataFrame(date_range, columns=['Date'])
    future_dates['Week_day'] = future_dates['Date'].dt.day_name()
    future_dates['Month'] = future_dates['Date'].dt.month_name()
    if df_mode == 'hourly':
        future_dates['Hour'] = future_dates['Date'].dt.hour

    predictions = []
    for _, row in future_dates.iterrows():
        month = row['Month']
        week_day = row['Week_day']
        hour = row['Hour'] if df_mode == 'hourly' else None
        prediction = predict_energy_occupation(df, result_mode, month, week_day, hour)
        predictions.append(prediction)

    future_dates['predicted_' + result_mode] = predictions
    return future_dates


def visualise_prediction(df, df_mode, result_mode):
    plt.figure(figsize=(10, 6))
    dates = df['Date']
    if result_mode == 'energy-consumption':
        plt.plot(dates, df['predicted_energy-consumption'], label='Predicted Energy Consumption', color='red')
    elif result_mode == 'occupation':
        plt.plot(dates, df['predicted_occupation'], label='Predicted Occupation', color='lightgreen')
    
    # Determine the number of ticks
    if df_mode == 'daily':
        p = len(df) // 15
    elif df_mode == 'hourly':
        p = len(df) // 24
    
    # Ensure the last date is included in the ticks
    if p == 0:
        p = 1  # Avoid division by zero if df is too small
    ticks = dates[::p].tolist()
    if dates.iloc[-1] not in ticks:
        ticks.append(dates.iloc[-1])
    
    plt.xticks(ticks, rotation=45)
    plt.title(df_mode + ' ' + result_mode + ' forecast')
    plt.xlabel('Date')
    plt.ylabel(result_mode.replace('-', ' ').capitalize())
    plt.legend()
    plt.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return image_base64


def prediction(df, df_mode, result_mode, end_date):
    future_dates = predict_until(df, df_mode, result_mode, end_date)
    image_base64 = visualise_prediction(future_dates, df_mode, result_mode)
    return future_dates, image_base64

@app.route('/predictions.html', methods=['POST'])
def create_prediction():
     # Get the directory of the current script
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the paths to the CSV files
    oce_d_path = os.path.join(current_dir, 'occupation_energy_daily.csv')
    oce_h_path = os.path.join(current_dir, 'occupation_energy_hourly.csv')

    oce_d = pd.read_csv(oce_d_path)
    oce_h = pd.read_csv(oce_h_path)
    
    oce_d['Date'] = pd.to_datetime(oce_d['Date'])
    oce_h['Date'] = pd.to_datetime(oce_h['Date'])

    data = request.json
    result_type = data['result-type']
    mode = data['mode']
    date = data['date']

    if mode == 'daily':
        df = oce_d
    elif mode == 'hourly':
        df = oce_h
    else:
        return jsonify({'error': 'Invalid mode'}), 400

    end_date = datetime.strptime(date, '%Y-%m-%d')

    print(mode, result_type, end_date)
    future_dates, image_base64 = prediction(df, mode, result_type, end_date)
    print(future_dates)
    return jsonify({'prediction': future_dates.to_dict(), 'image': image_base64})

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