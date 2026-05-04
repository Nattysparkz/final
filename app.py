import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAX_PLATFORMS'] = 'cpu'

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import math
import time
import threading
from datetime import datetime, timedelta
import requests
import urllib3
from sklearn.preprocessing import MinMaxScaler
import psycopg2
import psycopg2.extras

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__, template_folder='templates')
CORS(app)

API_KEY = os.environ.get("RAIL_API_KEY", "")
DB_URL = os.environ.get("DATABASE_URL")
REST_BASE_URL = "https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120"
MANCHESTER_LINE = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']

ROUTE_STATIONS = {
    'MAN': {'name': 'Manchester Piccadilly', 'crs': 'MAN', 'lat': 53.4774, 'lon': -2.2309},
    'SPT': {'name': 'Stockport', 'crs': 'SPT', 'lat': 53.4052, 'lon': -2.1634},
    'MAC': {'name': 'Macclesfield', 'crs': 'MAC', 'lat': 53.2586, 'lon': -2.1253},
    'SOT': {'name': 'Stoke-on-Trent', 'crs': 'SOT', 'lat': 52.9905, 'lon': -2.1814},
    'CRE': {'name': 'Crewe', 'crs': 'CRE', 'lat': 53.0876, 'lon': -2.4317},
    'RUG': {'name': 'Rugby', 'crs': 'RUG', 'lat': 52.3785, 'lon': -1.2502},
    'MKC': {'name': 'Milton Keynes Central', 'crs': 'MKC', 'lat': 52.0345, 'lon': -0.7740},
    'EUS': {'name': 'London Euston', 'crs': 'EUS', 'lat': 51.5282, 'lon': -0.1337},
}

class JaxRouteModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x); x = nn.relu(x)
        x = nn.Dense(32)(x); x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

def get_db():
    return psycopg2.connect(DB_URL)

def init_db():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rail_events (id SERIAL PRIMARY KEY, event_datetime TIMESTAMP, pfpi_minutes DOUBLE PRECISION, non_pfpi_minutes DOUBLE PRECISION, source_file TEXT);
                CREATE INDEX IF NOT EXISTS idx_event_datetime ON rail_events(event_datetime);
                CREATE TABLE IF NOT EXISTS jax_predictions (id SERIAL PRIMARY KEY, forecast_date DATE, predicted_minutes DOUBLE PRECISION, stress_coefficient DOUBLE PRECISION, data_type TEXT, rmse DOUBLE PRECISION, updated_at TIMESTAMP DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS live_snapshot (id SERIAL PRIMARY KEY, station TEXT, delay_minutes DOUBLE PRECISION, status TEXT, total_delayed_trains INTEGER, total_delay_minutes DOUBLE PRECISION, updated_at TIMESTAMP DEFAULT NOW());
            """)
        conn.commit(); conn.close()
        print("✅ Database tables ready.")
    except Exception as e:
        print(f"⚠️ Database init error: {e}")

def _iso_to_hhmm(iso_str):
    if not iso_str or not isinstance(iso_str, str): return ''
    try:
        if 'T' in iso_str: return iso_str.split('T')[1][:5]
        elif len(iso_str) >= 5 and ':' in iso_str: return iso_str[:5]
    except: pass
    return ''

def scan_live_departures():
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    results = []; total_delay = 0; total_delayed = 0
    for station in MANCHESTER_LINE:
        current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{station}/{current_time}?numRows=10&timeWindow=120"
        station_delay = 0; station_status = "⚠️"
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                data = response.json(); services = data.get('trainServices', [])
                if not services: results.append({'station': station, 'delay': 0, 'status': '⚪ None'}); continue
                for train in services[:6]:
                    std = train.get('std', 'N/A'); etd = train.get('etd', 'N/A'); atd = train.get('atd', 'N/A')
                    is_cancelled = train.get('isCancelled', False)
                    flag = atd if atd != 'N/A' else etd; flag_lower = str(flag).lower().strip()
                    if is_cancelled or flag_lower == 'cancelled': station_delay += 60; total_delayed += 1
                    elif flag_lower == 'delayed': total_delayed += 1
                    elif flag_lower not in ['on time', 'n/a', 'no report'] and std != 'N/A':
                        try:
                            dt_std = datetime.fromisoformat(str(std)); dt_flag = datetime.fromisoformat(str(flag))
                            delay = (dt_flag - dt_std).total_seconds() / 60
                            if delay < -720: delay += 1440
                            elif delay > 720: delay -= 1440
                            if delay > 0: station_delay += delay; total_delayed += 1
                        except: pass
                total_delay += station_delay
                station_status = f"🔴 {int(station_delay)}m" if station_delay > 0 else "🟢 OK"
            else: station_status = f"⚠️ {response.status_code}"
        except: station_status = "⚠️"
        results.append({'station': station, 'delay': station_delay, 'status': station_status}); time.sleep(0.5)
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM live_snapshot;")
            for r in results:
                cur.execute("INSERT INTO live_snapshot (station, delay_minutes, status, total_delayed_trains, total_delay_minutes) VALUES (%s,%s,%s,%s,%s)",
                    (r['station'], float(r['delay']), r['status'], int(total_delayed), float(total_delay)))
        conn.commit(); conn.close()
    except Exception as e: print(f"⚠️ Could not save live snapshot: {e}")
    return results, total_delayed, total_delay

def run_jax_training():
    print("\n🧠 --- STARTING JAX TRAINING ---")
    try:
        conn = get_db()
        df = pd.read_sql('SELECT event_datetime as "EVENT_DATETIME", pfpi_minutes as "PFPI_MINUTES", non_pfpi_minutes as "NON_PFPI_MINUTES" FROM rail_events ORDER BY event_datetime', conn)
        conn.close()
    except Exception as e: print(f"❌ Database error: {e}"); return
    if df.empty: print("❌ No data."); return
    print(f"✅ Loaded {len(df)} rows.")
    df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], errors='coerce')
    df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')
    if 'NON_PFPI_MINUTES' not in df.columns: df['NON_PFPI_MINUTES'] = 0
    if 'PFPI_MINUTES' not in df.columns: df['PFPI_MINUTES'] = 0
    daily = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES', 'NON_PFPI_MINUTES']].sum().reset_index()
    daily['TOTAL'] = daily['PFPI_MINUTES'] + daily['NON_PFPI_MINUTES']
    daily['EVENT_DATETIME'] = pd.to_datetime(daily['EVENT_DATETIME'])
    total_days = len(daily)
    if total_days < 15: print(f"❌ Not enough data ({total_days} days)."); return
    dataset = daily[['TOTAL']].values; training_data_len = math.ceil(total_days * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1)); scaled_data = scaler.fit_transform(dataset)
    window_size = 14; train_data = scaled_data[:training_data_len]
    x_train, y_train = [], []
    for i in range(window_size, len(train_data)): x_train.append(train_data[i-window_size:i, 0]); y_train.append(train_data[i, 0])
    x_train_jnp = jnp.array(x_train).reshape(-1, window_size, 1); y_train_jnp = jnp.array(y_train).reshape(-1, 1)
    model = JaxRouteModel(); key = random.PRNGKey(42)
    variables = model.init(key, jnp.ones((1, window_size, 1))); params = variables['params']
    optimizer = optax.adam(learning_rate=0.001); opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p): preds = model.apply({'params': p}, x); return jnp.mean((preds - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params); updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss
    for epoch in range(51):
        params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
        if epoch % 10 == 0: loss.block_until_ready(); print(f"  Epoch {epoch:02d} | Loss: {loss:.6f}")
    test_data = scaled_data[training_data_len - window_size:]
    x_test = [test_data[i-window_size:i, 0] for i in range(window_size, len(test_data))]
    x_test_jnp = jnp.array(x_test).reshape(-1, window_size, 1); y_test = dataset[training_data_len:]
    preds_scaled = model.apply({'params': params}, x_test_jnp)
    test_predictions = scaler.inverse_transform(np.array(preds_scaled))
    rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2)); print(f"  RMSE: {rmse:.2f}")
    future_days = 30; future_predictions = []; current_window = scaled_data[-window_size:]
    for _ in range(future_days):
        in_win = jnp.reshape(current_window, (1, window_size, 1)); next_pred = model.apply({'params': params}, in_win)
        future_predictions.append(next_pred[0, 0]); current_window = jnp.append(current_window[1:], next_pred, axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    last_date = daily['EVENT_DATETIME'].iloc[-1]; future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    historical = daily[['EVENT_DATETIME', 'TOTAL']].copy(); historical.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL': 'Minutes'}, inplace=True); historical['Data_Type'] = 'Actual'
    future = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()}); future['Data_Type'] = 'Forecast'
    master = pd.concat([historical, future], ignore_index=True); master['Stress_Coefficient'] = (master['Minutes'] / historical['Minutes'].max()).clip(upper=1.0)
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM jax_predictions;")
            for _, row in master.iterrows():
                cur.execute("INSERT INTO jax_predictions (forecast_date, predicted_minutes, stress_coefficient, data_type, rmse) VALUES (%s,%s,%s,%s,%s)",
                    (row['Date'], float(row['Minutes']), float(row['Stress_Coefficient']), row['Data_Type'], float(rmse)))
        conn.commit(); conn.close(); print("✅ Predictions saved!")
    except Exception as e: print(f"⚠️ Could not save predictions: {e}")

def scrape_and_store():
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}; rows_saved = 0
    for station in MANCHESTER_LINE:
        current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{station}/{current_time}?numRows=15&timeWindow=120"
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                data = response.json(); services = data.get('trainServices', []); station_delay = 0
                for train in services:
                    std = train.get('std', 'N/A'); atd = train.get('atd', 'N/A'); is_cancelled = train.get('isCancelled', False)
                    flag = atd if atd != 'N/A' else train.get('etd', 'N/A'); flag_lower = str(flag).lower().strip()
                    if is_cancelled or flag_lower == 'cancelled': station_delay += 60
                    elif flag_lower not in ['on time', 'n/a', 'no report', 'delayed'] and std != 'N/A':
                        try:
                            dt_std = datetime.fromisoformat(str(std)); dt_flag = datetime.fromisoformat(str(flag))
                            delay = (dt_flag - dt_std).total_seconds() / 60
                            if delay < -720: delay += 1440
                            elif delay > 720: delay -= 1440
                            if delay > 0: station_delay += delay
                        except: pass
                try:
                    conn = get_db()
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO rail_events (event_datetime, pfpi_minutes, non_pfpi_minutes, source_file) VALUES (%s,%s,%s,%s)",
                            (datetime.now(), float(station_delay), 0.0, f"scraper_{station}"))
                    conn.commit(); conn.close(); rows_saved += 1
                except Exception as e: print(f"⚠️ Scraper DB error for {station}: {e}")
        except: pass
        time.sleep(0.5)
    if rows_saved > 0: print(f"📥 Scraper: Saved {rows_saved} station readings.")
    return rows_saved

def background_tasks():
    time.sleep(5)
    print("📡 Scanning live departures..."); scan_live_departures()
    print("📥 Running initial data scrape..."); scrape_and_store(); run_jax_training()
    last_live = time.time(); last_train = time.time()
    while True:
        time.sleep(60)
        try:
            scrape_and_store()
            if time.time() - last_live > 300: print("📡 Refreshing live departures..."); scan_live_departures(); last_live = time.time()
            if time.time() - last_train > 21600: print("🔄 Retraining JAX model..."); run_jax_training(); last_train = time.time()
        except Exception as e: print(f"⚠️ Background task error: {e}")

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/departures/<crs>')
def departures(crs):
    """Return only Manchester-Euston corridor trains using filterCrs."""
    crs = crs.upper()
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    all_services = []
    seen_ids = set()

    # Southbound: trains heading towards London Euston
    if crs != 'EUS':
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{crs}/{current_time}?numRows=10&timeWindow=120&filterCrs=EUS&filterType=to"
        try:
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            if r.status_code == 200:
                for t in r.json().get('trainServices', []) or []:
                    tid = t.get('rid', t.get('trainid', ''))
                    if tid not in seen_ids and not t.get('atdSpecified') and not t.get('atd'):
                        seen_ids.add(tid)
                        all_services.append(t)
        except: pass

    # Northbound: trains heading towards Manchester
    if crs != 'MAN':
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{crs}/{current_time}?numRows=10&timeWindow=120&filterCrs=MAN&filterType=to"
        try:
            r = requests.get(url, headers=headers, timeout=15, verify=False)
            if r.status_code == 200:
                for t in r.json().get('trainServices', []) or []:
                    tid = t.get('rid', t.get('trainid', ''))
                    if tid not in seen_ids and not t.get('atdSpecified') and not t.get('atd'):
                        seen_ids.add(tid)
                        all_services.append(t)
        except: pass

    # Sort by scheduled departure time
    all_services.sort(key=lambda t: t.get('std', ''))

    return jsonify({
        'trainServices': all_services,
        'locationName': ROUTE_STATIONS.get(crs, {}).get('name', crs),
        'crs': crs
    })

@app.route('/api/stations')
def stations():
    return jsonify({'status': 'ok', 'stations': [{'crs': s['crs'], 'name': s['name'], 'lat': s['lat'], 'lon': s['lon']} for s in ROUTE_STATIONS.values()]})

@app.route('/api/debug/journey')
def debug_journey():
    from_crs = request.args.get('from', 'MAN').upper(); to_crs = request.args.get('to', 'EUS').upper()
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{from_crs}/{current_time}?numRows=2&timeWindow=120&filterCrs={to_crs}&filterType=to"
    try: response = requests.get(url, headers=headers, timeout=15, verify=False); return jsonify(response.json())
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/journey')
def journey():
    from_crs = request.args.get('from', '').upper(); to_crs = request.args.get('to', '').upper()
    date = request.args.get('date', ''); time_val = request.args.get('time', '')
    if from_crs not in ROUTE_STATIONS or to_crs not in ROUTE_STATIONS:
        return jsonify({'status': 'error', 'message': 'Both stations must be on the Manchester-Euston route.'}), 400
    if from_crs == to_crs:
        return jsonify({'status': 'error', 'message': 'Origin and destination must be different.'}), 400
    from_station = ROUTE_STATIONS[from_crs]; to_station = ROUTE_STATIONS[to_crs]
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

    # Check if date is too far ahead for live data
    is_future = False
    if date:
        try:
            requested = datetime.strptime(date, '%Y-%m-%d').date()
            tomorrow = (datetime.now() + timedelta(days=1)).date()
            if requested > tomorrow: is_future = True
        except: pass

    # Helper to get JAX prediction
    def get_jax(d):
        try:
            conn = get_db(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT predicted_minutes, stress_coefficient, rmse FROM jax_predictions WHERE forecast_date = %s", (d,))
            row = cur.fetchone(); conn.close()
            if row:
                stress = float(row['stress_coefficient'])
                if stress < 0.2: confidence, color = 'HIGH', '#22c55e'
                elif stress < 0.4: confidence, color = 'GOOD', '#84cc16'
                elif stress < 0.6: confidence, color = 'MODERATE', '#eab308'
                elif stress < 0.8: confidence, color = 'LOW', '#f97316'
                else: confidence, color = 'POOR', '#ef4444'
                return {'predicted_minutes': round(float(row['predicted_minutes']), 1), 'stress_coefficient': round(stress, 4),
                        'rmse': round(float(row['rmse']), 2), 'confidence': confidence, 'color': color}
        except: pass
        return None

    if is_future:
        return jsonify({'status': 'ok', 'from': from_station, 'to': to_station, 'date': date, 'time': time_val,
            'journeys': [], 'jax_prediction': get_jax(date),
            'future_notice': 'Live train times are only available for today and tomorrow. The AI delay prediction for this date is shown above.',
            'source': 'National Rail LDBWS'})

    if date and time_val: query_time = f"{date.replace('-', '')}T{time_val.replace(':', '')}00"
    else: query_time = datetime.now().strftime("%Y%m%dT%H%M%S")

    url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{from_crs}/{query_time}?numRows=10&timeWindow=120&filterCrs={to_crs}&filterType=to"
    try:
        print(f"📍 Journey: {from_crs} → {to_crs}")
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code != 200:
            return jsonify({'status': 'error', 'message': f'API returned {response.status_code}'}), 500
        data = response.json(); services = data.get('trainServices', [])
        jax_prediction = get_jax(date) if date else None

        is_custom_time = bool(date and time_val)
        journeys = []
        for train in (services or []):
            if not is_custom_time and (train.get('atdSpecified', False) or train.get('atd')): continue
            std_raw = train.get('std', ''); dep_display = _iso_to_hhmm(std_raw)
            etd_raw = train.get('etd', ''); platform = train.get('platform', '-')
            operator = train.get('operator', train.get('operatorCode', ''))
            is_cancelled = train.get('isCancelled', False)
            dest_name = 'Unknown'
            dest_list = train.get('destination', [])
            if isinstance(dest_list, list) and dest_list: dest_name = dest_list[0].get('locationName', 'Unknown')
            elif isinstance(dest_list, dict): dest_name = dest_list.get('locationName', 'Unknown')
            arrival_time = ''
            for loc in train.get('subsequentLocations', []):
                if isinstance(loc, dict) and loc.get('crs', '').upper() == to_crs:
                    arr_raw = loc.get('eta', '') or loc.get('sta', '') or loc.get('ata', '')
                    arrival_time = _iso_to_hhmm(arr_raw); break
            duration = ''
            if dep_display and arrival_time:
                try:
                    dp = dep_display.split(':'); ap = arrival_time.split(':')
                    dm = int(dp[0])*60+int(dp[1]); am = int(ap[0])*60+int(ap[1])
                    if am < dm: am += 1440
                    diff = am - dm; h = diff // 60; m = diff % 60
                    duration = f"{h}h {m:02d}m" if h > 0 else f"{m}m"
                except: pass
            status = 'ON TIME'; is_delayed = False
            if is_cancelled: status = 'CANCELLED'; is_delayed = True
            elif str(etd_raw).lower().strip() == 'delayed': status = 'LATE'; is_delayed = True
            elif train.get('atd'):
                atd_hhmm = _iso_to_hhmm(train.get('atd', ''))
                if atd_hhmm and dep_display:
                    try:
                        d_m = int(dep_display.split(':')[0])*60+int(dep_display.split(':')[1])
                        a_m = int(atd_hhmm.split(':')[0])*60+int(atd_hhmm.split(':')[1])
                        if a_m - d_m > 1: status = 'LATE'; is_delayed = True
                    except: pass
            journeys.append({'departure': dep_display, 'arrival': arrival_time, 'duration': duration,
                'destination': dest_name, 'operator': operator, 'platform': platform,
                'status': status, 'train_uid': train.get('trainid', ''), 'is_delayed': is_delayed,
                'legs': [{'mode': 'train', 'from': from_station['name'], 'to': to_station['name'],
                    'destination': dest_name, 'departure': dep_display, 'arrival': arrival_time,
                    'duration': duration, 'operator': operator, 'platform': platform}]})
        return jsonify({'status': 'ok', 'from': from_station, 'to': to_station, 'date': date, 'time': time_val,
            'journeys': journeys, 'jax_prediction': jax_prediction, 'source': 'National Rail LDBWS'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predictions')
def predictions():
    try:
        conn = get_db(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT forecast_date, predicted_minutes, stress_coefficient, data_type, rmse, updated_at FROM jax_predictions ORDER BY forecast_date")
        rows = cur.fetchall(); conn.close()
        return jsonify({'status': 'ok', 'predictions': [{'forecast_date': str(r['forecast_date']),
            'predicted_minutes': round(float(r['predicted_minutes']), 2), 'stress_coefficient': round(float(r['stress_coefficient']), 4),
            'data_type': r['data_type'], 'rmse': round(float(r['rmse']), 2),
            'updated_at': str(r['updated_at']) if r['updated_at'] else None} for r in rows]})
    except Exception as e: return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/live')
def live():
    try:
        conn = get_db(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT station, delay_minutes, status, total_delayed_trains, total_delay_minutes, updated_at FROM live_snapshot ORDER BY id")
        rows = cur.fetchall(); conn.close()
        return jsonify({'status': 'ok', 'stations': [{'station': r['station'], 'delay': float(r['delay_minutes']), 'status': r['status']} for r in rows],
            'total_delayed_trains': int(rows[0]['total_delayed_trains']) if rows else 0,
            'total_delay_minutes': float(rows[0]['total_delay_minutes']) if rows else 0,
            'updated': str(rows[0]['updated_at']) if rows else None})
    except Exception as e: return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plan/<date>')
def plan(date):
    try:
        conn = get_db(); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT forecast_date, predicted_minutes, stress_coefficient, data_type, rmse FROM jax_predictions WHERE forecast_date = %s", (date,))
        row = cur.fetchone(); conn.close()
        if not row: return jsonify({'status': 'error', 'message': 'No prediction for this date.'}), 404
        stress = float(row['stress_coefficient']); minutes = float(row['predicted_minutes'])
        if stress < 0.2: confidence, color, rec = 'HIGH', '#22c55e', 'Excellent day to travel. Minimal delays expected.'
        elif stress < 0.4: confidence, color, rec = 'GOOD', '#84cc16', 'Good conditions. Minor delays possible.'
        elif stress < 0.6: confidence, color, rec = 'MODERATE', '#eab308', 'Some delays expected. Allow 15-20 extra minutes.'
        elif stress < 0.8: confidence, color, rec = 'LOW', '#f97316', 'Significant delays likely. Consider travelling earlier or later.'
        else: confidence, color, rec = 'POOR', '#ef4444', 'Severe disruption expected. Consider alternative transport.'
        return jsonify({'status': 'ok', 'date': str(row['forecast_date']), 'predicted_minutes': round(minutes, 1),
            'stress_coefficient': round(stress, 4), 'data_type': row['data_type'], 'rmse': round(float(row['rmse']), 2),
            'confidence': confidence, 'recommendation': rec, 'color': color})
    except Exception as e: return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

@app.route('/api/stats')
def stats():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rail_events"); total_events = cur.fetchone()[0]
        cur.execute("SELECT MIN(event_datetime), MAX(event_datetime) FROM rail_events"); date_range = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM rail_events WHERE source_file LIKE 'scraper_%'"); scraped_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jax_predictions WHERE data_type = 'Forecast'"); forecast_days = cur.fetchone()[0]
        conn.close()
        return jsonify({'status': 'ok', 'total_events': total_events, 'scraped_events': scraped_rows,
            'date_from': str(date_range[0]) if date_range[0] else None, 'date_to': str(date_range[1]) if date_range[1] else None,
            'forecast_days': forecast_days})
    except Exception as e: return jsonify({'status': 'error', 'message': str(e)}), 500

init_db()
t = threading.Thread(target=background_tasks, daemon=True); t.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)