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
from datetime import datetime
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

# --- CONFIG ---
API_KEY = os.environ.get("RAIL_API_KEY", "")
DB_URL = os.environ.get("DATABASE_URL")
REST_BASE_URL = "https://api1.raildata.org.uk/1010-live-departure-board---staff-version1_0/LDBSVWS/api/20220120"
MANCHESTER_LINE = ['MAN', 'SPT', 'MAC', 'SOT', 'CRE', 'RUG', 'MKC', 'EUS']

# Station info for the Manchester-Euston route
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

# --- JAX MODEL ---
class JaxRouteModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# --- DATABASE HELPERS ---
def get_db():
    return psycopg2.connect(DB_URL)

def init_db():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rail_events (
                    id SERIAL PRIMARY KEY,
                    event_datetime TIMESTAMP,
                    pfpi_minutes DOUBLE PRECISION,
                    non_pfpi_minutes DOUBLE PRECISION,
                    source_file TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_event_datetime ON rail_events(event_datetime);
                CREATE TABLE IF NOT EXISTS jax_predictions (
                    id SERIAL PRIMARY KEY,
                    forecast_date DATE,
                    predicted_minutes DOUBLE PRECISION,
                    stress_coefficient DOUBLE PRECISION,
                    data_type TEXT,
                    rmse DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS live_snapshot (
                    id SERIAL PRIMARY KEY,
                    station TEXT,
                    delay_minutes DOUBLE PRECISION,
                    status TEXT,
                    total_delayed_trains INTEGER,
                    total_delay_minutes DOUBLE PRECISION,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
        conn.commit()
        conn.close()
        print("✅ Database tables ready.")
    except Exception as e:
        print(f"⚠️ Database init error: {e}")

# --- HELPER: extract HH:MM from ISO datetime string ---
def _iso_to_hhmm(iso_str):
    """Convert '2026-04-24T12:27:00' to '12:27'. Returns '' if invalid."""
    if not iso_str or not isinstance(iso_str, str):
        return ''
    try:
        if 'T' in iso_str:
            time_part = iso_str.split('T')[1]
            return time_part[:5]
        elif len(iso_str) >= 5 and ':' in iso_str:
            return iso_str[:5]
    except:
        pass
    return ''

# --- LIVE API SCANNER ---
def scan_live_departures():
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    results = []
    total_delay = 0
    total_delayed = 0

    for station in MANCHESTER_LINE:
        current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{station}/{current_time}?numRows=10&timeWindow=120"

        station_delay = 0
        station_status = "⚠️"

        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                data = response.json()
                services = data.get('trainServices', [])

                if not services:
                    results.append({'station': station, 'delay': 0, 'status': '⚪ None'})
                    continue

                for train in services[:6]:
                    std = train.get('std', 'N/A')
                    etd = train.get('etd', 'N/A')
                    atd = train.get('atd', 'N/A')
                    is_cancelled = train.get('isCancelled', False)
                    flag = atd if atd != 'N/A' else etd
                    flag_lower = str(flag).lower().strip()

                    if is_cancelled or flag_lower == 'cancelled':
                        station_delay += 60
                        total_delayed += 1
                    elif flag_lower == 'delayed':
                        total_delayed += 1
                    elif flag_lower not in ['on time', 'n/a', 'no report'] and std != 'N/A':
                        try:
                            dt_std = datetime.fromisoformat(str(std))
                            dt_flag = datetime.fromisoformat(str(flag))
                            delay = (dt_flag - dt_std).total_seconds() / 60
                            if delay < -720: delay += 1440
                            elif delay > 720: delay -= 1440
                            if delay > 0:
                                station_delay += delay
                                total_delayed += 1
                        except:
                            try:
                                dt_std = datetime.strptime(str(std).strip()[:5], '%H:%M')
                                dt_flag = datetime.strptime(str(flag).strip()[:5], '%H:%M')
                                delay = (dt_flag - dt_std).total_seconds() / 60
                                if delay < -720: delay += 1440
                                elif delay > 720: delay -= 1440
                                if delay > 0:
                                    station_delay += delay
                                    total_delayed += 1
                            except:
                                pass

                total_delay += station_delay
                if station_delay > 0:
                    station_status = f"🔴 {int(station_delay)}m"
                else:
                    station_status = "🟢 OK"
            else:
                station_status = f"⚠️ {response.status_code}"
        except Exception as e:
            station_status = "⚠️"

        results.append({'station': station, 'delay': station_delay, 'status': station_status})
        time.sleep(0.5)

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM live_snapshot;")
            for r in results:
                cur.execute(
                    "INSERT INTO live_snapshot (station, delay_minutes, status, total_delayed_trains, total_delay_minutes) VALUES (%s,%s,%s,%s,%s)",
                    (r['station'], float(r['delay']), r['status'], int(total_delayed), float(total_delay))
                )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Could not save live snapshot: {e}")

    return results, total_delayed, total_delay

# --- JAX TRAINING ---
def run_jax_training():
    print("\n🧠 --- STARTING JAX TRAINING ---")

    try:
        conn = get_db()
        df = pd.read_sql("""
            SELECT event_datetime as "EVENT_DATETIME",
                   pfpi_minutes as "PFPI_MINUTES",
                   non_pfpi_minutes as "NON_PFPI_MINUTES"
            FROM rail_events ORDER BY event_datetime
        """, conn)
        conn.close()
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    if df.empty:
        print("❌ No data in database.")
        return

    print(f"✅ Loaded {len(df)} rows.")

    df['EVENT_DATETIME'] = pd.to_datetime(df['EVENT_DATETIME'], errors='coerce')
    df = df.dropna(subset=['EVENT_DATETIME']).sort_values('EVENT_DATETIME')

    if 'NON_PFPI_MINUTES' not in df.columns:
        df['NON_PFPI_MINUTES'] = 0
    if 'PFPI_MINUTES' not in df.columns:
        df['PFPI_MINUTES'] = 0

    daily = df.groupby(df['EVENT_DATETIME'].dt.date)[['PFPI_MINUTES', 'NON_PFPI_MINUTES']].sum().reset_index()
    daily['TOTAL_COMBINED_MINUTES'] = daily['PFPI_MINUTES'] + daily['NON_PFPI_MINUTES']
    daily['EVENT_DATETIME'] = pd.to_datetime(daily['EVENT_DATETIME'])

    total_days = len(daily)
    if total_days < 15:
        print(f"❌ Not enough data ({total_days} days).")
        return

    dataset = daily[['TOTAL_COMBINED_MINUTES']].values
    training_data_len = math.ceil(total_days * 0.8)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    window_size = 14
    train_data = scaled_data[:training_data_len]

    x_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i - window_size:i, 0])
        y_train.append(train_data[i, 0])

    x_train_jnp = jnp.array(x_train).reshape(-1, window_size, 1)
    y_train_jnp = jnp.array(y_train).reshape(-1, 1)

    model = JaxRouteModel()
    key = random.PRNGKey(42)
    variables = model.init(key, jnp.ones((1, window_size, 1)))
    params = variables['params']
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            preds = model.apply({'params': p}, x)
            return jnp.mean((preds - y) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for epoch in range(51):
        params, opt_state, loss = train_step(params, opt_state, x_train_jnp, y_train_jnp)
        if epoch % 10 == 0:
            loss.block_until_ready()
            print(f"  Epoch {epoch:02d} | Loss: {loss:.6f}")

    test_data = scaled_data[training_data_len - window_size:]
    x_test = [test_data[i - window_size:i, 0] for i in range(window_size, len(test_data))]
    x_test_jnp = jnp.array(x_test).reshape(-1, window_size, 1)
    y_test = dataset[training_data_len:]
    preds_scaled = model.apply({'params': params}, x_test_jnp)
    test_predictions = scaler.inverse_transform(np.array(preds_scaled))
    rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
    print(f"  RMSE: {rmse:.2f}")

    future_days = 30
    future_predictions = []
    current_window = scaled_data[-window_size:]
    for _ in range(future_days):
        in_win = jnp.reshape(current_window, (1, window_size, 1))
        next_pred = model.apply({'params': params}, in_win)
        future_predictions.append(next_pred[0, 0])
        current_window = jnp.append(current_window[1:], next_pred, axis=0)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = daily['EVENT_DATETIME'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    historical = daily[['EVENT_DATETIME', 'TOTAL_COMBINED_MINUTES']].copy()
    historical.rename(columns={'EVENT_DATETIME': 'Date', 'TOTAL_COMBINED_MINUTES': 'Minutes'}, inplace=True)
    historical['Data_Type'] = 'Actual'

    future = pd.DataFrame({'Date': future_dates, 'Minutes': future_predictions.flatten()})
    future['Data_Type'] = 'Forecast'

    master = pd.concat([historical, future], ignore_index=True)
    master['Stress_Coefficient'] = (master['Minutes'] / historical['Minutes'].max()).clip(upper=1.0)

    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM jax_predictions;")
            for _, row in master.iterrows():
                cur.execute(
                    "INSERT INTO jax_predictions (forecast_date, predicted_minutes, stress_coefficient, data_type, rmse) VALUES (%s,%s,%s,%s,%s)",
                    (row['Date'], float(row['Minutes']), float(row['Stress_Coefficient']), row['Data_Type'], float(rmse))
                )
        conn.commit()
        conn.close()
        print("✅ Predictions saved to database!")
    except Exception as e:
        print(f"⚠️ Could not save predictions: {e}")

# --- DATA SCRAPER ---
def scrape_and_store():
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    rows_saved = 0

    for station in MANCHESTER_LINE:
        current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
        url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{station}/{current_time}?numRows=15&timeWindow=120"

        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                data = response.json()
                services = data.get('trainServices', [])
                station_delay = 0

                for train in services:
                    std = train.get('std', 'N/A')
                    etd = train.get('etd', 'N/A')
                    atd = train.get('atd', 'N/A')
                    is_cancelled = train.get('isCancelled', False)
                    flag = atd if atd != 'N/A' else etd
                    flag_lower = str(flag).lower().strip()

                    if is_cancelled or flag_lower == 'cancelled':
                        station_delay += 60
                    elif flag_lower not in ['on time', 'n/a', 'no report', 'delayed'] and std != 'N/A':
                        try:
                            dt_std = datetime.fromisoformat(str(std))
                            dt_flag = datetime.fromisoformat(str(flag))
                            delay = (dt_flag - dt_std).total_seconds() / 60
                            if delay < -720: delay += 1440
                            elif delay > 720: delay -= 1440
                            if delay > 0:
                                station_delay += delay
                        except:
                            pass

                try:
                    conn = get_db()
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO rail_events (event_datetime, pfpi_minutes, non_pfpi_minutes, source_file) VALUES (%s, %s, %s, %s)",
                            (datetime.now(), float(station_delay), 0.0, f"scraper_{station}")
                        )
                    conn.commit()
                    conn.close()
                    rows_saved += 1
                except Exception as e:
                    print(f"⚠️ Scraper DB error for {station}: {e}")
        except:
            pass

        time.sleep(0.5)

    if rows_saved > 0:
        print(f"📥 Scraper: Saved {rows_saved} station readings to database.")
    return rows_saved

# --- BACKGROUND TASKS ---
def background_tasks():
    time.sleep(5)
    print("📡 Scanning live departures...")
    scan_live_departures()
    print("📥 Running initial data scrape...")
    scrape_and_store()
    run_jax_training()

    last_live = time.time()
    last_train = time.time()

    while True:
        time.sleep(60)
        try:
            scrape_and_store()
            if time.time() - last_live > 300:
                print("📡 Refreshing live departures...")
                scan_live_departures()
                last_live = time.time()
            if time.time() - last_train > 21600:
                print("🔄 Retraining JAX model with new data...")
                run_jax_training()
                last_train = time.time()
        except Exception as e:
            print(f"⚠️ Background task error: {e}")

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/departures/<crs>')
def departures(crs):
    """Return live departures filtered to trains that travel along the corridor."""
    crs = crs.upper()
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{crs}/{current_time}?numRows=20&timeWindow=60"
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    corridor_crs = set(ROUTE_STATIONS.keys()) - {crs}  # other corridor stations (not the origin)
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        data = response.json()
        if 'trainServices' in data and data['trainServices']:
            filtered = []
            for t in data['trainServices']:
                # Skip trains that have already departed
                if t.get('atdSpecified', False) or t.get('atd'):
                    continue
                # Check if destination is a corridor station
                dest = t.get('destination', [])
                dest_codes = []
                if isinstance(dest, list):
                    dest_codes = [d.get('crs', '').upper() for d in dest if isinstance(d, dict)]
                elif isinstance(dest, dict):
                    dest_codes = [dest.get('crs', '').upper()]
                if any(dc in corridor_crs for dc in dest_codes):
                    filtered.append(t)
                    continue
                # Check if the train calls at any corridor station along the way
                for loc in t.get('subsequentLocations', []):
                    if isinstance(loc, dict) and loc.get('crs', '').upper() in corridor_crs and not loc.get('isPass', False):
                        filtered.append(t)
                        break
            data['trainServices'] = filtered
        return jsonify(data), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stations')
def stations():
    return jsonify({'status': 'ok', 'stations': [
        {'crs': s['crs'], 'name': s['name'], 'lat': s['lat'], 'lon': s['lon']}
        for s in ROUTE_STATIONS.values()
    ]})

@app.route('/api/debug/journey')
def debug_journey():
    from_crs = request.args.get('from', 'MAN').upper()
    to_crs = request.args.get('to', 'EUS').upper()
    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
    current_time = datetime.now().strftime("%Y%m%dT%H%M%S")
    url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{from_crs}/{current_time}?numRows=2&timeWindow=120&filterCrs={to_crs}&filterType=to"
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- JOURNEY PLANNER (National Rail LDBSVWS with filterCrs) ---
@app.route('/api/journey')
def journey():
    """Plan a journey using the National Rail LDBSVWS Staff API with filterCrs.
    The Staff API returns subsequentLocations (not subsequentCallingPoints)
    with ISO datetime strings like '2026-04-24T12:27:00'."""
    from_crs = request.args.get('from', '').upper()
    to_crs = request.args.get('to', '').upper()
    date = request.args.get('date', '')
    time_val = request.args.get('time', '')

    if from_crs not in ROUTE_STATIONS or to_crs not in ROUTE_STATIONS:
        return jsonify({'status': 'error', 'message': 'Both stations must be on the Manchester-Euston route.'}), 400

    if from_crs == to_crs:
        return jsonify({'status': 'error', 'message': 'Origin and destination must be different.'}), 400

    from_station = ROUTE_STATIONS[from_crs]
    to_station = ROUTE_STATIONS[to_crs]

    headers = {'x-apikey': API_KEY, 'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}

    # Check if the requested date is beyond what the live API can provide (~1 day ahead)
    is_future = False
    if date:
        try:
            from datetime import timedelta
            requested = datetime.strptime(date, '%Y-%m-%d').date()
            tomorrow = (datetime.now() + timedelta(days=1)).date()
            if requested > tomorrow:
                is_future = True
        except:
            pass

    # For dates too far ahead, return AI prediction only (no live train data available)
    if is_future:
        jax_prediction = None
        try:
            conn = get_db()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT predicted_minutes, stress_coefficient, rmse FROM jax_predictions WHERE forecast_date = %s", (date,))
            row = cur.fetchone()
            conn.close()
            if row:
                stress = float(row['stress_coefficient'])
                if stress < 0.2:
                    confidence, color = 'HIGH', '#22c55e'
                elif stress < 0.4:
                    confidence, color = 'GOOD', '#84cc16'
                elif stress < 0.6:
                    confidence, color = 'MODERATE', '#eab308'
                elif stress < 0.8:
                    confidence, color = 'LOW', '#f97316'
                else:
                    confidence, color = 'POOR', '#ef4444'
                jax_prediction = {
                    'predicted_minutes': round(float(row['predicted_minutes']), 1),
                    'stress_coefficient': round(stress, 4),
                    'rmse': round(float(row['rmse']), 2),
                    'confidence': confidence,
                    'color': color
                }
        except:
            pass

        return jsonify({
            'status': 'ok',
            'from': from_station,
            'to': to_station,
            'date': date,
            'time': time_val,
            'journeys': [],
            'jax_prediction': jax_prediction,
            'future_notice': 'Live train times are only available for today and tomorrow. The AI delay prediction for this date is shown above.',
            'source': 'National Rail LDBWS'
        })

    # Use the user's selected date/time if provided, otherwise use now
    if date and time_val:
        query_time = f"{date.replace('-', '')}T{time_val.replace(':', '')}00"
    else:
        query_time = datetime.now().strftime("%Y%m%dT%H%M%S")

    url = f"{REST_BASE_URL}/GetDepBoardWithDetails/{from_crs}/{query_time}?numRows=10&timeWindow=120&filterCrs={to_crs}&filterType=to"

    try:
        print(f"📍 Journey: {from_crs} → {to_crs} via LDBSVWS filterCrs")
        response = requests.get(url, headers=headers, timeout=15, verify=False)

        if response.status_code != 200:
            return jsonify({'status': 'error', 'message': f'National Rail API returned {response.status_code}'}), 500

        data = response.json()
        services = data.get('trainServices', [])

        # Get JAX prediction for the travel date
        jax_prediction = None
        if date:
            try:
                conn = get_db()
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT predicted_minutes, stress_coefficient, rmse FROM jax_predictions WHERE forecast_date = %s", (date,))
                row = cur.fetchone()
                conn.close()
                if row:
                    stress = float(row['stress_coefficient'])
                    if stress < 0.2:
                        confidence, color = 'HIGH', '#22c55e'
                    elif stress < 0.4:
                        confidence, color = 'GOOD', '#84cc16'
                    elif stress < 0.6:
                        confidence, color = 'MODERATE', '#eab308'
                    elif stress < 0.8:
                        confidence, color = 'LOW', '#f97316'
                    else:
                        confidence, color = 'POOR', '#ef4444'
                    jax_prediction = {
                        'predicted_minutes': round(float(row['predicted_minutes']), 1),
                        'stress_coefficient': round(stress, 4),
                        'rmse': round(float(row['rmse']), 2),
                        'confidence': confidence,
                        'color': color
                    }
            except:
                pass

        # Parse services into journey cards
        is_custom_time = bool(date and time_val)
        journeys = []
        for train in (services or []):
            # Skip already-departed trains only when querying current time
            if not is_custom_time and (train.get('atdSpecified', False) or train.get('atd')):
                continue

            # Departure time: std is ISO like "2026-04-24T10:14:00"
            std_raw = train.get('std', '')
            dep_display = _iso_to_hhmm(std_raw)

            # ETD for status
            etd_raw = train.get('etd', '')
            etd_display = _iso_to_hhmm(etd_raw) if etd_raw else ''

            platform = train.get('platform', '-')
            operator = train.get('operator', train.get('operatorCode', ''))
            is_cancelled = train.get('isCancelled', False)

            # Get destination name
            dest_name = 'Unknown'
            dest_list = train.get('destination', [])
            if isinstance(dest_list, list) and dest_list:
                dest_name = dest_list[0].get('locationName', 'Unknown')
            elif isinstance(dest_list, dict):
                dest_name = dest_list.get('locationName', 'Unknown')

            # Find arrival time at destination from subsequentLocations
            arrival_time = ''
            subsequent = train.get('subsequentLocations', [])
            if isinstance(subsequent, list):
                for loc in subsequent:
                    if isinstance(loc, dict) and loc.get('crs', '').upper() == to_crs:
                        # Use sta (scheduled arrival), eta (estimated), or ata (actual)
                        arr_raw = loc.get('eta', '') or loc.get('sta', '') or loc.get('ata', '')
                        arrival_time = _iso_to_hhmm(arr_raw)
                        break

            # Calculate duration
            duration = ''
            if dep_display and arrival_time:
                try:
                    dep_parts = dep_display.split(':')
                    arr_parts = arrival_time.split(':')
                    if len(dep_parts) == 2 and len(arr_parts) == 2:
                        dep_mins = int(dep_parts[0]) * 60 + int(dep_parts[1])
                        arr_mins = int(arr_parts[0]) * 60 + int(arr_parts[1])
                        if arr_mins < dep_mins:
                            arr_mins += 24 * 60
                        diff = arr_mins - dep_mins
                        hours = diff // 60
                        mins = diff % 60
                        if hours > 0:
                            duration = f"{hours}h {mins:02d}m"
                        else:
                            duration = f"{mins}m"
                except:
                    pass

            # Determine status
            status = 'ON TIME'
            is_delayed = False
            if is_cancelled:
                status = 'CANCELLED'
                is_delayed = True
            elif etd_raw:
                # If atd exists, train has already departed
                atd_raw = train.get('atd', '')
                if atd_raw:
                    # Compare actual vs scheduled departure
                    atd_hhmm = _iso_to_hhmm(atd_raw)
                    if atd_hhmm and dep_display:
                        try:
                            d_mins = int(dep_display.split(':')[0]) * 60 + int(dep_display.split(':')[1])
                            a_mins = int(atd_hhmm.split(':')[0]) * 60 + int(atd_hhmm.split(':')[1])
                            diff = a_mins - d_mins
                            if diff > 1:
                                status = 'LATE'
                                is_delayed = True
                            else:
                                status = 'ON TIME'
                        except:
                            pass
                elif etd_display and etd_display != dep_display:
                    status = 'LATE'
                    is_delayed = True

            journeys.append({
                'departure': dep_display,
                'expected_departure': etd_display or dep_display,
                'arrival': arrival_time,
                'duration': duration,
                'destination': dest_name,
                'operator': operator,
                'platform': platform,
                'status': status,
                'train_uid': train.get('trainid', ''),
                'is_delayed': is_delayed,
                'legs': [{
                    'mode': 'train',
                    'from': from_station['name'],
                    'to': to_station['name'],
                    'destination': dest_name,
                    'departure': dep_display,
                    'arrival': arrival_time,
                    'duration': duration,
                    'operator': operator,
                    'platform': platform,
                }]
            })

        return jsonify({
            'status': 'ok',
            'from': from_station,
            'to': to_station,
            'date': date,
            'time': time_val,
            'journeys': journeys,
            'jax_prediction': jax_prediction,
            'source': 'National Rail LDBWS'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predictions')
def predictions():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT forecast_date, predicted_minutes, stress_coefficient, data_type, rmse, updated_at FROM jax_predictions ORDER BY forecast_date")
        rows = cur.fetchall()
        conn.close()
        return jsonify({'status': 'ok', 'predictions': [{
            'forecast_date': str(r['forecast_date']),
            'predicted_minutes': round(float(r['predicted_minutes']), 2),
            'stress_coefficient': round(float(r['stress_coefficient']), 4),
            'data_type': r['data_type'],
            'rmse': round(float(r['rmse']), 2),
            'updated_at': str(r['updated_at']) if r['updated_at'] else None
        } for r in rows]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/live')
def live():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT station, delay_minutes, status, total_delayed_trains, total_delay_minutes, updated_at FROM live_snapshot ORDER BY id")
        rows = cur.fetchall()
        conn.close()

        total_delayed = int(rows[0]['total_delayed_trains']) if rows else 0
        total_delay = float(rows[0]['total_delay_minutes']) if rows else 0
        updated = str(rows[0]['updated_at']) if rows else None

        return jsonify({
            'status': 'ok',
            'stations': [{'station': r['station'], 'delay': float(r['delay_minutes']), 'status': r['status']} for r in rows],
            'total_delayed_trains': total_delayed,
            'total_delay_minutes': total_delay,
            'updated': updated
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plan/<date>')
def plan(date):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT forecast_date, predicted_minutes, stress_coefficient, data_type, rmse FROM jax_predictions WHERE forecast_date = %s", (date,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify({'status': 'error', 'message': 'No prediction for this date. The model forecasts 30 days ahead.'}), 404

        stress = float(row['stress_coefficient'])
        minutes = float(row['predicted_minutes'])

        if stress < 0.2:
            confidence, color = 'HIGH', '#22c55e'
            rec = 'Excellent day to travel. Minimal delays expected.'
        elif stress < 0.4:
            confidence, color = 'GOOD', '#84cc16'
            rec = 'Good conditions. Minor delays possible.'
        elif stress < 0.6:
            confidence, color = 'MODERATE', '#eab308'
            rec = 'Some delays expected. Allow 15-20 extra minutes.'
        elif stress < 0.8:
            confidence, color = 'LOW', '#f97316'
            rec = 'Significant delays likely. Consider travelling earlier or later.'
        else:
            confidence, color = 'POOR', '#ef4444'
            rec = 'Severe disruption expected. Consider alternative transport.'

        return jsonify({
            'status': 'ok',
            'date': str(row['forecast_date']),
            'predicted_minutes': round(minutes, 1),
            'stress_coefficient': round(stress, 4),
            'data_type': row['data_type'],
            'rmse': round(float(row['rmse']), 2),
            'confidence': confidence,
            'recommendation': rec,
            'color': color
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

@app.route('/api/stats')
def stats():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rail_events")
        total_events = cur.fetchone()[0]
        cur.execute("SELECT MIN(event_datetime), MAX(event_datetime) FROM rail_events")
        date_range = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM rail_events WHERE source_file LIKE 'scraper_%'")
        scraped_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jax_predictions WHERE data_type = 'Forecast'")
        forecast_days = cur.fetchone()[0]
        conn.close()
        return jsonify({
            'status': 'ok',
            'total_events': total_events,
            'scraped_events': scraped_rows,
            'date_from': str(date_range[0]) if date_range[0] else None,
            'date_to': str(date_range[1]) if date_range[1] else None,
            'forecast_days': forecast_days
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- START ---
init_db()
t = threading.Thread(target=background_tasks, daemon=True)
t.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
