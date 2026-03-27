# TrainJAX — Code Report

## Overview

This project is a Flask-based web application called **TrainJAX** that predicts train delays on the Manchester Piccadilly → London Euston route using a JAX/Flax neural network. It combines live rail data, a PostgreSQL database, and a single-page frontend.

---

## `app.py` — Flask Backend

### Purpose
`app.py` is the main server-side application. It handles data collection, machine learning model training, and exposes a REST API consumed by the frontend.

---

### Key Libraries

| Library | Purpose |
|---|---|
| `Flask` / `Flask-CORS` | Web server and route handling |
| `JAX` / `Flax` / `Optax` | Neural network definition, training, and optimisation |
| `psycopg2` | PostgreSQL database access |
| `pandas` / `numpy` | Data processing and feature engineering |
| `scikit-learn` (`MinMaxScaler`) | Data normalisation before model training |
| `requests` / `urllib3` | HTTP calls to external rail APIs |

---

### Configuration

On startup, two external API keys are read from environment variables (with hardcoded defaults as fallbacks):

- **`RAIL_API_KEY`** — Used to authenticate against the Network Rail Staff Departure Board API (`raildata.org.uk`).
- **`TRANSPORT_API_ID` / `TRANSPORT_API_KEY`** — Used with TransportAPI for journey planning.
- **`DATABASE_URL`** — PostgreSQL connection string.

The route corridor is fixed as 8 stations:  
`MAN → SPT → MAC → SOT → CRE → RUG → MKC → EUS`

---

### Neural Network: `JaxRouteModel`

A simple feedforward neural network built with Flax `linen`:

```
Input (14-day window) → Flatten → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)
```

- **Input**: A 14-day sliding window of normalised daily total delay minutes.
- **Output**: A single value — predicted delay for the next day.
- **Optimiser**: Adam (learning rate 0.001) via Optax.
- **Training**: 51 epochs using MSE loss; JIT-compiled with JAX for performance.

---

### Database Tables

Three PostgreSQL tables are created on startup (`init_db`):

| Table | Description |
|---|---|
| `rail_events` | Stores raw delay data scraped from live departures (used as training data) |
| `jax_predictions` | Stores both historical actuals and 30-day future forecasts produced by the model |
| `live_snapshot` | Stores the most recent live delay status per station |

---

### Core Functions

#### `scan_live_departures()`
- Queries the Network Rail API for each station on the route.
- Calculates delay in minutes per station (including 60-min penalty for cancellations).
- Writes a snapshot to the `live_snapshot` table.

#### `scrape_and_store()`
- Similar to the live scanner but stores per-station delay readings into `rail_events` to accumulate training data over time.

#### `run_jax_training()`
1. Loads all `rail_events` from the database.
2. Aggregates to daily totals.
3. Splits data 80% train / 20% test.
4. Normalises with `MinMaxScaler`.
5. Trains `JaxRouteModel` for 51 epochs.
6. Evaluates RMSE on test split.
7. Rolls forward 30 days of future forecasts using the trained model.
8. Saves all predictions (historical + forecast) to `jax_predictions`.

#### `background_tasks()` (runs in a daemon thread)
Runs on a continuous schedule:
- **Every 60 seconds** — scrapes and stores new live departure data.
- **Every 5 minutes** — refreshes the live snapshot.
- **Every 6 hours** — retrains the JAX model from scratch on accumulated data.

---

### API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Renders `index.html` |
| `GET` | `/api/departures/<crs>` | Proxies live departures for any UK station CRS code |
| `GET` | `/api/journey` | Plans a journey (from, to, date, time) via TransportAPI timetables; also returns JAX confidence for that date |
| `GET` | `/api/stations` | Returns all 8 route stations with names, CRS codes, and coordinates |
| `GET` | `/api/station_timetable/<crs>` | Raw TransportAPI timetable for a route station |
| `GET` | `/api/predictions` | Returns all rows from `jax_predictions` |
| `GET` | `/api/live` | Returns the latest `live_snapshot` data |
| `GET` | `/api/plan/<date>` | Returns JAX delay prediction + confidence level for a given date |
| `GET` | `/health` | Simple health check returning server time |
| `GET` | `/api/stats` | Returns database stats (event count, date range, forecast days) |

---

### Startup Sequence

1. `init_db()` — creates database tables if missing.
2. A daemon thread starts `background_tasks()`, which immediately runs an initial scan + scrape + JAX training, then continues on its schedule.
3. Flask listens on `0.0.0.0:8080` (or the `PORT` environment variable).

---

---

## `templates/index.html` — Frontend

### Purpose
`index.html` is a self-contained single-page application (SPA) served by Flask. It provides four interactive tabs allowing users to plan journeys, view live departures, check AI confidence, and browse the 30-day delay forecast.

---

### Styling

- **Dark theme** using CSS custom properties (`--bg`, `--surface`, `--accent`, `--red`, `--green`, etc.).
- **Fonts**: JetBrains Mono (monospace data display) and DM Sans (body text), loaded from Google Fonts.
- **Responsive** — collapses to single-column layout below 768px.
- A subtle grid overlay is applied via a CSS `::before` pseudo-element for visual depth.
- CSS animations: `pulse` (status indicator), `spin` (loading spinner), `fadeUp` (tab transitions).

---

### Tab Structure

#### Tab 1 — Journey Planner
- Form inputs: **From** station, **To** station, **Date**, **Time** (with a swap button).
- Calls `GET /api/journey` and renders:
  - A **JAX prediction banner** showing confidence level and colour-coded stress coefficient for the travel date.
  - A list of **journey cards**, each showing departure/arrival times, duration, and leg details.

#### Tab 2 — Departure Board
- User enters any UK station **CRS code** (3-letter code, e.g. `MAN`).
- Calls `GET /api/departures/<crs>` and renders a styled table with departure time, destination, platform, status (colour-coded green/red), and operator.

#### Tab 3 — AI Confidence
- User selects a **travel date**.
- Calls `GET /api/plan/<date>` and displays:
  - A circular confidence badge (colour-coded: High → Poor).
  - Predicted delay in minutes, stress coefficient (%), and model RMSE (± minutes).
  - A text recommendation ("Excellent day to travel", "Significant delays likely", etc.).

#### Tab 4 — 30-Day Forecast
- Loaded automatically on page load via `GET /api/predictions`.
- Renders a grid of forecast day cards showing date, predicted delay (mins), stress percentage, and a colour-coded stress bar.
- Clicking a forecast day switches to the AI Confidence tab and loads that date.
- Summary statistics shown: model RMSE, number of training days, last updated timestamp.

---

### JavaScript Functions

| Function | Description |
|---|---|
| `switchTab(el, name)` | Switches active tab and content panel |
| `swapStations()` | Swaps the From/To station dropdowns |
| `planTrip()` | Fetches and renders journey results from `/api/journey` |
| `getBoard()` | Fetches and renders live departures from `/api/departures/<crs>` |
| `checkConfidence()` | Fetches and renders JAX prediction for a date from `/api/plan/<date>` |
| `loadForecast()` | Fetches all predictions from `/api/predictions` and renders the forecast grid |
| `selectForecastDate(ds)` | Navigates from forecast grid to AI Confidence tab for a chosen date |
| `stressColor(s)` | Returns a hex colour based on stress coefficient (0–1 scale) |

---

### External Dependencies (CDN)

- Google Fonts: `JetBrains Mono` and `DM Sans` (CSS only, no JavaScript frameworks).
- No JavaScript libraries or frameworks — all logic is plain vanilla JavaScript using the Fetch API.

---

## Summary

The application is a full-stack AI rail delay prediction tool:

- **Backend (`app.py`)** collects live train data from the Network Rail API, stores it in PostgreSQL, trains a JAX neural network on that data, and serves predictions and live data through a REST API.
- **Frontend (`index.html`)** is a polished dark-theme SPA that presents the model's predictions alongside live departures and a journey planner, all without any JavaScript framework dependencies.
