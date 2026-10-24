"""Historical weather for a capture (Open-Meteo archive API, free, no key).

One request per (rounded location, date) is cached as JSON under
``data/processed/context/weather_cache/`` so the archive is hit once per photo day.
Hourly variables are picked at the hour nearest the capture time; daily aggregates are
always available so date-only captures still get something.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cloud_aesthetics.settings import resolve_path

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "temperature_2m",
    "wind_speed_10m",
)
CACHE_DIR = "data/processed/context/weather_cache"


def _cache_key(latitude: float, longitude: float, date: str) -> str:
    return f"{round(latitude, 2):.2f}_{round(longitude, 2):.2f}_{date}".replace("-", "").replace(".", "p")


def fetch_archive_day(
    latitude: float,
    longitude: float,
    date: str,
    *,
    cache_dir: str | Path = CACHE_DIR,
    getter: Callable[..., Any] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any] | None:
    """Hourly archive data for one day; cached. Returns None when the request fails."""
    directory = resolve_path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / f"{_cache_key(latitude, longitude, date)}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache_path.unlink(missing_ok=True)
    params = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }
    try:
        if getter is None:
            import requests

            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout_seconds)
        else:
            response = getter(ARCHIVE_URL, params=params, timeout=timeout_seconds)
        if int(getattr(response, "status_code", 200)) >= 400:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - weather is optional context
        return None
    if not isinstance(payload, dict) or "hourly" not in payload:
        return None
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def weather_features(payload: dict[str, Any] | None, capture_utc: datetime | None) -> dict[str, float | None]:
    """Nearest-hour values (``wx_*``) and daily aggregates (``wx_day_*``) from an archive payload."""
    out: dict[str, float | None] = {f"wx_{name}": None for name in HOURLY_VARIABLES}
    out.update({"wx_day_cloud_cover_mean": None, "wx_day_cloud_cover_min": None, "wx_day_precipitation_sum": None, "wx_day_humidity_mean": None, "wx_hour_offset": None})
    if not payload:
        return out
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return out

    def _clean(values: list[Any]) -> list[float]:
        return [float(v) for v in values if v is not None]

    cover = _clean(hourly.get("cloud_cover") or [])
    if cover:
        out["wx_day_cloud_cover_mean"] = sum(cover) / len(cover)
        out["wx_day_cloud_cover_min"] = min(cover)
    precipitation = _clean(hourly.get("precipitation") or [])
    if precipitation:
        out["wx_day_precipitation_sum"] = sum(precipitation)
    humidity = _clean(hourly.get("relative_humidity_2m") or [])
    if humidity:
        out["wx_day_humidity_mean"] = sum(humidity) / len(humidity)
    if capture_utc is None:
        return out
    capture_utc = capture_utc.astimezone(timezone.utc) if capture_utc.tzinfo else capture_utc.replace(tzinfo=timezone.utc)
    best_index, best_delta = None, None
    for index, stamp in enumerate(times):
        try:
            when = datetime.fromisoformat(str(stamp)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = abs((when - capture_utc).total_seconds())
        if best_delta is None or delta < best_delta:
            best_index, best_delta = index, delta
    if best_index is None:
        return out
    out["wx_hour_offset"] = (best_delta or 0.0) / 3600.0
    for name in HOURLY_VARIABLES:
        values = hourly.get(name) or []
        value = values[best_index] if best_index < len(values) else None
        out[f"wx_{name}"] = float(value) if value is not None else None
    return out
