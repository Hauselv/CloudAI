from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from cloud_aesthetics.features.capture import capture_info, estimate_offset, parse_filename_datetime
from cloud_aesthetics.features.context import build_context_table
from cloud_aesthetics.features.palette import extract_palette_features
from cloud_aesthetics.features.solar import classify_phase, sun_position
from cloud_aesthetics.features.weather import fetch_archive_day, weather_features


def test_sun_position_matches_known_values():
    # Mainz, 2023-06-21 12:00 UTC (~13:30 solar time): high sun around 63 deg
    noon = sun_position(datetime(2023, 6, 21, 12, 0, tzinfo=timezone.utc), 49.99, 8.27)
    assert 60 < noon.elevation_deg < 65 and noon.phase == "day"
    # Sunset in Mainz on 2023-06-21 is about 19:38 UTC
    sunset = sun_position(datetime(2023, 6, 21, 19, 38, tzinfo=timezone.utc), 49.99, 8.27)
    assert -1.5 < sunset.elevation_deg < 0.5 and sunset.phase == "golden_hour"
    assert sunset.hours_from_sunset is not None and abs(sunset.hours_from_sunset) < 0.25
    assert 290 < sunset.azimuth_deg < 320
    midnight = sun_position(datetime(2023, 12, 21, 0, 0, tzinfo=timezone.utc), 49.99, 8.27)
    assert midnight.elevation_deg < -50 and midnight.phase == "night"
    assert classify_phase(-5) == "blue_hour" and classify_phase(3) == "golden_hour" and classify_phase(10) == "low_sun"


def test_filename_parsing_and_offsets():
    stamp, quality = parse_filename_datetime("PXL_20230706_133909254.jpg")
    assert quality == "exact_utc" and stamp == datetime(2023, 7, 6, 13, 39, 9, tzinfo=timezone.utc)
    stamp, quality = parse_filename_datetime("IMG_20180716_194903.jpg")
    assert quality == "local_estimated_offset" and stamp == datetime(2018, 7, 16, 19, 49, 3)
    stamp, quality = parse_filename_datetime("IMG-20220608-WA0003.jpg")
    assert quality == "date_only" and stamp == datetime(2022, 6, 8)
    assert parse_filename_datetime("cloud.jpg") == (None, "unknown")
    assert estimate_offset(8.27).total_seconds() == 3600 and estimate_offset(-74.0).total_seconds() == -5 * 3600


def test_capture_info_from_filename_with_default_location(tmp_path: Path):
    path = tmp_path / "IMG-20220608-WA0003.jpg"
    cv2.imwrite(str(path), np.zeros((20, 20, 3), dtype=np.uint8))
    info = capture_info(path, default_latitude=49.99, default_longitude=8.27)
    assert info.time_quality == "date_only" and info.date == "2022-06-08" and info.datetime_utc is None
    assert info.location_source == "default" and info.latitude == 49.99
    path2 = tmp_path / "PXL_20230706_133909254.jpg"
    cv2.imwrite(str(path2), np.zeros((20, 20, 3), dtype=np.uint8))
    info2 = capture_info(path2)
    assert info2.time_quality == "exact_utc" and info2.time_source == "filename" and info2.location_source == "none"


def test_capture_info_reads_exif(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "photo.jpg"
    image = Image.new("RGB", (16, 16), (120, 160, 220))
    exif = Image.Exif()
    exif[306] = "2023:07:06 18:52:04"
    exif.get_ifd(0x8769)[36867] = "2023:07:06 18:52:04"
    exif.get_ifd(0x8769)[36881] = "+02:00"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (50.0, 0.0, 34.73)
    gps[3] = "E"
    gps[4] = (8.0, 16.0, 12.0)
    image.save(path, exif=exif)
    info = capture_info(path)
    assert info.location_source == "exif" and abs(info.latitude - 50.0096) < 0.01 and abs(info.longitude - 8.27) < 0.01
    assert info.time_quality == "local_with_offset" and info.datetime_utc.startswith("2023-07-06T16:52:04")


def test_weather_features_and_cache(tmp_path: Path):
    calls: list[dict] = []
    payload = {
        "hourly": {
            "time": [f"2023-07-06T{hour:02d}:00" for hour in range(24)],
            "cloud_cover": [10 * (hour % 10) for hour in range(24)],
            "cloud_cover_low": [1] * 24,
            "cloud_cover_mid": [2] * 24,
            "cloud_cover_high": [3] * 24,
            "relative_humidity_2m": [50] * 24,
            "surface_pressure": [1013] * 24,
            "precipitation": [0.0] * 23 + [1.5],
            "temperature_2m": [20] * 24,
            "wind_speed_10m": [5] * 24,
        }
    }

    def getter(url, params=None, timeout=None):
        calls.append(params)
        return SimpleNamespace(status_code=200, json=lambda: payload)

    first = fetch_archive_day(49.99, 8.27, "2023-07-06", cache_dir=tmp_path, getter=getter)
    second = fetch_archive_day(49.99, 8.27, "2023-07-06", cache_dir=tmp_path, getter=getter)
    assert first == payload and second == payload and len(calls) == 1
    assert calls[0]["hourly"].startswith("cloud_cover,")
    features = weather_features(payload, datetime(2023, 7, 6, 16, 52, tzinfo=timezone.utc))
    assert features["wx_cloud_cover"] == 70.0  # 17:00 is the nearest hour, 17 % 10 = 7
    assert features["wx_hour_offset"] == pytest.approx(8 / 60, abs=1e-6)
    assert features["wx_day_precipitation_sum"] == 1.5
    date_only = weather_features(payload, None)
    assert date_only["wx_cloud_cover"] is None and date_only["wx_day_cloud_cover_mean"] is not None
    failing = fetch_archive_day(1.0, 2.0, "2023-07-06", cache_dir=tmp_path, getter=lambda *a, **k: SimpleNamespace(status_code=500, json=dict))
    assert failing is None


def test_palette_features_separate_sunset_from_blue_sky():
    sunset = np.zeros((120, 160, 3), dtype=np.uint8)
    sunset[:60] = [250, 140, 40]
    sunset[60:] = [230, 90, 120]
    blue = np.zeros((120, 160, 3), dtype=np.uint8)
    blue[:] = [70, 130, 230]
    blue[30:60, 40:120] = [240, 240, 240]
    warm = extract_palette_features(sunset)
    cool = extract_palette_features(blue)
    assert warm["palette_warm_share"] > 0.9 and cool["palette_warm_share"] < 0.1
    assert cool["palette_blue_share"] > 0.5 and warm["palette_blue_share"] == 0.0
    assert cool["palette_neutral_share"] > 0.1  # the white block covers 12.5 % of the image
    assert warm["colorfulness"] > cool["colorfulness"] * 0.5
    assert 0.0 <= warm["palette_hue_concentration"] <= 1.0
    assert set(warm) == set(cool) and len(warm) == 13


def test_context_table_propagates_source_to_crops(tmp_path: Path):
    original = tmp_path / "PXL_20230706_133909254.jpg"
    crop = tmp_path / "crops" / "x_crop01.jpg"
    crop.parent.mkdir()
    cv2.imwrite(str(original), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(crop), np.zeros((10, 10, 3), dtype=np.uint8))
    manifest = pd.DataFrame({"image_id": ["o", "c"], "relative_path": [str(original), str(crop)], "kind": ["original", "crop"]})
    derivatives = pd.DataFrame([{"relative_path": str(crop), "source_relative_path": str(original), "derivative_kind": "sky_crop"}])
    table = build_context_table(manifest, default_latitude=49.99, default_longitude=8.27, weather=False, derivatives=derivatives)
    by_id = table.set_index("image_id")
    assert by_id.loc["o", "time_quality"] == "exact_utc" and by_id.loc["c", "context_source"] == "source_original"
    assert by_id.loc["c", "datetime_utc"] == by_id.loc["o", "datetime_utc"]
    assert by_id.loc["o", "sun_phase"] in {"day", "low_sun"} and by_id.loc["o", "sun_phase_day"] in {0.0, 1.0}
    assert by_id.loc["o", "weather_available"] is np.False_ or by_id.loc["o", "weather_available"] == False  # noqa: E712
