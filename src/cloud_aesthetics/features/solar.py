"""Sun position (NOAA solar calculator approximation) and light phases.

Accurate to a few tenths of a degree, which is far below the uncertainty of the capture
time for most photos. No external dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

PHASES = ("night", "astronomical_twilight", "nautical_twilight", "blue_hour", "golden_hour", "low_sun", "day")


@dataclass(frozen=True)
class SunPosition:
    elevation_deg: float
    azimuth_deg: float
    phase: str
    hours_from_sunset: float | None  # negative before sunset, positive after; None for polar cases


def _julian_day(when: datetime) -> float:
    when = when.astimezone(timezone.utc)
    year, month = when.year, when.month
    day = when.day + (when.hour + when.minute / 60.0 + when.second / 3600.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5


def _solar_geometry(when: datetime) -> tuple[float, float]:
    """(declination in degrees, equation of time in minutes)."""
    jd = _julian_day(when)
    t = (jd - 2451545.0) / 36525.0
    mean_long = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccent = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m = math.radians(mean_anom)
    center = math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * m) * (0.019993 - 0.000101 * t) + math.sin(3 * m) * 0.000289
    true_long = mean_long + center
    omega = 125.04 - 1934.136 * t
    apparent_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    mean_obliq = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    obliq = mean_obliq + 0.00256 * math.cos(math.radians(omega))
    declination = math.degrees(math.asin(math.sin(math.radians(obliq)) * math.sin(math.radians(apparent_long))))
    y = math.tan(math.radians(obliq / 2.0)) ** 2
    l0 = math.radians(mean_long)
    eq_time = 4.0 * math.degrees(
        y * math.sin(2 * l0) - 2 * eccent * math.sin(m) + 4 * eccent * y * math.sin(m) * math.cos(2 * l0) - 0.5 * y * y * math.sin(4 * l0) - 1.25 * eccent * eccent * math.sin(2 * m)
    )
    return declination, eq_time


def sun_position(when: datetime, latitude: float, longitude: float) -> SunPosition:
    when = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    declination, eq_time = _solar_geometry(when)
    minutes_utc = when.hour * 60.0 + when.minute + when.second / 60.0
    true_solar = (minutes_utc + eq_time + 4.0 * longitude) % 1440.0
    hour_angle = true_solar / 4.0 - 180.0
    lat = math.radians(latitude)
    dec = math.radians(declination)
    ha = math.radians(hour_angle)
    cos_zenith = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    zenith = math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))
    elevation = 90.0 - zenith
    # Atmospheric refraction near the horizon (NOAA)
    if elevation > 85:
        refraction = 0.0
    elif elevation > 5:
        te = math.tan(math.radians(elevation))
        refraction = (58.1 / te - 0.07 / te**3 + 0.000086 / te**5) / 3600.0
    elif elevation > -0.575:
        refraction = (1735 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)))) / 3600.0
    else:
        refraction = (-20.774 / math.tan(math.radians(elevation))) / 3600.0
    elevation += refraction
    denom = math.cos(lat) * math.sin(math.radians(zenith))
    if abs(denom) < 1e-9:
        azimuth = 180.0
    else:
        cos_az = (math.sin(lat) * math.cos(math.radians(zenith)) - math.sin(dec)) / denom
        azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
        azimuth = (azimuth + 180.0) % 360.0 if hour_angle > 0 else (540.0 - azimuth) % 360.0
    # Hours from sunset (geometric, -0.833 deg)
    cos_ha0 = (math.cos(math.radians(90.833)) / (math.cos(lat) * math.cos(dec))) - math.tan(lat) * math.tan(dec)
    hours_from_sunset: float | None
    if abs(cos_ha0) > 1:
        hours_from_sunset = None
    else:
        ha_sunset = math.degrees(math.acos(cos_ha0))
        sunset_minutes = 720.0 - 4.0 * (longitude - ha_sunset) - eq_time  # UTC minutes (NOAA: sunrise uses lon + ha)
        delta = minutes_utc - sunset_minutes
        if delta > 720:
            delta -= 1440
        elif delta < -720:
            delta += 1440
        hours_from_sunset = delta / 60.0
    return SunPosition(round(elevation, 3), round(azimuth, 2), classify_phase(elevation), hours_from_sunset)


def classify_phase(elevation_deg: float) -> str:
    if elevation_deg < -18:
        return "night"
    if elevation_deg < -12:
        return "astronomical_twilight"
    if elevation_deg < -6:
        return "nautical_twilight"
    if elevation_deg < -4:
        return "blue_hour"
    if elevation_deg < 6:
        return "golden_hour"
    if elevation_deg < 15:
        return "low_sun"
    return "day"
