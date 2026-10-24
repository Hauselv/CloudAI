"""When and where was a photo taken?

Reads EXIF (DateTimeOriginal, OffsetTimeOriginal, GPS position and GPS UTC time stamp) and
falls back to the date/time patterns phone cameras and messengers put into file names
(``PXL_20230706_133909254`` is UTC, ``IMG_20180716_194903`` is local time,
``IMG-20220608-WA0003`` is a WhatsApp export with the date only). Each result says how
much it knows (``time_quality``), so downstream features can be honest about it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cloud_aesthetics.settings import resolve_path

_EXIF_DATETIME = 306
_EXIF_IFD = 0x8769
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_OFFSET_ORIGINAL = 36881
_GPS_IFD = 0x8825

_PXL = re.compile(r"PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
_IMG_TIME = re.compile(r"(?:IMG|VID|DSC|PANO)[_-]?(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")
_DATE_ONLY = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")


@dataclass(frozen=True)
class CaptureInfo:
    datetime_utc: str | None  # ISO 8601
    date: str | None  # YYYY-MM-DD in local terms (best effort)
    time_quality: str  # exact_utc | local_with_offset | local_estimated_offset | date_only | unknown
    time_source: str  # exif_gps | exif | filename | none
    latitude: float | None
    longitude: float | None
    location_source: str  # exif | default | none

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _dms_to_degrees(value, ref: str | None) -> float | None:
    try:
        degrees, minutes, seconds = (float(part) for part in value)
    except (TypeError, ValueError):
        return None
    result = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in {"S", "W"}:
        result = -result
    return result


def _parse_exif_datetime(text: object) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(str(text).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _parse_offset(text: object) -> timedelta | None:
    if not text:
        return None
    match = re.match(r"([+-])(\d{2}):?(\d{2})", str(text).strip())
    if not match:
        return None
    sign = 1 if match.group(1) == "+" else -1
    return sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))


def read_exif(path: str | Path) -> dict[str, object]:
    """Raw pieces we care about; empty dict when the file has no EXIF."""
    try:
        from PIL import Image

        with Image.open(resolve_path(path)) as image:
            exif = image.getexif()
            if not exif:
                return {}
            ifd = exif.get_ifd(_EXIF_IFD) if hasattr(exif, "get_ifd") else {}
            gps = exif.get_ifd(_GPS_IFD) if hasattr(exif, "get_ifd") else {}
    except Exception:  # noqa: BLE001 - unreadable EXIF is just "no EXIF"
        return {}
    return {
        "datetime_original": (ifd or {}).get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME),
        "offset_original": (ifd or {}).get(_EXIF_OFFSET_ORIGINAL),
        "gps_latitude": _dms_to_degrees(gps.get(2), gps.get(1)) if gps else None,
        "gps_longitude": _dms_to_degrees(gps.get(4), gps.get(3)) if gps else None,
        "gps_timestamp": gps.get(7) if gps else None,
        "gps_datestamp": gps.get(29) if gps else None,
    }


def _gps_utc(timestamp, datestamp) -> datetime | None:
    if not timestamp or not datestamp:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in timestamp)
        date = datetime.strptime(str(datestamp).strip(), "%Y:%m:%d")
    except (TypeError, ValueError):
        return None
    return date.replace(tzinfo=timezone.utc) + timedelta(hours=hours, minutes=minutes, seconds=seconds)


def parse_filename_datetime(name: str) -> tuple[datetime | None, str]:
    """(datetime, quality) from a file name; PXL stamps are UTC, IMG stamps local, bare dates date-only."""
    match = _PXL.search(name)
    if match:
        parts = [int(value) for value in match.groups()]
        return datetime(*parts, tzinfo=timezone.utc), "exact_utc"
    match = _IMG_TIME.search(name)
    if match:
        parts = [int(value) for value in match.groups()]
        try:
            return datetime(*parts), "local_estimated_offset"
        except ValueError:
            pass
    match = _DATE_ONLY.search(name)
    if match:
        year, month, day = (int(value) for value in match.groups())
        try:
            return datetime(year, month, day), "date_only"
        except ValueError:
            return None, "unknown"
    return None, "unknown"


def estimate_offset(longitude: float | None) -> timedelta:
    """Rough local offset from longitude (solar time); good enough for sun elevation within ~1 h."""
    if longitude is None:
        return timedelta(hours=1)  # project default region: Central Europe
    return timedelta(hours=int(round(longitude / 15.0)))


def capture_info(
    path: str | Path,
    *,
    default_latitude: float | None = None,
    default_longitude: float | None = None,
) -> CaptureInfo:
    exif = read_exif(path)
    latitude = exif.get("gps_latitude")
    longitude = exif.get("gps_longitude")
    location_source = "exif" if latitude is not None and longitude is not None else "none"
    if location_source == "none" and default_latitude is not None and default_longitude is not None:
        latitude, longitude, location_source = default_latitude, default_longitude, "default"

    utc = _gps_utc(exif.get("gps_timestamp"), exif.get("gps_datestamp"))
    local = _parse_exif_datetime(exif.get("datetime_original"))
    if utc is not None and local is not None and abs((utc.replace(tzinfo=None) - local).total_seconds()) > 36 * 3600:
        utc = None  # GPS stamp from a different day than the photo: distrust it
    if utc is not None:
        return CaptureInfo(utc.isoformat(), (local or utc).strftime("%Y-%m-%d"), "exact_utc", "exif_gps", latitude, longitude, location_source)
    if local is not None:
        offset = _parse_offset(exif.get("offset_original"))
        quality = "local_with_offset" if offset is not None else "local_estimated_offset"
        offset = offset if offset is not None else estimate_offset(longitude)
        utc_dt = (local - offset).replace(tzinfo=timezone.utc)
        return CaptureInfo(utc_dt.isoformat(), local.strftime("%Y-%m-%d"), quality, "exif", latitude, longitude, location_source)

    stamp, quality = parse_filename_datetime(Path(str(path)).name)
    if stamp is None:
        return CaptureInfo(None, None, "unknown", "none", latitude, longitude, location_source)
    if quality == "exact_utc":
        return CaptureInfo(stamp.isoformat(), stamp.strftime("%Y-%m-%d"), quality, "filename", latitude, longitude, location_source)
    if quality == "local_estimated_offset":
        utc_dt = (stamp - estimate_offset(longitude)).replace(tzinfo=timezone.utc)
        return CaptureInfo(utc_dt.isoformat(), stamp.strftime("%Y-%m-%d"), quality, "filename", latitude, longitude, location_source)
    return CaptureInfo(None, stamp.strftime("%Y-%m-%d"), "date_only", "filename", latitude, longitude, location_source)
