from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import pandas as pd
import requests

from cloud_aesthetics.settings import ensure_parent, load_yaml, resolve_path
from cloud_aesthetics.utils.io import write_table

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API_URL = "https://api.openverse.engineering/v1/images/"
USER_AGENT = "CloudAestheticsResearchWorkbench/0.1 (local research dataset builder)"


@dataclass(slots=True)
class DownloadConfig:
    output_dir: Path
    metadata_path: Path
    source: str
    target_count: int
    min_width: int
    min_height: int
    target_width: int | None
    target_height: int | None
    download_width: int
    request_delay_seconds: float
    search_limit_per_query: int
    max_pages_per_query: int
    reset_output: bool
    categories: list[str]
    queries: list[str]
    allowed_licenses: list[str]


def load_download_config(config_path: str | Path) -> DownloadConfig:
    raw = load_yaml(config_path)
    return DownloadConfig(
        output_dir=resolve_path(raw.get("output_dir", "data/raw/images/web_sample")),
        metadata_path=resolve_path(raw.get("metadata_path", "data/raw/metadata/web_sample_metadata.parquet")),
        source=str(raw.get("source", "openverse")),
        target_count=int(raw.get("target_count", 120)),
        min_width=int(raw.get("min_width", 1024)),
        min_height=int(raw.get("min_height", 768)),
        target_width=int(raw["target_width"]) if raw.get("target_width") else None,
        target_height=int(raw["target_height"]) if raw.get("target_height") else None,
        download_width=int(raw.get("download_width", 1600)),
        request_delay_seconds=float(raw.get("request_delay_seconds", 0.2)),
        search_limit_per_query=int(raw.get("search_limit_per_query", 100)),
        max_pages_per_query=int(raw.get("max_pages_per_query", 4)),
        reset_output=bool(raw.get("reset_output", False)),
        categories=list(raw.get("categories", [])),
        queries=list(raw.get("queries", [])),
        allowed_licenses=[_normalise_license_text(str(item)) for item in raw.get("allowed_licenses", [])],
    )


def _plain_metadata(extmetadata: dict[str, dict[str, object]], key: str) -> str:
    value = extmetadata.get(key, {}).get("value", "")
    return re.sub(r"<[^>]+>", "", str(value)).strip()


def _safe_filename(title: str, fallback_extension: str = ".jpg") -> str:
    name = title.removeprefix("File:")
    name = unquote(name)
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
        suffix = fallback_extension
        stem = name
    else:
        stem = name[: -len(suffix)]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    safe_stem = safe_stem.rstrip(" .")
    if not safe_stem:
        safe_stem = "image"
    return f"{safe_stem[:120]}{suffix}"


def _normalise_license_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _license_allowed(short_name: str, usage_terms: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    text = _normalise_license_text(f"{short_name} {usage_terms}")
    return any(term in text for term in allowed)


def _search_commons(query: str, download_width: int, limit: int = 100, max_pages: int = 4) -> list[dict[str, object]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": download_width,
    }
    pages: list[dict[str, object]] = []
    continuation: dict[str, object] = {}
    for _ in range(max_pages):
        response = requests.get(
            COMMONS_API_URL,
            params={**params, **continuation},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        pages.extend(payload.get("query", {}).get("pages", {}).values())
        continuation = payload.get("continue", {})
        if not continuation:
            break
    return pages


def _category_commons(category: str, download_width: int, limit: int = 100, max_pages: int = 4) -> list[dict[str, object]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": download_width,
    }
    pages: list[dict[str, object]] = []
    continuation: dict[str, object] = {}
    for _ in range(max_pages):
        response = requests.get(
            COMMONS_API_URL,
            params={**params, **continuation},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        pages.extend(payload.get("query", {}).get("pages", {}).values())
        continuation = payload.get("continue", {})
        if not continuation:
            break
    return pages


def _download_file(url: str, destination: Path) -> None:
    ensure_parent(destination)
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def _normalise_image_resolution(path: Path, target_width: int, target_height: int) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to normalise unreadable image: {path}")
    height, width = image.shape[:2]
    scale = max(target_width / width, target_height / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    x0 = max((resized_width - target_width) // 2, 0)
    y0 = max((resized_height - target_height) // 2, 0)
    cropped = resized[y0 : y0 + target_height, x0 : x0 + target_width]
    cv2.imwrite(str(path), cropped)


def _search_openverse(query: str, page_size: int = 50, max_pages: int = 8) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    page_size = min(page_size, 20)
    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "page": page,
            "page_size": page_size,
            "license_type": "commercial,modification",
        }
        response = requests.get(OPENVERSE_API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        results.extend(payload.get("results", []))
        if page >= int(payload.get("page_count", page)):
            break
    return results


def _download_openverse_dataset(config: DownloadConfig) -> pd.DataFrame:
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    rows: list[dict[str, object]] = []
    for query in config.queries:
        if len(rows) >= config.target_count:
            break
        for item in _search_openverse(query, config.search_limit_per_query, config.max_pages_per_query):
            if len(rows) >= config.target_count:
                break
            item_id = str(item.get("id", ""))
            url = str(item.get("url", ""))
            license_name = str(item.get("license", ""))
            if not item_id or item_id in seen_ids or not url or url in seen_urls:
                continue
            if not _license_allowed(license_name, str(item.get("license_version", "")), config.allowed_licenses):
                continue
            parsed_suffix = Path(urlparse(url).path).suffix.lower()
            if parsed_suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                parsed_suffix = ".jpg"
            title = str(item.get("title") or item_id)
            destination = config.output_dir / query.replace(" ", "_").lower() / _safe_filename(f"{item_id}_{title}", parsed_suffix)
            try:
                _download_file(url, destination)
                dimensions = _image_dimensions(destination)
                if dimensions is None:
                    destination.unlink(missing_ok=True)
                    continue
                width, height = dimensions
                if width < config.min_width or height < config.min_height:
                    destination.unlink(missing_ok=True)
                    continue
                original_width, original_height = width, height
                if config.target_width and config.target_height:
                    _normalise_image_resolution(destination, config.target_width, config.target_height)
                    width, height = config.target_width, config.target_height
            except Exception:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            seen_ids.add(item_id)
            seen_urls.add(url)
            rows.append(
                {
                    "source": "openverse",
                    "collection_type": "query",
                    "collection": query,
                    "title": title,
                    "file_path": destination.relative_to(resolve_path(".")).as_posix(),
                    "width": width,
                    "height": height,
                    "original_width": original_width,
                    "original_height": original_height,
                    "source_url": str(item.get("foreign_landing_url", "")),
                    "download_url": url,
                    "artist": str(item.get("creator", "")),
                    "credit": str(item.get("provider", "")),
                    "license": license_name,
                    "usage_terms": str(item.get("license_version", "")),
                    "license_url": str(item.get("license_url", "")),
                    "attribution_required": "true",
                    "copyrighted": "true",
                }
            )
            time.sleep(config.request_delay_seconds)
    return pd.DataFrame(rows)


def download_wikimedia_cloud_dataset(config_path: str | Path) -> pd.DataFrame:
    config = load_download_config(config_path)
    if config.reset_output and config.output_dir.exists():
        resolved_output = config.output_dir.resolve()
        raw_images_root = resolve_path("data/raw/images").resolve()
        if raw_images_root in resolved_output.parents or resolved_output == raw_images_root:
            shutil.rmtree(resolved_output)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.source == "openverse":
        metadata = _download_openverse_dataset(config)
        write_table(metadata, config.metadata_path)
        metadata.to_csv(config.metadata_path.with_suffix(".csv"), index=False)
        return metadata

    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    rows: list[dict[str, object]] = []

    collections = [
        ("category", category, _category_commons(category, config.download_width, config.search_limit_per_query, config.max_pages_per_query))
        for category in config.categories
    ]
    collections.extend(
        [
            ("query", query, _search_commons(query, config.download_width, config.search_limit_per_query, config.max_pages_per_query))
            for query in config.queries
        ]
    )

    for collection_type, collection_name, pages in collections:
        if len(rows) >= config.target_count:
            break
        for page in pages:
            if len(rows) >= config.target_count:
                break
            title = str(page.get("title", ""))
            if not title or title in seen_titles:
                continue
            imageinfo = (page.get("imageinfo") or [{}])[0]
            extmetadata = imageinfo.get("extmetadata", {})
            short_license = _plain_metadata(extmetadata, "LicenseShortName")
            usage_terms = _plain_metadata(extmetadata, "UsageTerms")
            if not _license_allowed(short_license, usage_terms, config.allowed_licenses):
                continue
            url = str(imageinfo.get("thumburl") or imageinfo.get("url") or "")
            if not url or url in seen_urls:
                continue
            parsed_suffix = Path(urlparse(url).path).suffix.lower()
            if parsed_suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                parsed_suffix = ".jpg"
            group_name = collection_name.removeprefix("Category:").replace(" ", "_").lower()
            destination = config.output_dir / group_name / _safe_filename(title, parsed_suffix)
            if destination.exists():
                seen_titles.add(title)
                seen_urls.add(url)
                continue
            try:
                _download_file(url, destination)
                dimensions = _image_dimensions(destination)
                if dimensions is None:
                    destination.unlink(missing_ok=True)
                    continue
                width, height = dimensions
                if width < config.min_width or height < config.min_height:
                    destination.unlink(missing_ok=True)
                    continue
            except Exception:
                destination.unlink(missing_ok=True)
                continue
            seen_titles.add(title)
            seen_urls.add(url)
            rows.append(
                {
                    "source": "wikimedia_commons",
                    "collection_type": collection_type,
                    "collection": collection_name,
                    "title": title,
                    "file_path": destination.relative_to(resolve_path(".")).as_posix(),
                    "width": width,
                    "height": height,
                    "source_url": str(imageinfo.get("descriptionurl", "")),
                    "download_url": url,
                    "artist": _plain_metadata(extmetadata, "Artist"),
                    "credit": _plain_metadata(extmetadata, "Credit"),
                    "license": short_license,
                    "usage_terms": usage_terms,
                    "attribution_required": _plain_metadata(extmetadata, "AttributionRequired"),
                    "copyrighted": _plain_metadata(extmetadata, "Copyrighted"),
                }
            )
            time.sleep(config.request_delay_seconds)

    metadata = pd.DataFrame(rows)
    write_table(metadata, config.metadata_path)
    metadata_csv = config.metadata_path.with_suffix(".csv")
    metadata.to_csv(metadata_csv, index=False)
    return metadata
