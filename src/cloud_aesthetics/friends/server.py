"""The friend labeling server: a FastAPI app with one page and a handful of JSON endpoints.

Security model: every route needs a valid, active friend token (anything else is 404);
images are addressed by manifest image id only, never by path; the only writes are
append-only rating / duel rows and undo tombstones for the token's own rater; a simple
per-token rate limit caps request bursts. Bind to localhost and put a tunnel (and, when
available, Cloudflare Access) in front.
"""

from __future__ import annotations

import io
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from cloud_aesthetics.data.exclusions import active_excluded_ids
from cloud_aesthetics.data.manifest import ensure_kind_columns
from cloud_aesthetics.data.pairwise import select_duel_pair
from cloud_aesthetics.data.ratings import load_raw_pairwise_preferences, load_raw_scalar_ratings, record_pairwise_preference, record_rating, void_rating
from cloud_aesthetics.friends.queue import FriendQueueConfig, build_friend_queue, next_items, progress
from cloud_aesthetics.friends.registry import FRIENDS_PATH, resolve_token
from cloud_aesthetics.friends.ui import PAGE
from cloud_aesthetics.judge.store import RATERS_PATH
from cloud_aesthetics.settings import resolve_path
from cloud_aesthetics.utils.io import read_table


@dataclass
class FriendServerConfig:
    manifest_path: str | Path = "data/processed/image_manifest.parquet"
    ratings_dir: str | Path = "data/raw/metadata/ratings"
    pairwise_dir: str | Path = "data/raw/metadata/pairwise"
    friends_path: str | Path = FRIENDS_PATH
    raters_path: str | Path = RATERS_PATH
    exclusions_path: str | Path = "data/raw/metadata/exclusions.csv"
    map_queue_path: str | Path = "data/processed/embeddings/map_queue.parquet"
    triage_path: str | Path = "data/processed/judge/triage.parquet"
    cache_dir: str | Path = "data/processed/friend_cache"
    image_edge: int = 1280
    jpeg_quality: int = 84
    queue: FriendQueueConfig = field(default_factory=FriendQueueConfig)
    requests_per_minute: int = 240


class RateBody(BaseModel):
    image_id: str = Field(min_length=1, max_length=64)
    score: float = Field(ge=0.0, le=10.0)
    note: str | None = Field(default=None, max_length=300)


class UndoBody(BaseModel):
    rating_id: str = Field(min_length=1, max_length=64)


class DuelBody(BaseModel):
    left: str = Field(min_length=1, max_length=64)
    right: str = Field(min_length=1, max_length=64)
    winner: str | None = Field(default=None, max_length=64)


class FriendService:
    """State and data access shared by the endpoints (manifest cache, queues, rate limits)."""

    def __init__(self, config: FriendServerConfig) -> None:
        self.config = config
        self._manifest: pd.DataFrame | None = None
        self._manifest_sig: tuple[float, int] | None = None
        self._queues: dict[str, tuple[int, list[str]]] = {}
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    # ---- data
    def manifest(self) -> pd.DataFrame:
        path = resolve_path(self.config.manifest_path)
        if not path.exists():
            return pd.DataFrame(columns=["image_id", "relative_path"])
        stat = path.stat()
        signature = (stat.st_mtime, stat.st_size)
        if self._manifest is None or self._manifest_sig != signature:
            frame = ensure_kind_columns(read_table(path))
            frame["image_id"] = frame["image_id"].astype(str)
            self._manifest = frame.drop_duplicates("image_id").reset_index(drop=True)
            self._manifest_sig = signature
            self._queues.clear()
        return self._manifest

    def image_row(self, image_id: str) -> pd.Series | None:
        frame = self.manifest()
        hit = frame[frame["image_id"] == str(image_id)]
        return hit.iloc[0] if not hit.empty else None

    def ratings(self) -> pd.DataFrame:
        return load_raw_scalar_ratings(self.config.ratings_dir)

    def rated_ids(self, rater_id: str, ratings: pd.DataFrame | None = None) -> set[str]:
        frame = self.ratings() if ratings is None else ratings
        if frame.empty:
            return set()
        return set(frame.loc[frame["rater_id"].astype(str) == rater_id, "image_id"].astype(str))

    def queue(self, friend: dict[str, object]) -> list[str]:
        token = str(friend["token"])
        version = len(self.manifest())
        cached = self._queues.get(token)
        if cached and cached[0] == version:
            return cached[1]
        ratings = self.ratings()
        core_ids = self.rated_ids(self.config.queue.core_rater, ratings)
        map_queue = read_table(self.config.map_queue_path) if resolve_path(self.config.map_queue_path).exists() else None
        triage = read_table(self.config.triage_path) if resolve_path(self.config.triage_path).exists() else None
        excluded = active_excluded_ids(self.config.exclusions_path) if resolve_path(self.config.exclusions_path).exists() else set()
        queue = build_friend_queue(self.manifest(), token=token, core_rated_ids=core_ids, excluded_ids=excluded, map_queue=map_queue, triage=triage, config=self.config.queue)
        self._queues[token] = (version, queue)
        return queue

    def progress(self, friend: dict[str, object], ratings: pd.DataFrame | None = None) -> dict[str, int]:
        ratings = self.ratings() if ratings is None else ratings
        return progress(self.queue(friend), self.rated_ids(str(friend["rater_id"]), ratings), self.rated_ids(self.config.queue.core_rater, ratings))

    # ---- images
    def cached_image(self, image_id: str) -> Path | None:
        row = self.image_row(image_id)
        if row is None:
            return None
        source = resolve_path(str(row["relative_path"]))
        if not source.exists():
            return None
        cache_dir = resolve_path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{image_id}_{self.config.image_edge}.jpg"
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        from PIL import Image, ImageOps

        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((self.config.image_edge, self.config.image_edge))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.config.jpeg_quality, optimize=True)
        target.write_bytes(buffer.getvalue())
        return target

    # ---- guards
    def authenticate(self, token: str) -> dict[str, object]:
        friend = resolve_token(token, path_like=self.config.friends_path)
        if friend is None:
            raise HTTPException(status_code=404, detail="not found")
        self._rate_limit(token)
        return friend

    def _rate_limit(self, token: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[token]
            while hits and now - hits[0] > 60.0:
                hits.popleft()
            if len(hits) >= self.config.requests_per_minute:
                raise HTTPException(status_code=429, detail="slow down")
            hits.append(now)

    def session_id(self, friend: dict[str, object]) -> str:
        return f"friend_{str(friend['rater_id']).removeprefix('friend_')}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"


def create_app(config: FriendServerConfig | None = None) -> FastAPI:
    config = config or FriendServerConfig()
    service = FriendService(config)
    app = FastAPI(title="Cloud friends", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def root() -> JSONResponse:
        return JSONResponse({"detail": "not found"}, status_code=404)

    @app.get("/f/{token}", response_class=HTMLResponse)
    def page(token: str) -> HTMLResponse:
        service.authenticate(token)
        return HTMLResponse(PAGE, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex"})

    @app.get("/api/{token}/next")
    def api_next(token: str, limit: int = 6, exclude: str = "") -> dict[str, object]:
        friend = service.authenticate(token)
        limit = max(1, min(int(limit), 24))
        ratings = service.ratings()
        rated = service.rated_ids(str(friend["rater_id"]), ratings)
        excluded = {item for item in exclude.split(",") if item}
        ids = next_items(service.queue(friend), rated, exclude=excluded, limit=limit)
        manifest = service.manifest().set_index("image_id")
        items = [{"image_id": image_id, "kind": str(manifest.loc[image_id].get("kind", "original")), "crop_class": str(manifest.loc[image_id].get("crop_class", "original"))} for image_id in ids if image_id in manifest.index]
        return {"display_name": friend["display_name"], "items": items, "progress": service.progress(friend, ratings)}

    @app.get("/api/{token}/image/{image_id}")
    def api_image(token: str, image_id: str) -> FileResponse:
        service.authenticate(token)
        if len(image_id) > 64 or not image_id.replace("_", "").isalnum():
            raise HTTPException(status_code=404, detail="not found")
        path = service.cached_image(image_id)
        if path is None:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/{token}/rate")
    def api_rate(token: str, body: RateBody) -> dict[str, object]:
        friend = service.authenticate(token)
        if service.image_row(body.image_id) is None:
            raise HTTPException(status_code=404, detail="unknown image")
        score = round(body.score * 2) / 2
        row = record_rating(config.ratings_dir, image_id=body.image_id, rater_id=str(friend["rater_id"]), score=float(score), rating_session_id=service.session_id(friend), note=body.note)
        return {"rating_id": row["rating_id"], "score": float(score), "progress": service.progress(friend)}

    @app.post("/api/{token}/undo")
    def api_undo(token: str, body: UndoBody) -> dict[str, object]:
        friend = service.authenticate(token)
        ratings = service.ratings()
        own = ratings[(ratings["rating_id"].astype(str) == body.rating_id) & (ratings["rater_id"].astype(str) == str(friend["rater_id"]))] if not ratings.empty else ratings
        if own.empty:
            raise HTTPException(status_code=404, detail="unknown rating")
        void_rating(config.ratings_dir, body.rating_id, reason="friend_undo")
        return {"voided": body.rating_id, "progress": service.progress(friend)}

    @app.get("/api/{token}/duel")
    def api_duel(token: str) -> dict[str, object]:
        friend = service.authenticate(token)
        rater_id = str(friend["rater_id"])
        ratings = service.ratings()
        own = ratings[ratings["rater_id"].astype(str) == rater_id] if not ratings.empty else ratings
        pairs = load_raw_pairwise_preferences(config.pairwise_dir)
        own_pairs = pairs[pairs["rater_id"].astype(str) == rater_id] if not pairs.empty else pairs
        n_duels = int(len(own_pairs))
        if own.empty:
            return {"pair": None, "n_duels": n_duels, "message": "Bewerte zuerst ein paar Bilder, dann gibt es Duelle zwischen deinen guten."}
        latest = own.sort_values("rating_timestamp").groupby("image_id")["raw_score_1_to_10"].last()
        queue_ids = set(service.queue(friend))
        pool = latest[(latest >= config.queue.duel_threshold) & latest.index.astype(str).isin(queue_ids)]
        if len(pool) < 2:
            return {"pair": None, "n_duels": n_duels, "message": f"Duelle starten, sobald du mindestens zwei Bilder mit {config.queue.duel_threshold:g} oder besser bewertet hast."}
        pool_frame = pd.DataFrame({"image_id": pool.index.astype(str), "score": pool.to_numpy(dtype=float)})
        pair = select_duel_pair(pool_frame, own_pairs, seed=int(time.time()) % 100000)
        if pair is None:
            return {"pair": None, "n_duels": n_duels, "message": "Alle Paare deiner guten Bilder sind durch. Bewerte weiter, dann gibt es neue."}
        return {"pair": [pair[0], pair[1]], "n_duels": n_duels}

    @app.post("/api/{token}/duel")
    def api_duel_post(token: str, body: DuelBody) -> dict[str, object]:
        friend = service.authenticate(token)
        if body.left == body.right or service.image_row(body.left) is None or service.image_row(body.right) is None:
            raise HTTPException(status_code=400, detail="invalid pair")
        if body.winner is not None and body.winner not in (body.left, body.right):
            raise HTTPException(status_code=400, detail="winner must be one of the pair")
        row = record_pairwise_preference(config.pairwise_dir, left_image_id=body.left, right_image_id=body.right, rater_id=str(friend["rater_id"]), winner=body.winner, tie_flag=body.winner is None, preference_strength=0.5 if body.winner is None else 1.0)
        return {"pair_id": row.get("pair_id"), "winner": body.winner}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    return app


def serve(config: FriendServerConfig | None = None, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(config), host=host, port=port, log_level="info")
