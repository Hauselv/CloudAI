"""Friend registry: one personal token per friend, mapped to a human rater id.

The token is the whole access control of the friend server (plus whatever sits in front
of the tunnel, e.g. Cloudflare Access). Tokens live in ``data/raw/metadata/friends.csv``
(private, git-ignored) and can be revoked at any time; the ratings a friend recorded stay
in their own append-only rater file.
"""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cloud_aesthetics.judge.store import RATERS_PATH, register_rater
from cloud_aesthetics.settings import ensure_parent, resolve_path

FRIENDS_PATH = "data/raw/metadata/friends.csv"
FRIEND_COLUMNS = ["token", "rater_id", "display_name", "created_at", "active", "note"]


def load_friends(path_like: str | Path = FRIENDS_PATH) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        return pd.DataFrame(columns=FRIEND_COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in FRIEND_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["active"] = frame["active"].astype(str).str.lower().isin(["true", "1", "yes"])
    return frame[FRIEND_COLUMNS]


def _save(frame: pd.DataFrame, path_like: str | Path) -> None:
    frame = frame.copy()
    frame["active"] = frame["active"].map(lambda value: "true" if bool(value) else "false")
    frame.to_csv(ensure_parent(path_like), index=False)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")).strip("_")
    return slug or "friend"


def add_friend(display_name: str, *, path_like: str | Path = FRIENDS_PATH, raters_path: str | Path = RATERS_PATH, note: str = "") -> dict[str, object]:
    """Create a token and rater id for a friend; registers the rater as human."""
    frame = load_friends(path_like)
    base = f"friend_{slugify(display_name)}"
    rater_id = base
    counter = 2
    while (frame["rater_id"] == rater_id).any():
        rater_id = f"{base}{counter}"
        counter += 1
    token = secrets.token_urlsafe(9)
    while (frame["token"] == token).any():
        token = secrets.token_urlsafe(9)
    row = {"token": token, "rater_id": rater_id, "display_name": display_name.strip(), "created_at": datetime.now(timezone.utc).isoformat(), "active": True, "note": note}
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    _save(frame, path_like)
    register_rater(rater_id, "human", note=f"friend link for {display_name.strip()}", path_like=raters_path)
    return row


def revoke_friend(token_or_rater: str, *, path_like: str | Path = FRIENDS_PATH) -> bool:
    frame = load_friends(path_like)
    mask = (frame["token"] == token_or_rater) | (frame["rater_id"] == token_or_rater)
    if not mask.any():
        return False
    frame.loc[mask, "active"] = False
    _save(frame, path_like)
    return True


def reactivate_friend(token_or_rater: str, *, path_like: str | Path = FRIENDS_PATH) -> bool:
    frame = load_friends(path_like)
    mask = (frame["token"] == token_or_rater) | (frame["rater_id"] == token_or_rater)
    if not mask.any():
        return False
    frame.loc[mask, "active"] = True
    _save(frame, path_like)
    return True


def resolve_token(token: str, *, path_like: str | Path = FRIENDS_PATH) -> dict[str, object] | None:
    """Active friend for a token, or None. Tokens are compared exactly; no prefix matching."""
    if not token or len(token) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        return None
    frame = load_friends(path_like)
    hit = frame[(frame["token"] == token) & frame["active"]]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {"token": str(row["token"]), "rater_id": str(row["rater_id"]), "display_name": str(row["display_name"]) or str(row["rater_id"])}


def friend_link(base_url: str, token: str) -> str:
    return f"{base_url.rstrip('/')}/f/{token}"
