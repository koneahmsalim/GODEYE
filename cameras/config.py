"""Chargement des sources vidéo configurées localement."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    source: int | str


def load_cameras(config_path="config/cameras.json"):
    """Retourne les caméras activées définies dans le fichier JSON."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    cameras = []
    for item in payload.get("cameras", []):
        if not item.get("enabled", True):
            continue
        source = item["source"]
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cameras.append(
            CameraConfig(
                camera_id=item["id"],
                name=item.get("name", item["id"]),
                source=source,
            )
        )

    if not cameras:
        raise ValueError("Aucune caméra activée dans config/cameras.json.")
    return cameras
