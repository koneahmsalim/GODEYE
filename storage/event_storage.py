"""Sauvegarde locale des crops de visage et de leur contexte."""

from datetime import datetime
from pathlib import Path

import cv2


class EventStorage:
    def __init__(self, base_dir="faces/captures"):
        self.base_dir = Path(base_dir)
        self.faces_dir = self.base_dir / "faces"
        self.frames_dir = self.base_dir / "frames"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def save(self, face_crop, frame, camera_id, track_id):
        """Enregistre une image du visage et le frame complet associé."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_camera_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(camera_id)
        )
        stem = f"{safe_camera_id}_{timestamp}_track_{track_id}"
        face_path = self.faces_dir / f"face_{stem}.jpg"
        frame_path = self.frames_dir / f"frame_{stem}.jpg"

        face_saved = cv2.imwrite(str(face_path), face_crop)
        frame_saved = cv2.imwrite(str(frame_path), frame)
        if not face_saved or not frame_saved:
            return None, None
        return face_path, frame_path
