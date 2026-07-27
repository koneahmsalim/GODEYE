"""Point d'entrée GODEYE pour les caméras configurées localement."""

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

from cameras import load_cameras
from database import init_db, save_face_event
from detection import PersonDetector
from face_recognition import recognize_face
from recognition import FaceEngine
from storage import EventStorage


def face_crop_from_detection(person_crop, face):
    """Extrait un visage détecté dans le crop haut du corps."""
    x1, y1, x2, y2 = map(int, face.bbox)
    height, width = person_crop.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return person_crop[y1:y2, x1:x2]


def open_cameras(configurations):
    """Ouvre les caméras et crée un tracker indépendant par source."""
    runtimes = []
    for camera in configurations:
        capture = cv2.VideoCapture(camera.source)
        if not capture.isOpened():
            print(f"Impossible d'ouvrir {camera.name} ({camera.source}).")
            capture.release()
            continue
        runtimes.append(
            {
                "config": camera,
                "capture": capture,
                "tracker": DeepSort(max_age=30, n_init=3),
                "saved_track_ids": set(),
            }
        )
    return runtimes


def process_frame(runtime, frame, detector, face_engine, storage):
    camera = runtime["config"]
    tracks = runtime["tracker"].update_tracks(detector.detect(frame), frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        left, top, right, bottom = map(int, track.to_ltrb())
        height, width = frame.shape[:2]
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(width, right), min(height, bottom)
        name, score = "Inconnu", 0.0

        face_region_y2 = y1 + int((y2 - y1) * 0.45)
        person_crop = frame[y1:face_region_y2, x1:x2]
        if person_crop.size:
            faces = face_engine.detect(person_crop)
            if faces:
                face = faces[0]
                name, score = recognize_face(face.embedding)

                if track_id not in runtime["saved_track_ids"]:
                    face_crop = face_crop_from_detection(person_crop, face)
                    if face_crop.size:
                        face_path, frame_path = storage.save(
                            face_crop, frame, camera.camera_id, track_id
                        )
                        if face_path:
                            stored_name = None if name == "Inconnu" else name
                            face_id, _ = save_face_event(
                                embedding=face.embedding,
                                image_path=face_path,
                                frame_path=frame_path,
                                camera_id=camera.camera_id,
                                track_id=track_id,
                                confidence=getattr(face, "det_score", 0.0),
                                name=stored_name,
                            )
                            runtime["saved_track_ids"].add(track_id)
                            print(f"Visage indexé : {face_id} ({camera.camera_id}, {name})")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id} - {name}",
            (left, max(25, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


def main():
    init_db()
    detector = PersonDetector()
    face_engine = FaceEngine()
    storage = EventStorage()
    runtimes = open_cameras(load_cameras())
    if not runtimes:
        raise RuntimeError("Aucune caméra configurée n'a pu être ouverte.")

    try:
        while runtimes:
            active = []
            for runtime in runtimes:
                success, frame = runtime["capture"].read()
                if not success:
                    print(f"Flux arrêté : {runtime['config'].name}")
                    runtime["capture"].release()
                    continue

                process_frame(runtime, frame, detector, face_engine, storage)
                cv2.imshow(f"GODEYE - {runtime['config'].name}", frame)
                active.append(runtime)

            runtimes = active
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        for runtime in runtimes:
            runtime["capture"].release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
