import cv2
import os
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import insightface
from face_recognition import save_face, recognize_face
from database import init_db, save_face_event
MODEL_PATH = "models/yolov8s.pt"
CAMERA_ID = "camera_interieure_0"
CAPTURES_DIR = Path("faces") / "captures"


init_db()
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


# YOLO
model = YOLO(MODEL_PATH)


# DeepSort
tracker = DeepSort(
    max_age=30,
    n_init=3,
)


# Face recognition
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0)


cap = cv2.VideoCapture(0)
saved_track_ids = set()


while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model(frame)


    detections = []


    for r in results:

        for box in r.boxes:

            class_id = int(box.cls[0])

            # COCO classe 0 = person : ignorer chaises, voitures, etc.
            if class_id != 0:
                continue


            x1,y1,x2,y2 = map(
                int,
                box.xyxy[0]
            )


            confidence=float(box.conf[0])


            detections.append(
                (
                    [x1,y1,x2-x1,y2-y1],
                    confidence,
                    "person"
                )
            )


    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )


    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, r, b = track.to_ltrb()

        cv2.rectangle(
            frame,
            (int(l), int(t)),
            (int(r), int(b)),
            (0, 255, 0),
            2
        )

        name = "Inconnu"
        score = 0

        x1, y1, x2, y2 = map(
            int,
            track.to_ltrb()
        )

        # protéger les coordonnées
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        # récupérer seulement la partie haute du corps
        face_region_y2 = y1 + int((y2 - y1) * 0.45)

        person_crop = frame[
            y1:face_region_y2,
            x1:x2
        ]

        # vérifier que l'image existe
        if person_crop.size > 0:
            faces = face_app.get(person_crop)

            if faces:
                face = faces[0]
                name, score = recognize_face(
                    face.embedding
                )

                # Une seule capture par piste pendant cette session caméra.
                if track_id not in saved_track_ids:
                    fx1, fy1, fx2, fy2 = map(int, face.bbox)
                    ph, pw = person_crop.shape[:2]
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2, fy2 = min(pw, fx2), min(ph, fy2)
                    face_crop = person_crop[fy1:fy2, fx1:fx2]

                    if face_crop.size > 0:
                        captured_at = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"face_{captured_at}_track_{track_id}.jpg"
                        image_path = CAPTURES_DIR / filename

                        if cv2.imwrite(str(image_path), face_crop):
                            stored_name = None if name == "Inconnu" else name
                            face_id, _ = save_face_event(
                                embedding=face.embedding,
                                image_path=image_path,
                                camera_id=CAMERA_ID,
                                track_id=track_id,
                                confidence=getattr(face, "det_score", 0.0),
                                name=stored_name,
                            )
                            saved_track_ids.add(track_id)
                            print(f"Visage indexé : {face_id} ({name})")

        cv2.putText(
            frame,
            f"ID {track_id} - {name}",
            (int(l), int(t) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "GODEYE",
        frame
    )

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
