import cv2
import insightface

from face_recognition import save_face


name = input("Nom de la personne : ")


app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    faces = app.get(frame)


    for face in faces:

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            "Appuyer S pour enregistrer",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    cv2.imshow(
        "GODEYE - Enregistrement",
        frame
    )


    key = cv2.waitKey(1) & 0xff


    if key == ord("s"):

        if faces:

            save_face(
                name,
                faces[0].embedding
            )

            print("Visage sauvegarde.")

            break

        else:
            print("Aucun visage detecte")


    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()