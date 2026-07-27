import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDINGS_DIR = "embeddings"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)


def save_face(name, embedding):

    path = os.path.join(
        EMBEDDINGS_DIR,
        f"{name}.npy"
    )

    np.save(path, embedding)

    print(f"Visage enregistré : {name}")



def recognize_face(embedding):

    best_name = "Inconnu"
    best_score = 0


    for file in os.listdir(EMBEDDINGS_DIR):

        if not file.endswith(".npy"):
            continue


        known = np.load(
            os.path.join(
                EMBEDDINGS_DIR,
                file
            )
        )


        score = cosine_similarity(
            [embedding],
            [known]
        )[0][0]


        if score > best_score:
            best_score = score
            best_name = file.replace(
                ".npy",
                ""
            )


    if best_score > 0.55:
        return best_name, best_score


    return "Inconnu", best_score