"""Recherche locale d'un visage dans les événements enregistrés.

Usage :
    python search.py chemin/vers/photo.jpg --top 5
"""

import argparse
from pathlib import Path

import cv2
import insightface
import numpy as np

from database import get_all_faces, init_db


def cosine_score(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def get_query_embedding(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Image introuvable ou illisible : {image_path}")

    app = insightface.app.FaceAnalysis(
        name="buffalo_l", providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0)
    faces = app.get(image)
    if not faces:
        raise ValueError("Aucun visage détecté dans l'image fournie.")

    # Le plus grand visage est généralement celui recherché.
    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    return face.embedding


def search(image_path, limit=5):
    query_embedding = get_query_embedding(image_path)
    results = []
    for record in get_all_faces():
        record["similarity"] = cosine_score(query_embedding, record["embedding"])
        results.append(record)
    return sorted(results, key=lambda item: item["similarity"], reverse=True)[:limit]


def main():
    parser = argparse.ArgumentParser(description="Recherche dans l'index local GODEYE")
    parser.add_argument("image", type=Path, help="photo contenant le visage à rechercher")
    parser.add_argument("--top", type=int, default=5, help="nombre maximal de résultats")
    args = parser.parse_args()

    init_db()
    results = search(args.image, max(1, args.top))
    if not results:
        print("Aucun visage enregistré dans la base.")
        return

    for index, item in enumerate(results, start=1):
        name = item["name"] or "Inconnu"
        print(
            f"{index}. {item['similarity'] * 100:.1f}% | {name} | "
            f"{item['timestamp']} | {item['camera_id']} | "
            f"{item['image_path']} | face_id={item['face_id']}"
        )


if __name__ == "__main__":
    main()
