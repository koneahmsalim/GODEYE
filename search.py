"""Recherche locale d'événements à partir d'une image.

Usage:
    python search.py chemin/vers/photo.jpg --top 20 --threshold 0.65
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

    return max(
        faces,
        key=lambda item: (item.bbox[2] - item.bbox[0])
        * (item.bbox[3] - item.bbox[1]),
    ).embedding


def search(image_path, limit=20, threshold=0.65):
    """Sélectionne les meilleures correspondances puis les trie par passage."""
    query_embedding = get_query_embedding(image_path)
    matches = []
    for record in get_all_faces():
        record["similarity"] = cosine_score(query_embedding, record["embedding"])
        if record["similarity"] >= threshold:
            matches.append(record)

    best_matches = sorted(
        matches, key=lambda item: item["similarity"], reverse=True
    )[:limit]
    return sorted(best_matches, key=lambda item: item["timestamp"])


def main():
    parser = argparse.ArgumentParser(description="Recherche dans l'index local GODEYE")
    parser.add_argument("image", type=Path, help="photo contenant le visage à rechercher")
    parser.add_argument("--top", type=int, default=20, help="nombre maximal de résultats")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="similarité minimale entre 0 et 1 (défaut : 0.65)",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold doit être compris entre 0 et 1")

    init_db()
    results = search(args.image, max(1, args.top), args.threshold)
    if not results:
        print("Aucune correspondance au-dessus du seuil.")
        return

    for index, item in enumerate(results, start=1):
        name = item["name"] or "Inconnu"
        print(
            f"{index}. {item['similarity'] * 100:.1f}% | {name} | "
            f"{item['timestamp']} | {item['camera_id']} | "
            f"visage={item['image_path']} | frame={item['frame_path']} | "
            f"face_id={item['face_id']}"
        )


if __name__ == "__main__":
    main()
