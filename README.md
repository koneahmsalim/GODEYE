# GODEYE

Moteur local d'indexation d'événements vidéo pour des sources autorisées. Le projet détecte les personnes, suit les pistes, extrait les visages et conserve localement les événements associés.

Le contexte et les limites sont détaillés dans [PROJECT.md](PROJECT.md), les prochaines étapes dans [ROADMAP.md](ROADMAP.md), et les règles de contribution pour les assistants dans [CODING_GUIDELINES.md](CODING_GUIDELINES.md).

## Structure

```text
GODEYE/
├── api/            # future API locale
├── cameras/        # futures sources vidéo
├── config/         # futures configurations
├── database/       # SQLite et accès aux événements
├── detection/      # futurs composants YOLO
├── embeddings/     # embeddings de profils enregistrés
├── faces/          # captures de visages et de contexte
├── models/         # modèles locaux
├── recognition/    # futurs composants InsightFace
├── storage/        # futur stockage et archivage
├── tests/          # tests automatisés
├── main.py         # flux webcam actuel
├── search.py       # recherche locale par image
├── PROJECT.md
└── ROADMAP.md
```

`search.py` reste un fichier pour le moment ; il pourra devenir un paquet `search/` lors d'une refactorisation ultérieure.

Les caméras sont définies dans `config/cameras.json`. La valeur `source` peut être un index de webcam (`0`), un chemin de fichier vidéo ou l'URL d'un flux RTSP autorisé.

## Installation

Depuis le dossier `GODEYE` :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place le modèle YOLO dans `models/yolov8s.pt`, puis lance :

```powershell
python main.py
```

Appuie sur `Échap` pour fermer la fenêtre caméra.

## Recherche locale

```powershell
python search.py chemin\vers\photo.jpg --top 20 --threshold 0.65
```

Les résultats affichent le score de similarité, la date, l'identifiant de caméra et le chemin de la capture associée.

## Données locales

- La base SQLite est `database/god_eye.db`.
- Les captures de visage et leurs frames complets sont créées dans `faces/captures/`.
- Les modèles et données produites localement ne sont pas inclus dans Git par défaut.
