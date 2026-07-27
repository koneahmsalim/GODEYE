# ROADMAP

## Version 0.1

- [x] Détection des personnes
- [x] Détection des visages
- [x] Tracking DeepSORT

## Version 0.2

- [x] Base SQLite
- [x] Sauvegarde des embeddings
- [x] Sauvegarde des images de visage
- [ ] Capture du contexte complet
- [ ] Déduplication configurable par similarité et délai

## Version 0.3 — Recherche par photo

```text
python search.py image.jpg
```

- [x] Recherche locale par similarité
- [x] Retour du score, de la date, de la caméra et de l'image
- [ ] Top 20 au-dessus d'un seuil configurable
- [ ] Regroupement chronologique des apparitions

## Version 0.4 — Sources vidéo

- [ ] Webcam
- [ ] RTSP
- [ ] Fichiers vidéo
- [ ] Configuration multi-caméras

## Version 0.5 — Interface web

- [ ] Dashboard local
- [ ] Historique
- [ ] Recherche
- [ ] Filtres

## Version 0.6 — Événements temps réel

- [ ] Alertes locales configurables
- [ ] Popup et signal sonore optionnels
- [ ] Journal des alertes

## Version 0.7 — Recherche à grande échelle

- [ ] Index vectoriel FAISS
- [ ] Base de plusieurs dizaines de milliers d'événements
- [ ] Mesures de performances

## Version 1.0 — Version stable

- [ ] Historique et recherche
- [ ] API locale
- [ ] Dashboard
- [ ] Sources vidéo multiples
- [ ] Docker et déploiement Linux
