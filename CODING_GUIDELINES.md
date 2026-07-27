# Instructions pour les agents IA

Ce projet est développé avec l'aide de Codex et d'autres assistants IA.

## Règles

- Ne jamais casser une fonctionnalité existante.
- Privilégier les bibliothèques open-source.
- Ne pas utiliser TensorFlow lorsqu'une alternative ONNX est disponible.
- Préférer InsightFace pour les embeddings et YOLOv8 pour la détection.
- Préférer SQLite pendant le développement.
- Écrire du code modulaire, testable et compatible Windows.
- Commenter les fonctions importantes et documenter tout changement d'architecture dans `PROJECT.md` ou `ROADMAP.md`.
- Ne jamais ajouter de service cloud, d'API propriétaire ou d'envoi de données sans une demande explicite de l'utilisateur.
- Limiter les traitements aux caméras, vidéos et jeux de données que l'utilisateur est autorisé à utiliser.

## Objectif technique

Construire progressivement un système local de vision par ordinateur inspiré de GodEye (*Fast & Furious*), mais réaliste, légal, modulaire et open-source.

Le logiciel doit pouvoir évoluer sans réécriture du cœur : nouvelles caméras, modèles, API REST, interface web, stockage et index de recherche.
