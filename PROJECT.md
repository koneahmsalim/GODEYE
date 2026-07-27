# GODEYE

## Vision

GODEYE est un projet personnel de Computer Vision inspiré du système fictif « God's Eye » de *Fast & Furious*.

Le but n'est pas de reproduire les capacités fictives du film. L'objectif est de construire une version réaliste, entièrement locale, utilisant des modèles open-source et des données légalement accessibles.

Aucune donnée n'est envoyée sur Internet : tout fonctionne sur la machine locale.

## Objectifs

Le logiciel doit pouvoir :

- détecter des personnes ;
- suivre une personne entre plusieurs images ;
- détecter son visage ;
- générer un embedding du visage ;
- enregistrer automatiquement ce visage dans une base locale ;
- retrouver des événements associés à partir d'une photo fournie par un utilisateur autorisé.

Le système fonctionne comme une caméra intelligente. Les visages détectés peuvent être indexés localement, y compris lorsqu'ils ne sont pas nommés.

## Cas d'utilisation

Exemple : une caméra autorisée filme l'entrée d'une maison pendant plusieurs mois. Les événements sont indexés localement. Plus tard, une photo est fournie au logiciel, qui peut retourner les événements correspondants : date, heure, caméra, capture et score de similarité.

## Technologies

- Python
- OpenCV
- YOLOv8
- DeepSORT
- InsightFace
- SQLite
- ONNX Runtime
- NumPy

## Contraintes

Le projet doit rester :

- 100 % local ;
- légal et limité aux sources autorisées ;
- open-source ;
- sans cloud ;
- sans API propriétaire ;
- sans reconnaissance biométrique en ligne.

Les données utilisées doivent provenir de caméras appartenant à l'utilisateur, de jeux de données publics ou d'images obtenues avec autorisation.
