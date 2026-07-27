"""Moteur local de détection faciale InsightFace."""

import insightface


class FaceEngine:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

    def detect(self, image):
        return self.app.get(image)
