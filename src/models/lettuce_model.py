from lettucedetect.models.inference import HallucinationDetector

MODEL_NAME = "KRLabsOrg/lettucedect-base-modernbert-en-v1"


def load_lettuce_model():
    detector = HallucinationDetector(
        method="transformer",
        model_path=MODEL_NAME
    )
    return detector


def get_lettuce_spans(detector, context, question, response):
    spans = detector.predict(
        context=[context],
        question=question,
        answer=response,
        output_format="spans"
    )

    return spans