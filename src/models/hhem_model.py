from transformers import AutoModelForSequenceClassification

MODEL_NAME = "vectara/hallucination_evaluation_model"
def load_hhem_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    return model


def get_hhem_score(model, context, response):
    pairs = list(zip(context, response))

    scores = model.predict(pairs)

    return [float(score) for score in scores]