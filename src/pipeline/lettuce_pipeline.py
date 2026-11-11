import pandas as pd

from src.models.lettuce_model import (
    load_lettuce_model,
    get_lettuce_spans
)


def run_lettuce_pipeline(df_input):

    # Load LettuceDetect model
    detector = load_lettuce_model()

    all_spans = []

    # Run span-level hallucination detection
    for _, row in df_input.iterrows():

        spans = get_lettuce_spans(
            detector,
            context=row["passages"],
            question=row["question"],
            response=row["response"]
        )

        all_spans.append(spans)

    # Add LettuceDetect results
    df_lettuce_scored = df_input.copy()
    df_lettuce_scored["lettuce_spans"] = all_spans

    df_lettuce_scored["lettuce_span_count"] = (
        df_lettuce_scored["lettuce_spans"].apply(len)
    )

    df_lettuce_scored["lettuce_label"] = (
        df_lettuce_scored["lettuce_span_count"].apply(
            lambda x: "HALLUCINATED" if x > 0 else "GROUNDED"
        )
    )
    df_lettuce_scored.to_csv(
        "/content/drive/MyDrive/TrustLoop/results/lettuce_scored_stage1_results.csv",
        index=False
      )

    return df_lettuce_scored