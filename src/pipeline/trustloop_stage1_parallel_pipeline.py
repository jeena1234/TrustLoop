from src.pipeline.hhem_pipeline import run_hhem_pipeline
from src.pipeline.lettuce_pipeline import run_lettuce_pipeline

def run_stage1_parallel_pipeline(df_input):
    # Run HHEM pipeline
    df_hhem = run_hhem_pipeline(df_input)

    # Run LettuceDetect pipeline
    df_lettuce = run_lettuce_pipeline(df_input)


    df_stage1 = df_hhem.copy()

    df_stage1["lettuce_spans"] = df_lettuce["lettuce_spans"]
    df_stage1["lettuce_span_count"] = df_lettuce["lettuce_span_count"]
    df_stage1["lettuce_label"] = df_lettuce["lettuce_label"]

    # Combine decisions
    def assign_stage1_route(row):

        if (
            row["hhem_label"] == "GROUNDED"
            and row["lettuce_label"] == "GROUNDED"
        ):
            return "FINAL_GROUNDED"

        if (
            row["hhem_label"] == "HALLUCINATED"
            and row["lettuce_label"] == "HALLUCINATED"
        ):
            return "FINAL_HALLUCINATED"

        return "SEND_TO_STAGE2"

    df_stage1["stage1_route"] = df_stage1.apply(
        assign_stage1_route,
        axis=1
    )

    df_stage2_input = df_stage1[
    df_stage1["stage1_route"] == "SEND_TO_STAGE2"
    ].copy()

     # Save outputs
    df_stage1.to_csv(
        "/content/drive/MyDrive/TrustLoop/results/parallel_stage1_results.csv",
        index=False
    )

    df_stage2_input.to_csv(
        "/content/drive/MyDrive/TrustLoop/results/stage2_input.csv",
        index=False
    )

    return df_stage1, df_stage2_input