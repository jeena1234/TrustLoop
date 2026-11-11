import pandas as pd
from src.models.stage2_llm_model import verify_with_llm ,load_stage2_client
import time


def run_stage2_pipeline(df_input,output_path="data/results/stage2/stage2_results.csv"):

    df_stage2 = df_input.copy().reset_index(drop=True)

    client = load_stage2_client()

    verdicts = []
    reasons = []
    latencies = [] 
    for i , row in df_stage2.iterrows():
        context = row["passages"]
        question = row["question"]
        generated_answer = row["response"]

        start_time =  time.perf_counter()
        result = verify_with_llm(client, context, question, generated_answer)
        latency = time.perf_counter() - start_time

        verdicts.append(result["verdict"])
        reasons.append(result["reason"])
        latencies.append(latency)

        print(
            f"Processed {i + 1}/{len(df_stage2)} | "
            f"{result['verdict']} | "
            f"{latency:.2f}s"
        )

    df_stage2["stage2_verdict"] = verdicts
    df_stage2["stage2_reasons"] = reasons
    df_stage2["stage2_latency"] = latencies

    df_stage2["stage2_label"] = df_stage2["stage2_verdict"].apply(lambda x: "GROUNDED" if x== "ANSWER_VERIFIED" else "HALLUCINATED")

    df_stage2.to_csv(
    output_path,
    index=False
    )

    return df_stage2    