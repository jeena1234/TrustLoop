import pandas as pd
from src.models.nli_model import load_nli_model, load_nli_tokenizer, get_nli_score
import torch

def run_nli_pipeline(df_nli_input, batch_size = 32):
    # Load NLI model and tokenizer

    tokenizer = load_nli_tokenizer()
    model = load_nli_model()

    # Create batches for full data processing
    
    output_probs_list = []
    for start in range(0, len(df_nli_input), batch_size):
        end = start + batch_size
        batch = df_nli_input.iloc[start:end]
        context = batch["passages"].tolist()
        response = batch["response"].tolist()
        output_probs = get_nli_score(model, tokenizer, context, response)
        output_probs_list.extend(output_probs.tolist())

    final_predictions = torch.argmax(torch.tensor(output_probs_list), dim=1)
    final_labels = [model.config.id2label[int(i)] for i in final_predictions]

    ## prepare Stage 2 input
    df_nli_scored = df_nli_input.copy()
    df_nli_scored["NLI_label"] = final_labels
    df_stage2_input = df_nli_scored[df_nli_scored["NLI_label"].isin(["neutral", "contradiction"])].copy()

    df_nli_scored.to_csv(
        "/content/drive/MyDrive/TrustLoop/results/nli_scored_results.csv",
        index=False
      )

    df_stage2_input.to_csv(
        "/content/drive/MyDrive/TrustLoop/results/stage2_input_from_nli.csv",
        index=False
    )
    return df_nli_scored, df_stage2_input