import pandas as pd

from src.models.hhem_model import load_hhem_model, get_hhem_score



def run_hhem_pipeline(df_input,batch_size = 32):
  
  ####  Load HHEM Model
  model = load_hhem_model()

  #### create batches for full data processing

  scores = []
  for start in range(0, len(df_input), batch_size):
      end = start + batch_size
      batch = df_input.iloc[start:end]
      score = get_hhem_score(
                  model, 
                  batch["passages"].tolist(), 
                  batch["response"].tolist()
                  )
      scores.extend(score)


  ### Add scores to the new datafame 
  df_hhem_scored = df_input.copy()
  df_hhem_scored["hhem_score"] = scores

  df_hhem_scored["hhem_label"] = df_hhem_scored["hhem_score"].apply(lambda x: 
                                                                    "GROUNDED" if x >= 0.90 
                                                                    else "GREY ZONE" if x >= 0.40 
                                                                    else "HALLUCINATED")


  ## Just for analysis export to csv
  df_hhem_scored.to_csv(
    "/content/drive/MyDrive/TrustLoop/results/hhem_scored_stage1_results.csv",
    index=False
  )

  return df_hhem_scored