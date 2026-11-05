from src.pipeline.nli_pipeline import run_nli_pipeline
from src.pipeline.hhem_pipeline import run_hhem_pipeline

df_hhem_scored, df_nli_input = run_hhem_pipeline(batch_size = 32)
df_nli_scored, df_stage2_input = run_nli_pipeline(df_nli_input, batch_size = 32)