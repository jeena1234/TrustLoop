import pandas as pd

from src.models.hhem_model import load_hhem_model, get_hhem_score
from src.data.prepare_data import prepare_hhem_data
BATCH_SIZE = 32

# Load prepared dataset
df_hhem_validate, df_hhem_test = prepare_hhem_data()


#####  Load HHEM Model
model = load_hhem_model()

# context_validate = df_hhem_validate.iloc[:100]["passages"]
# response_validate = df_hhem_validate.iloc[:100]["response"]



#### create batches for full data processing

scores = []
for start in range(0, len(df_hhem_validate), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = df_hhem_validate.iloc[start:end]
    score = get_hhem_score(
                model, 
                batch["passages"].tolist(), 
                batch["response"].tolist()
                )
    scores.extend(score)


### Add scores to the new datafame 
df_hhem_scored = df_hhem_validate.copy()
df_hhem_scored["hhem_score"] = scores



## Test data
# context_test = df_hhem_test.iloc[:100]["passages"]
# response_test = df_hhem_test.iloc[:100]["response"]






# get scores for 
# scores = get_hhem_score(model, context_validate, response_validate)

# df_validate_sample = df_hhem_validate.iloc[:100].copy()
# df_validate_sample["hhem_score"] = scores

# print(
#     df_validate_sample[
#         ["source_id", "hallucination_label", "hhem_score"]
#     ].head(20)
# )

