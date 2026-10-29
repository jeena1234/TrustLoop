from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
def prepare_master_data():
    source_path = PROJECT_ROOT / "data" / "ragtruth" / "raw" / "source_info.jsonl"
    response_path = PROJECT_ROOT / "data" / "ragtruth" / "raw" / "response.jsonl"

    ## Read source_info.jsonl
    df_source_info = pd.read_json(source_path, lines=True)

    ## Read response.jsonl
    df_response_info = pd.read_json(response_path, lines=True)

    ## filter only the QA datset form source_info.jsonl

    df_source_info = df_source_info[df_source_info["task_type"] == "QA"]
    # print(df_source_info.shape)

    ## preserve source_id
    df_source_id = df_source_info["source_id"]

    ## Get passage and Question form source_info
    source_info = df_source_info["source_info"].to_dict()
    df_source = pd.DataFrame(source_info)
    df_source = df_source.transpose()
    df_source["source_id"] = df_source_id
    df_source.reset_index(drop=True, inplace= True)

    # print(df_source.head(2))
    # print(df_source.shape)
    ## modify df_response_info 

    df_response = df_response_info.drop(columns=["id"]).copy()
    df_response["hallucination_label"] = df_response["labels"].apply(lambda x: "GROUNDED" if len(x) == 0 else "HALLUCINATED")
    df_response = df_response.rename(columns={"labels": "hallucination_span"})
    # print(df_response.head(2))
    # print(df_response.shape)

    ## Create the sub final dataframe 

    df_subfinal = df_source.merge(df_response, on = "source_id", how="inner")
    # print(df_subfinal.shape)
    # print(df_subfinal.columns)
    # print(df_subfinal.head(2))
    
    return df_subfinal

def prepare_hhem_data():
    df_final = prepare_master_data()
    df_hhem = df_final.drop(columns=["model","temperature","hallucination_span","quality"])
    df_hhem_validate = df_hhem[df_hhem["split"]== "train"].copy()
    df_hhem_test = df_hhem[df_hhem["split"]== "test"].copy()
    return df_hhem_validate, df_hhem_test

