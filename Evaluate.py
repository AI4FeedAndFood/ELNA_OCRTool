import os
import pandas as pd
import json 
from datetime import date
import time
today = str(date.today().strftime("%b-%d-%Y"))
import numpy as np

from LaunchTool import TextCVTool
from JaroDistance import jaro_distance

OCR_HELPER_JSON_PATH  = r"CONFIG\\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH)) 

def _condition_fuction(col, proposition, data):
    if data in ["None", "Nan", "NaN", "nan", "NAN", np.nan]:
        return None

    proposition = str(proposition).lower()
    data = str(data).lower()
    if proposition == data:
        return 1
    if proposition in ["", " ", []]:
        return 0
    else :
        return -1
        
def eval_text_extraction(path_to_eval, eval_path = r"C:\Users\CF6P\Desktop\cv_text\Evaluate",
                         result_name = "0_results", config= ["paddle", "structure", "en", "no_bin"]):
    
    result_excel_path = os.path.join(eval_path, result_name+".xlsx")
    data_col = ["root_path", "page_name"]
    zones_col = list(OCR_HELPER["hand"].keys())+["format"]
    
    if os.path.exists(result_excel_path):
        eval_df = pd.read_excel(result_excel_path, sheet_name="results")
        proba_df= pd.read_excel(result_excel_path, sheet_name="proba")
    else :
        eval_df = pd.DataFrame(columns=data_col+zones_col) # All detected text for all zones
        proba_df = pd.DataFrame(columns=data_col+zones_col)
    res_image_name, res_dict_per_image, res_image = TextCVTool(path_to_eval, config=config)
    for image, zone_dict in res_dict_per_image["RESPONSE"].items():
        row = [path_to_eval, image]
        for _, landmark_text_dict in zone_dict.items():
            row.append(landmark_text_dict["sequence"])
        if len(row) < len(zones_col):
            row.insert(5, [])
            row.insert(9, [])
            row.insert(10, [])
        row.append(landmark_text_dict["format"])
        eval_df.loc[len(eval_df)] = row
    eval_df.to_excel(result_excel_path, sheet_name="results", index=False)

    for image, zone_dict in res_dict_per_image["RESPONSE"].items():
        row = [path_to_eval, image]
        for _, landmark_text_dict in zone_dict.items():
            row.append(int(landmark_text_dict["confidence"]*100))
        if len(row) < len(zones_col):
            row.insert(5, [])
            row.insert(9, [])
            row.insert(10, [])
        row.append(landmark_text_dict["format"])
        proba_df.loc[len(proba_df)] = row
    
    with pd.ExcelWriter(result_excel_path, mode = 'a') as writer:
        proba_df.to_excel(writer, sheet_name="proba", index=False)
    

if __name__ == "__main__":
    
    result_name = "prod_V2.4_Bin"
    eval_path = r"C:\Users\CF6P\Desktop\ELPV\Eval"
    start = time.time()
    eval_text_extraction(eval_path, eval_path=eval_path, result_name=result_name)
    