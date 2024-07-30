import matplotlib.pyplot as plt
import json
import os
import numpy as np
import cv2
import pandas as pd

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

whitelist =  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz(),:.-/°&=àéçëôùê''"
LANG = 'eng+eng2'
TESSCONFIG = [1, 6, whitelist, LANG]

from ProcessPDF import PDF_to_images

def getAllImages(path):
    def _get_all_images(path, extList=[".tiff", ".tif", ".png"]):
        docs = os.listdir(path)
        pdf_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() == ".pdf"]
        image_in_folder = [file for file in docs if os.path.splitext(file)[1].lower() in extList]
        return pdf_in_folder, image_in_folder
    
    pdf_in_folder, image_in_folder =  _get_all_images(path) # Return pathes
    res_path = os.path.join(path, "RES")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    scan_dict = {}

    for pdf in sorted(pdf_in_folder):
        new_images = PDF_to_images(os.path.join(path, pdf))
        pdf_dict = {}
        for i, image in enumerate(new_images):
            pdf_dict[f"image_{i}"] = image
        scan_dict[os.path.splitext(pdf)[0]] = pdf_dict
        
    
    for image in image_in_folder:
        new_image = np.array(cv2.imread(os.path.join(path,image)))
        scan_dict[os.path.splitext(new_images)[0]] = {"image_0" : new_image}

    return scan_dict

def TextCVTool(path, model, config = ["paddle", "OCR", "en"]):
    """The final tool to use with the GUI

    Args:
        path (path): a folder path
        custom_config (list, optional): tesseract config [oem, psm, whitelist, datatrain]
    """
    pdfs_res_dict = {
        "PARAMETRE" : {
            "model" : model,
            "ocr" : config
        }
    }
    scan_dict = getAllImages(path)

    if model == 'Nutricia':
        from Model_Nutricia import main
    
    if model == "RoyalCanin":
        from Model_RoyalCanin import main
        
    pdfs_res_dict["RESPONSE"] = main(scan_dict)

    return  scan_dict, pdfs_res_dict

def save_as_excel(path, pdfs_res_dict):
    def_col = ["pdf", "image", "sample"]
    field_col = []

    df = []
    for pdf, sample_dict in pdfs_res_dict["RESPONSE"].items():
        for sample, im_ext_dict in sample_dict.items():
            extract_dict = im_ext_dict["EXTRACTION"]

            if field_col == []:
                field_col = list(extract_dict.keys())

            row = [pdf, im_ext_dict["IMAGE"], sample] + [extract_dict[k]["sequence"] for k in field_col]
            df.append(row)
    
    df = pd.DataFrame(df, columns=def_col+field_col)
    df.to_excel(os.path.join(path, "res.xlsx"))

if __name__ == "__main__":
    
    import time
    start = time.time()
    path = r"C:\Users\CF6P\Desktop\ELNA\Data_ELNA\RC\test"
    scan_dict, pdfs_res_dict =TextCVTool(path, "RoyalCanin")
    save_as_excel(path, pdfs_res_dict)
    print("taken time : ", round(time.time()-start,3))