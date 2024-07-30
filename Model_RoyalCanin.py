import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from copy import deepcopy
from unidecode import unidecode

from datetime import datetime
year = datetime.now().year

from paddleocr import PaddleOCR

from JaroDistance import jaro_distance
from ProcessPDF import  binarized_image, get_adjusted_image, HoughLines
from ConditionFilter import condition_filter

MODEL = "RoyalCanin"

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding="utf-8"))[MODEL]

lists_df = pd.read_excel(r"CONFIG\\lists.xlsx")

NULL_OCR = {"text" : "",
            "box" : [],
            "proba" : 0
           }

class KeyMatch:
    def __init__(self, seq_index, confidence, number_of_match, last_place_word, key_index, OCR):
        self.seq_index = seq_index
        self.confidence = confidence
        self.number_of_match = number_of_match
        self.last_place_word = last_place_word
        self.key_index = key_index
        self.OCR = OCR

class ZoneMatch:
    def __init__(self, local_OCR, match_indices, confidence, res_seq):
        self.local_OCR = local_OCR
        self.match_indices = match_indices
        self.confidence = confidence
        self.res_seq = res_seq

def paddle_OCR(image, show=False):

    def _cleanPaddleOCR(OCR_text):
        res = []
        for line in OCR_text:
            for t in line:
                    model_dict = {
                        "text" : "",
                        "box" : [],
                        "proba" : 0
                    }
                    model_dict["text"] = t[1][0]
                    model_dict["box"] = t[0][0]+t[0][2]
                    model_dict["proba"] = round(t[1][1],3)
                    res.append(model_dict)
        
        return res

    def _order_by_tbyx(OCR_text):
        res = sorted(OCR_text, key=lambda r: (r["box"][1], r["box"][0]))
        for i in range(len(res) - 1):
            for j in range(i, 0, -1):
                if abs(res[j + 1]["box"][1] - res[j]["box"][1]) < 20 and \
                        (res[j + 1]["box"][0] < res[j]["box"][0]):
                    tmp = deepcopy(res[j])
                    res[j] = deepcopy(res[j + 1])
                    res[j + 1] = deepcopy(tmp)
                else:
                    break
        return res
    
    ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log = False) # need to run only once to download and load model into memory
    results = ocr.ocr(image, cls=True)
    results = _cleanPaddleOCR(results)
    results = _order_by_tbyx(results)

    if show:
        im = deepcopy(image)
        for i, cell in enumerate(results):
            x1,y1,x2,y2 = cell["box"]
            cv2.rectangle(
                im,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,0,0),2)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

    return results

def find_match(key_sentences, paddleOCR, box, eta=0.95): # Could be optimized
    """
    Detect if the key sentence is seen by the OCR.
    If it's the case return the index where the sentence can be found in the text returned by the OCR,
    else return an empty array
    Args:
        key_sentences (list) : contains a list with one are more sentences, each word is a string.
        text (list) : text part of the dict returned by pytesseract 

    Returns:
        res_indexes (list) : [[start_index, end_index], [empty], ...]   Contains for each key sentence of the landmark the starting and ending index in the detected text.
                            if the key sentence is not detected res is empty.
    """
    def _get_best(base_match, new_match):

        best = base_match
        if new_match == None:
            return best
        elif base_match.number_of_match < new_match.number_of_match: # Choose best match
            best = new_match
        elif (base_match.number_of_match == new_match.number_of_match): # If same number of match, choose the first one
            best=base_match
        return best
    
    xmin,ymin,xmax, ymax = box
    best_matches = None
    for i_place, dict_sequence in enumerate(paddleOCR):
        x1,y1 = dict_sequence["box"][:2]
        seq_match = None
        if xmin<x1<xmax and ymin<y1<ymax:
            sequence = dict_sequence["text"]
            for i_key, key in enumerate(key_sentences): # for landmark sentences from the json
                key_match = None
                for i_word, word in enumerate(sequence):
                    word = unidecode(word).lower()
                    for _, key_word in enumerate(key.split(" ")): # among all words of the landmark
                        key_word = unidecode(key_word).lower()
                        if word[:min(len(word),len(key_word))] == key_word:
                            distance = 1
                        else :
                            distance = jaro_distance("".join(key_word), "".join(word)) # compute the neighborood matching

                        if key_word == "produit" and "fin" in "".join(sequence):
                            distance = 0

                        if distance > eta : # take the matching neighborood among all matching words
                            if key_match == None:
                                key_match = KeyMatch(i_place, distance, 1, i_word, i_key, dict_sequence)
                            elif key_match.last_place_word<i_word:
                                key_match.confidence = min(key_match.confidence, distance)
                                key_match.number_of_match+=1
                if seq_match==None : 
                    seq_match=key_match
                else:
                    seq_match = _get_best(seq_match, key_match)

        if best_matches==None : 
            best_matches=seq_match
        else:
            best_matches = _get_best(best_matches, seq_match)
    
    # if best_matches != None : print(best_matches.OCR["text"], key_sentences[best_matches.key_index], best_matches.number_of_match)
    
    return best_matches

def clean_sequence(paddle_list, full = "|\[]!<>{}—;$€&*‘§—~", left="'(*): |\[]_!.<>{}—;$€&-"):
    res_dicts = []
    for dict_seq in paddle_list:
        text = dict_seq["text"]
        text = text.replace(" :", ":") if " :" in text else text
        text = text.replace(":", ": ") if ":" in text else text
        text = text.replace(":  ", ": ") if ":  " in text else text

        text = text.replace("`", "'") if "`" in text else text
        text = text.replace("_", " ") if "_" in text else text
        text = text.replace("*", "°") if "*" in text else text
        text = text.replace("I'", "l'") if "I'" in text else text

        text = text.replace("-1", "") if "-1" in text else text

        text = text.replace("AA9HO", "AA9H0") if "AA9HO" in text else text
        text = text.replace("AAOVP", "AA0VP") if "AAOVP" in text else text

        if not text in full+left:
            text = [word.strip(full) for word in text.split(" ")]
            dict_seq["text"] = [word for word in text if word]
            res_dicts.append(dict_seq)

    return res_dicts

def get_samples_dict(image, samples_dict, n_image, image_name):
    im_y, im_x = image.shape[:2]

    bd = cv2.barcode.BarcodeDetector()
    found, points = bd.detect(image)

    clean_points = []
    for point in points:
        if not any([abs(point[0][1]-p[0][1])<10 for p in clean_points]):
            clean_points.append(point)

    h_lines = HoughLines(image, mode="horizontal")
    h_lines = [h_line[0][1] for h_line in h_lines if abs(h_line[0][0] - h_line[1][0])>im_x*0.95]

    sample_linesy_points = []
    for point in clean_points:
        sample_linesy_points.append((sorted([h_line for h_line in h_lines], key=lambda l: abs(point[1][1]-l))[0], point))

    sample_linesy_points = sorted(sample_linesy_points, key = lambda x: x[0])

    # sample_linesy_points[0] = lines_y / sample_linesy_points[1] = BC points
    for i_line in range(0, len(sample_linesy_points)-1):

        # Header case with global infos
        if (n_image==0 and i_line==0):
            name = "header"
            global_info = 1
            upper = 0
            lower = sample_linesy_points[i_line+1][0]
        
        # Samples
        else:
            name = f"sample_{len(samples_dict.keys())-1}"
            global_info = 0
            upper = sample_linesy_points[i_line][0]
            lower = sample_linesy_points[i_line+1][0]
        
        # Add header and sanmples in a dict
        samples_dict[name] = {
            "upper" : upper,
            "lower" : lower,
            "global_info" : global_info,
            "images" : [image_name],
            "barcode" : sample_linesy_points[i_line][1]
        }

    # Add the last sample
    samples_dict[f"sample_{len(samples_dict.keys())-1}"] = {
            "upper" : sample_linesy_points[-1][0],
            "lower" : im_y,
            "global_info" : 0,
            "images" : [image_name],
            "barcode" : sample_linesy_points[-1][1]
        }

    return samples_dict

def get_image_OCR_sample(images_OCR_dict, images_dict, sample_dict):

    if len(sample_dict["images"])==1: # The sample is located on one image only

        image_name = sample_dict["images"][0]
        image = images_dict[image_name]

        ymin, ymax = sample_dict["upper"], sample_dict["lower"]

        sample_image = image[ymin:ymax, :]

        full_img_OCR = images_OCR_dict[image_name]

        sample_img_OCR = [dict_sequence for dict_sequence in full_img_OCR if (ymin<dict_sequence["box"][1]<ymax)]
        

    else: # Sample spill out the image
        pass

    return sample_img_OCR, sample_image, image_name

def get_full_image_OCR(image, show=False):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        cropped_image (array)

    Returns:
        zone_match_dict (dict) :  { zone : Match,
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    # Search text on the whole image
    full_img_OCR =  paddle_OCR(image, show)
    full_img_OCR = clean_sequence(full_img_OCR)
    return full_img_OCR

def get_key_matches(full_img_OCR, sample_image, sample_dict, global_info, JSON_HELPER):
    """
    Perform the OCR on the processed image, find the landmarks and make sure there are in the right area 
    Args:
        sample_image (array)

    Returns:
        zone_match_dict (dict) :  { zone : Match,
        }
        The coordinate of box around the key sentences for each zone, empty if not found
        OCR_data (dict) : pytesseract returned dict
    """
    image_height, image_width = sample_image.shape[:2]
    zone_match_dict = {}

    for zone, key_points in JSON_HELPER.items():

        if global_info != key_points["global_info"]:
            continue

        landmark_region = key_points["subregion"] # Area informations
        ymin, ymax = image_height*landmark_region[0][0],image_height*landmark_region[0][1]
        xmin, xmax = image_width*landmark_region[1][0],image_width*landmark_region[1][1]

        if not key_points["key_sentences"]:
                match=None

        else:
            match = find_match(key_points["key_sentences"], full_img_OCR, (xmin,ymin,xmax, ymax))

        if match != None:
            # print("found : ", zone, " - ", match.OCR["box"])
            zone_match_dict.update({zone : match })
            # cv2.rectangle(sample_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else :
            base_match = deepcopy(KeyMatch(0, -1, 0, 0, 0, NULL_OCR))
            base_match.OCR["box"] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            zone_match_dict.update({zone : base_match})

        # print("base : ", zone, (xmin, ymin, xmax, ymax))
        # plt.imshow(sample_image[int(ymin):int(ymax), int(xmin):int(xmax)])
        # plt.show()

    return zone_match_dict

def get_area(cropped_image, box, relative_position, corr_ratio=1.1):
    """
    Get the area coordinates of the zone thanks to the landmark and the given relative position
    Args:
        box (list): detected landmark box [x1,y1,x2,y2]
        relative_position ([[vertical_min,vertical_max], [horizontal_min,horizontal_max]]): number of box height and width to go to search the tet
    """
    im_y, im_x = cropped_image.shape[:2]
    x1,y1,x2,y2 = box
    h, w = abs(y2-y1), abs(x2-x1)
    h_relative, w_relative = h*(relative_position[0][1]-relative_position[0][0])//2, w*(relative_position[1][1]-relative_position[1][0])//2
    y_mean, x_mean = y1+h*relative_position[0][0]+h_relative, x1+w*relative_position[1][0]+w_relative
    x_min, x_max = max(x_mean-w_relative*corr_ratio,0), min(x_mean+w_relative*corr_ratio*2, im_x)
    y_min, y_max = max(y_mean-h_relative*corr_ratio, 0), min(y_mean+h_relative*corr_ratio*2, im_y)
    (y_min, x_min) , (y_max, x_max) = np.array([[y_min, x_min], [y_max, x_max]]).astype(int)[:2]
    return x_min, y_min, x_max, y_max

def get_wanted_text(cropped_image, zone_key_match_dict, full_img_OCR, sample_dict, JSON_HELPER, show=False):
    
    zone_matches = {}
    for zone, key_points in JSON_HELPER.items():

        if sample_dict["global_info"] != key_points["global_info"]:
            continue

        key_match =  zone_key_match_dict[zone]
        box = key_match.OCR["box"]
        condition, relative_position = key_points["conditions"], key_points["relative_position"]         

        # Image relative positions
        xmin, ymin, xmax, ymax = box if key_match.confidence==-1 else get_area(cropped_image, box, relative_position, corr_ratio=1.15)

        candidate_dicts = [dict_sequence for dict_sequence in full_img_OCR if 
                      (xmin<dict_sequence["box"][0]<xmax) and (ymin<dict_sequence["box"][1]-sample_dict["upper"]<ymax)]
                
        zone_match = ZoneMatch(candidate_dicts, [], 0, [])

        match_indices, res_seq = condition_filter(candidate_dicts, condition, MODEL)

        if len(res_seq)!=0:
            if type(res_seq[0]) != type([]):
                res_seq = unidecode(" ".join(res_seq))
            else:
                res_seq = [unidecode(" ".join(seq)) for seq in res_seq]

        zone_match.match_indices , zone_match.res_seq = match_indices, res_seq
        zone_match.confidence = min([candidate_dicts[i]["proba"] for i in zone_match.match_indices]) if zone_match.match_indices else 0

        zone_matches[zone] = {
                "sequence" : zone_match.res_seq,
                "confidence" : float(zone_match.confidence),
                "area" : (int(xmin), int(ymin), int(xmax), int(ymax))
            }
        
        if show:
            print(zone, ": ", res_seq)
            #print(box, key_match.confidence, (xmin, ymin, xmax, ymax))
            plt.imshow(cropped_image[ymin:ymax, xmin:xmax])
            plt.show()

    return zone_matches 

def model_particularities(zone_matches):
        
    if "N_de_commande" in zone_matches.keys():
        seq_split = zone_matches["N_de_commande"]["sequence"].split("-")
        zone_matches["N_de_commande"]["sequence"] = seq_split[0].strip(" ")

    return zone_matches

def textExtraction(sample_image, sample_dict, zone_match_dict, sample_img_OCR,  JSON_HELPER):
    """
    The main fonction to extract text from FDA

    Returns:
        zone_matches (dict) : { zone : {
                                    "sequence": ,
                                    "confidence": ,
                                    "area": }
        }
    """
    zone_matches = get_wanted_text(sample_image, zone_match_dict, sample_img_OCR, sample_dict, JSON_HELPER, show=False)
    zone_matches = model_particularities(zone_matches)

    print("_________________________")
    for zone, dict in zone_matches.items():
        print(zone, ":", dict["sequence"])

    return zone_matches

def main(scan_dict):

    pdfs_res_dict = {}

    for pdf, images_dict in scan_dict.items():
        print("Traitement de :", pdf)

        # Stack samples informations
        samples_dict = {}
        # Stack OCR image by image
        images_OCR_dict = {}
        
        # extract samples from all images
        for n_image, (image_name, image) in enumerate(list(images_dict.items())): # The first image is the only one to concider
            pdfs_res_dict[pdf] = {}

            bin_image = binarized_image(image)
            image = get_adjusted_image(bin_image, show=False)

            samples_dict = get_samples_dict(image, samples_dict, n_image, image_name)

            images_OCR_dict[image_name] = get_full_image_OCR(image, show=False)

        # Process sample by sample
        for sample_name, sample_dict in samples_dict.items():
            
            sample_img_OCR, sample_image, image_name = get_image_OCR_sample(images_OCR_dict, images_dict, sample_dict)

            # print(sample_img_OCR)
            # plt.imshow(sample_image)
            # plt.show()

            zone_match_dict = get_key_matches(sample_img_OCR, sample_image, sample_dict, global_info=sample_dict["global_info"], JSON_HELPER=OCR_HELPER)

            sample_matches = textExtraction(sample_image, sample_dict, zone_match_dict, sample_img_OCR, JSON_HELPER=OCR_HELPER)

            pdfs_res_dict[pdf][sample_name] = {"IMAGE" : image_name,
                                            "EXTRACTION" : sample_matches} # Image Name

        header = pdfs_res_dict[pdf].pop("header")
        for sample_name in pdfs_res_dict[pdf].keys():
            loc_head = deepcopy(header["EXTRACTION"])
            loc_head.update(pdfs_res_dict[pdf][sample_name]["EXTRACTION"])
            pdfs_res_dict[pdf][sample_name]["EXTRACTION"] = loc_head

    
    return pdfs_res_dict

if __name__ == "__main__":

    path = r"C:\Users\CF6P\Desktop\ELNA\Data_ELNA\RC\test\RC Randburg.pdf"

    from ProcessPDF import PDF_to_images
    import os

    images = PDF_to_images(path)

    images = images[:1]
    images_names = ["res"+f"_{i}" for i in range(1,len(images)+1)]

    scan_dict = {"test" : {}}
    for im_n, im in zip(images_names, images):
        scan_dict["test"][im_n] = im

    import time
    start = time.time()
    main(scan_dict)
    print("taken time : ", round(time.time()-start,3))
    
