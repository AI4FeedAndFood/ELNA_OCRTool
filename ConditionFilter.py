import numpy as np
import pandas as pd
 
import re
import os

from copy import deepcopy
from JaroDistance import jaro_distance
from unidecode import unidecode

from datetime import datetime
year = datetime.now().year

def _get_sheet(path, condition_dict, model):

    if not "sheet_name" in condition_dict.keys():
        df = pd.read_excel(path)
    
    else:
        sheet_name = condition_dict["sheet_name"]

        xl = pd.ExcelFile(path)
        sheet_names = xl.sheet_names  # see all sheet names

        if sheet_name in sheet_names:
            df = xl.parse(sheet_name)
        elif sheet_name+" "+model in sheet_names:
            df = xl.parse(sheet_name+" "+model)
        else :
            df = pd.read_excel(path)

    return df

def list_process(candidates, condition_dict, lists_df, min_jaro = 0.95):
    """ Search candidates that contains all the word from a list of word/sequence, with a word by word 'min_jaro' tolerance"

    Args:
        candidates (list of dict): 
        condition (list of three elements): ["list", column of lists_df to extract all the target words, multiple or single]
        lists_df (DataFrame): _description_
    """

    def sub_mistakes(word):
        # Replace common mistakes from the OCR to find a perfect match
        word = word.replace("O", "0") if "O" in word else word
        word = word.replace("l", "I") if "l" in word else word
        return word

    def best_matches_dict(check_word, candidate_sequence, candidate_index):
        """ 
        Returned a dict wich carry the information about the best matched word among all sequences
        """
        max_jaro = min_jaro
        status_dict = {
            "find" : False,
            "index" : -1,
            "jaro" : 0,
            "word" : ""
        }

        for i_word, word in enumerate(candidate_sequence):
            if unidecode(check_word.lower()) in [unidecode(word.strip(": _;").lower()), unidecode(sub_mistakes(word).strip(": _;").lower())]:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = 1, check_word
                return status_dict
            
            jaro = jaro_distance(unidecode(word.lower()), unidecode(check_word.lower()))
            if jaro > max_jaro:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = jaro, check_word

        return status_dict

    all_text, all_indices = [], []
    matched_elmts = []
    mode = condition_dict["mode"]

    for i_text, dict_text in enumerate(candidates):
        all_text+=dict_text["text"]
        all_indices+= [i_text for i in range(len(dict_text["text"]))]

    check_list = list(lists_df[condition_dict["column"]].dropna().unique())

    for check_elmt in check_list:
        found_elmts = []
        check_words = check_elmt.split(" ")
        for check_word in check_words:
            found_dict = best_matches_dict(check_word, all_text, all_indices)
            if found_dict["find"]: # If a check word is found : stack it
                found_elmts.append(found_dict)
    
            if 0<len(found_elmts)<=len(check_words) and check_elmt not in [matched_elmt["element"] for matched_elmt in matched_elmts]: # if some checking ellement are found and there are not assigned yet
                
                matched_elmts.append({
                                    "element": check_elmt,
                                    "words" : found_elmts,
                                    "jaro" : min([d["jaro"] for d in found_elmts]),
                                    "index" : np.median(np.array([d["index"] for d in found_elmts], dtype=np.int64))
                                    })
                
    matched_elmts = sorted(matched_elmts, key=lambda x: x["index"])

    res_seq, match_indices = [], []
    if matched_elmts:
        if mode == "multiple": # Return all found elements sorted by index
            res_seq, match_indices = [[matched_elmt["element"]] for matched_elmt in matched_elmts], [int(matched_elmt["index"]) for matched_elmt in matched_elmts]
        if mode == "single":
            matched_elmts = sorted(matched_elmts, key=lambda x: (-len(x["words"]), -x["jaro"], x["index"]), reverse=False)
            res_seq, match_indices = [matched_elmts[0]["element"]], [int(matched_elmts[0]["index"])]

    return match_indices, res_seq

def cell_process(candidates):
    match_indices, res_seq = [], []
    for i_text, dict_text in enumerate(candidates):
        res_seq+=dict_text["text"]
        match_indices+=[i_text]
    return match_indices, res_seq

def after_key_process(candidates, bound_keys, similarity=0.85):

    def _get_wanted_seq(full_seq, target_words, search_range):
        """
        Find the place of the target_word and the following of the sentence after the word
        
        Return the index and new full sequence
        """
        res_index = 0
        for target_word in target_words:
            target_word = unidecode(target_word).lower()
            for place in range(min(search_range+2, len(full_seq))):
                word = unidecode(full_seq[place]).lower()
                try:
                    index = word.rindex(target_word)
                    res_word = word[index+len(target_word):]
                    if res_word == "" and place < len(full_seq)-1: # The target is at the end of a word
                        res_index+=place
                        full_seq = full_seq[place+1:]
                        break
                    if res_word == "" and place == len(full_seq)-1: # The target is at the end of the seq
                        return -1, []
                    
                    full_seq[place] = res_word
                    res_index+=place
                    full_seq = full_seq[place:]
                    break
                
                except ValueError:
                    pass
        
        return res_index, full_seq
    
    strip = '().*:‘;,§"'+"'"
    key_boundaries = {"after" : [], "before" : []} 

    # Get the all matched between keys and sequences for the start and the end key
    for state, bound_key in bound_keys.items():
        if bound_key ==  "":
            continue
        bound_word = bound_key.split(" ")
        for i_key, key_word in enumerate(bound_word):
            for i_candidate, candidate_sequence in enumerate(candidates):
                    for i_word, word in enumerate(candidate_sequence["text"]):
                        word, key_word = unidecode(word).lower(), unidecode(key_word).lower()
                        word_match = {"i_key" : -1,
                                      "i_candidate" : -1,
                                      "i_word" : -1} # Store the matching key index and the place on the candidate seq
                        # If the rwo words are close
                        if (jaro_distance(key_word, unidecode(word).strip(strip))>similarity):
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[state].append(word_match)
                            break
                        # If a sequence word is containing a key_word
                        if len(key_word)>2 and key_word in word:
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[state].append(word_match)
                            break
    
    if key_boundaries["after"] == [] :
        if len(candidates)==1:
            print("DEFAULT CASE :", candidates[0]["text"])
            return [0], candidates[0]["text"]
        return [], []

    # Get boundaries of the desired text
    _order_by_match_candidate = lambda match_list : sorted(match_list, 
                                                key=lambda match : ([d["i_candidate"] for d in match_list].count(match["i_candidate"]), -match["i_candidate"]), 
                                                                    reverse=True) # Most matched sentence, the firt one in case of tie
   
    last_start_match = sorted(_order_by_match_candidate(key_boundaries["after"]), key=lambda d:d["i_key"], reverse=True)[0]

    if last_start_match["i_word"] == len(candidates[last_start_match["i_candidate"]]["text"])-1:
        if last_start_match["i_candidate"] == len(candidates)-1:
            return [], []
        last_start_match["i_candidate"], last_start_match["i_word"] = last_start_match["i_candidate"]+1, 0

    if key_boundaries["before"]==[]:
        first_end_seq_id = deepcopy(last_start_match) # Equivalent to a mis match : Will be changed in the next step
    else:
        first_end_seq_id = sorted(_order_by_match_candidate(key_boundaries["before"]), key=lambda d:d["i_key"], reverse=False)[0]
        # Empty field case
        if (first_end_seq_id["i_candidate"], first_end_seq_id["i_word"]) == (last_start_match["i_candidate"], last_start_match["i_word"]):
            return [],  []
 
    # If there is a mismatch, end is set as the sequence that followed the start
    if first_end_seq_id["i_candidate"] <= last_start_match["i_candidate"] :
        first_end_seq_id["i_candidate"] = min(last_start_match["i_candidate"]+1, len(candidates))
        first_end_seq_id["i_word"] = 0

    text_candidates = [candidates[i]["text"] for i in range(last_start_match["i_candidate"], first_end_seq_id["i_candidate"])]
    text_candidates[0] = text_candidates[0][last_start_match["i_word"]:]

    # Get all the found text as a list of string words
    all_text = [word for text_candidate in text_candidates for word in text_candidate]
    all_local_indices = [i for i,text in enumerate(text_candidates) for s in range(len(text))]

    search_range = (len(bound_keys["after"]) - last_start_match["i_key"]) # 0> if the last detected word is not the last key word 
    target_words = [bound_keys["after"].split(" ")[-1]] + ["(*):"]
    
    line_index, res_seq = _get_wanted_seq(all_text, target_words, search_range)

    return [all_local_indices[line_index]], res_seq

def date_process(candidates, strip):
    match_indices, res_seq = [], []
    for i_candidate, candidate in enumerate(candidates):
        try:
            word_test = " ".join(candidate["text"])
            date  = datetime.strptime(word_test, "%B %d, %Y")
            return [i_candidate], [date.strftime("%d/%m/%Y")]
        except ValueError:
            pass
    
        for i_word, word in enumerate(candidate["text"]):
            try:
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                date = datetime.strptime(word_test, "%d/%m/%Y")
                return [i_candidate], [datetime.strftime(date, "%d/%m/%Y")]
            except ValueError:
                pass

            try:
                word_test = word[:10].lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                date = datetime.strptime(word_test, "%d/%m/%Y")
                return [i_candidate], [datetime.strftime(date, "%d/%m/%Y")]
            except ValueError:
                pass
            
            try: # Case dd/mm/yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                word_test = word_test[:-2] + "20" + word_test[-2:]
                date = datetime.strptime(word_test, "%d/%m/%Y")
                return [i_candidate], [datetime.strftime(date, "%d/%m/%Y")]
            except ValueError:
                pass

            try: # Case dd,mm,yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                word_test = word_test[:-2] + "20" + word_test[-2:]
                date = datetime.strptime(word_test, "%d,%m,%Y")
                return [i_candidate], [datetime.strftime(date, "%d/%m/%Y")]
            except ValueError:
                pass

            try: # Case dd-mm-yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                date = datetime.strptime(word_test, "%d/%m/%Y")
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try: # Case month in letter
                word = word.lower().strip(strip)
                _ = bool(datetime.strptime(word, "%B"))
                full_date = "".join(candidate[i_word-1:i_word+2])
                date = datetime.strptime(full_date, "%d%B%Y")
                return [i_candidate], [datetime.strftime(date, "%d/%m/%Y")]
            except:
                continue
            
    return match_indices, res_seq

def contain_process(candidates, keys):
    matched_candidates = []
    matched_indices = []
    for i_cand, candidate in enumerate(candidates):
        for i_word, word in enumerate(candidate["text"]):
            for _, key_word in enumerate(keys):
                if key_word in word:
                    matched_candidates.append(word)
                    matched_indices.append(i_cand)

    return matched_indices, matched_candidates

def format_process(candidates, keys):
    
    def _format_start_with(key_word, word):
        if key_word == "YYMM":
            YY = year-2000
            if word[:3].isnumeric() and len(word)>3:
                if YY-2<int(word[:2])<YY+1 and 0<int(word[2:4])<13:
                    return True
        
        if set(key_word) == {"N"}:
            if word[:len(key_word)].isnumeric() and len(word)>=len(key_word):
                return True

        elif word.startswith(key_word):
            return True

        return False

    best_count_index, best_count = -1, 0
    for i_cand, candidate in enumerate(candidates):
        count = 0
        word = "".join(candidate["text"])
        word = re.sub("_-/\[]]", "", word)
        for _, key_word in enumerate(keys):
            key_word = re.sub("_-/\[]]", "", key_word)
            if _format_start_with(key_word, word):
                count+=1
                word = word[len(key_word):]
        if count>best_count:
            best_count=count
            best_count_index = i_cand       

    if best_count_index == -1:
        return [], []

    matched_indices, matched_candidates = [best_count_index], candidates[best_count_index]["text"]

    return matched_indices, matched_candidates

def check_process(candidates, sense, checkboxes, lists_df, list_condition_dict):
    """_summary_

    Args:
        candidates (list of dict):  of candidate dict
        sense (right or left): Direction where to search after the checkbox as been detected
        checkboxes (list of arrays): All screened checkboxes
        lists_df (pd.Dataframe): df of all lists as columns
        list_condition_dict (dict): "mode" : single or multiple values ; "column" : the column where of the list

    Returns:
        candidates_indices, check_candidates
    """
    check_candidates = []
    candidates_indices = []

    # First extract candidates according to checkboxes
    for checkbox in checkboxes:
        min_dist = 10000
        nearest_candidate, n_index = None, None

        up, down = checkbox["TOP_LEFT_Y"], checkbox["BOTTOM_RIGHT_Y"]
        left, right = checkbox["TOP_LEFT_X"], checkbox["BOTTOM_RIGHT_X"]

        for i_cand, candidate in enumerate(candidates):
            x1,y1,x2,y2 = candidate["box"]
            if (up<=y1<=down) or (up<=y2<=down) or (y1<=up<=y2) or (y1<=down<=y2):
                dist = left-x2 if sense == "left" else x1-right
                if 0<dist<min_dist:
                    min_dist = dist
                    nearest_candidate, n_index = candidate, i_cand

        # Found  list
        if nearest_candidate and list_condition_dict:

            _, res_seq = list_process([nearest_candidate], list_condition_dict, lists_df)

            if res_seq and not res_seq in check_candidates:
                check_candidates.append(res_seq)
                candidates_indices.append(i_cand)

        else:
            candidates_indices, check_candidates = n_index, nearest_candidate
    
    return candidates_indices, check_candidates

def Nechantillon_ELPV(candidates):
    match_indices, res_seq = [], []
    base1, base2 = year-1, year+2 # Thresholds
    
    for i_candidate, candidate in enumerate(candidates):
        if len(candidate["text"])<4:
            for i_word, word in enumerate(candidate["text"]):
                if len(word)>3 and not "/" in word:
                    try_list = [(base1-2000, base2-2000), (base1,base2)]
                    for date_tuple in try_list:
                        num1, num2 = date_tuple
                        if word[:len(str(num1))].isnumeric():
                            date_num, code_num = word[:len(str(num1))], word[len(str(num1)):].upper()
                            if num1 <= int(date_num) < num2 : # Try to avoid strings shorter than NUM
                                res="".join(candidate["text"][i_word:])
                                res_upper = res.upper()
                                
                                # Replace common mistake
                                correction_list = [("MPA", "MP4"), ("N0", "NO"), ("AUOP", "AU0P"), ("CEOP", "CE0P"), ("GEL0", "GELO"), ("PLOP", "PL0P"), 
                                    ("PLIP", "PL1P"), ("NCIP", "NC1P"), ("NCIE", "NC1E"), ("S0R", "SOR"), ("1F", "IF")]
                                for cor_tuple in correction_list:
                                    error, correction = cor_tuple
                                    if code_num[:len(error)] == error:
                                        res_upper = res_upper.replace(error, correction, 1)
                                        break # One possibility
                                if code_num[5:9] == "S0PDT": # Unique case
                                    res_upper.replace("SP0DT", "SOPODT", 1)
                                
                                match_indices, res_seq = [i_candidate], [res_upper]
                    # Special case
                    if "GECA" in word:
                        res_upper = str(year)+word if str(year) not in word else word
                        try :
                            if candidate["text"][i_word+1].isnumeric():
                                res_upper += candidate["text"][i_word+1]
                        except:
                            pass
                        match_indices, res_seq = [i_candidate], [res_upper]

    return match_indices, res_seq

def condition_filter(candidates_dicts, condition, model, application_path, ocr_pathes, checkboxes=None):
    """_summary_

    Args:
        candidates_dicts (_type_): _description_
        key_main_sentences (_type_): _description_
        conditions (_type_): _description_

    Returns:
        _type_: _description_
    """
    strip =  "|\[]_!<>{}—;$€&*‘§—~-'(*): " + '"'
    # Arbitrary set
    candidates = deepcopy(candidates_dicts)
    match_indices, res_seq = [], []

    if condition[0] == "contains":
        keys = condition[1]
        match_indices, res_seq = contain_process(candidates, keys)

    if condition[0] == "after_key":
        bound_keys = condition[1] # Start and end key_sentences
        match_indices, res_seq = after_key_process(candidates, bound_keys)
            
    if condition[0] == "date": # Select a date format
        match_indices, res_seq = date_process(candidates, strip)

    if condition[0] == "list": # In this case itertion is over element in the condition list
        condition_dict = condition[1]
        path = os.path.join(application_path, ocr_pathes[condition_dict["path"]])

        lists_df = _get_sheet(path, condition_dict, model)
        
        match_indices, res_seq= list_process(candidates, condition_dict, lists_df)

    if condition[0] == "format": # In this case itertion is over element in the condition list
        keys = condition[1]
        match_indices, res_seq = format_process(candidates, keys)
    
    if condition[0] == "constant":
        match_indices, res_seq = [], [condition[1]]
    
    if condition[0] == "checkbox":
        # The sens where to extract sentences starting from the checkboxe
        sense = condition[1]
        # If a list could help to find the sentence
        try:
            list_condition_dict = condition[2]
            # if the list file is global        
            path = os.path.join(application_path, ocr_pathes[list_condition_dict["path"]])
            lists_df = _get_sheet(path, list_condition_dict, model)
        except:
            list_condition_dict, lists_df = None, None

        match_indices, res_seq = check_process(candidates, sense, checkboxes, lists_df, list_condition_dict)
    
    if condition[0] == "echantillon_ELPV": # A special filter for numero d'echantillon
        match_indices, res_seq = Nechantillon_ELPV(candidates)
    
    if condition[0] == "cell": # A special filter for numero d'echantillon
        match_indices, res_seq = cell_process(candidates)

    return match_indices, res_seq

if __name__ == "__main__":
    candidates = [{'text': ['December', '18,', '2023'], 'box': [1723.0, 908.0, 2091.0, 949.0], 'proba': 0.998}]
    condition = ["date"]

    print(condition_filter(candidates, condition))