import numpy as np
import pandas as pd

from copy import deepcopy
from JaroDistance import jaro_distance
from unidecode import unidecode

from datetime import datetime
year = datetime.now().year

def list_process(candidates, condition, lists_df):
    def sub_mistakes(word):
        # Replace common mistakes from the OCR to find a perfect match
        word = word.replace("O", "0") if "O" in word else word
        word = word.replace("l", "I") if "l" in word else word
        return word

    def best_matches_dict(check_word, candidate_sequence, candidate_index):
        """ 
        Returned a dict wich carry the information about the best matched word among all sequences
        """
        max_jaro = 0.89
        status_dict = {
            "find" : False,
            "index" : -1,
            "jaro" : 0,
            "word" : ""
        }

        for i_word, word in enumerate(candidate_sequence):
            if check_word.lower() in [ word.strip(": _;").lower(), sub_mistakes(word).strip(": _;").lower()]:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = 1, check_word
                return status_dict
            
            jaro = jaro_distance(word.lower(), check_word.lower())
            if jaro > max_jaro:
                status_dict["find"], status_dict["index"] = True, candidate_index[i_word]
                status_dict["jaro"], status_dict["word"] = jaro, check_word

        return status_dict

    all_text, all_indices = [], []
    matched_elmts = []
    mode = condition[2]

    for i_text, dict_text in enumerate(candidates):
        all_text+=dict_text["text"]
        all_indices+= [i_text for i in range(len(dict_text["text"]))]

    check_list = list(lists_df[condition[1]].dropna().unique())
    for check_elmt in check_list:
        found_elmts = []
        check_words = check_elmt.split(" ")
        for check_word in check_words:
            found_dict = best_matches_dict(check_word, all_text, all_indices)
            if found_dict["find"]: # If a check word is found : stack it
                found_elmts.append(found_dict)
    
            if len(found_elmts) == len(check_words) and check_elmt not in [matched_elmt["element"] for matched_elmt in matched_elmts]: # All word of the checking elements are in the same candidate sequence                          print(check_elmt, jaro_elmt)
                
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
            matched_elmts = sorted(matched_elmts, key=lambda x: (x["jaro"], -x["index"]), reverse=True)
            res_seq, match_indices = [matched_elmts[0]["element"]], [int(matched_elmts[0]["index"])]

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
    ref = {0:"start", 1:"end"}
    key_boundaries = {"start" : [], "end" : []} 

    # Get the all matched between keys and sequences for the start and the end key
    for state, bound_key in enumerate(bound_keys):
        for i_key, key_word in enumerate(bound_key):
            for i_candidate, candidate_sequence in enumerate(candidates):
                    for i_word, word in enumerate(candidate_sequence["text"]):
                        word, key_word = unidecode(word).lower(), unidecode(key_word).lower()
                        word_match = {"i_key" : -1,
                                      "i_candidate" : -1,
                                      "i_word" : -1} # Store the matching key index and the place on the candidate seq
                        # If the rwo words are close
                        if (jaro_distance(key_word, unidecode(word).strip(strip))>similarity):
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[ref[state]].append(word_match)
                            break
                        # If a sequence word is containing a key_word
                        if len(key_word)>2 and key_word in word:
                            word_match["i_key"],  word_match["i_candidate"], word_match["i_word"] = i_key, i_candidate, i_word
                            key_boundaries[ref[state]].append(word_match)
                            break
    
    if key_boundaries["start"] == [] :
        if len(candidates)==1:
            print("DEFAULT CASE :", candidates[0]["text"])
            return [0], candidates[0]["text"]
        return [], []

    # Get boundaries of the desired text
    _order_by_match_candidate = lambda match_list : sorted(match_list, 
                                                key=lambda match : ([d["i_candidate"] for d in match_list].count(match["i_candidate"]), -match["i_candidate"]), 
                                                                    reverse=True) # Most matched sentence, the firt one in case of tie
   
    last_start_match = sorted(_order_by_match_candidate(key_boundaries["start"]), key=lambda d:d["i_key"], reverse=True)[0]

    if last_start_match["i_word"] == len(candidates[last_start_match["i_candidate"]]["text"])-1:
        if last_start_match["i_candidate"] == len(candidates)-1:
            return [], []
        last_start_match["i_candidate"], last_start_match["i_word"] = last_start_match["i_candidate"]+1, 0

    if key_boundaries["end"]==[]:
        first_end_seq_id = deepcopy(last_start_match) # Equivalent to a mis match : Will be changed in the next step
    else:
        first_end_seq_id = sorted(_order_by_match_candidate(key_boundaries["end"]), key=lambda d:d["i_key"], reverse=False)[0]
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

    search_range = (len(bound_keys[0]) - last_start_match["i_key"]) # 0> if the last detected word is not the last key word 
    target_words = [bound_keys[0][-1]] + ["(*):"]
    
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
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try:
                word_test = word[:10].lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass
            
            try: # Case dd/mm/yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                word_test = word_test[:-2] + "20" + word_test[-2:]
                _ = bool(datetime.strptime(word_test, "%d/%m/%Y"))
                return [i_candidate], [word_test]
            except ValueError:
                pass

            try: # Case dd-mm-yy
                word_test = word.lower().strip(strip+"abcdefghijklmnopqrstuvwxyz")
                _ = bool(datetime.strptime(word_test, "%d-%m-%Y"))
                return [i_candidate], [word_test.replace("-", "/")]
            except ValueError:
                pass

            try: # Case month in letter
                word = word.lower().strip(strip)
                _ = bool(datetime.strptime(word, "%B"))
                full_date = "".join(candidate[i_word-1:i_word+2])
                _ = bool(datetime.strptime(full_date, "%d%B%Y"))
                return [i_candidate], [word]

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
        for _, key_word in enumerate(keys):
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

def condition_filter(candidates_dicts, condition, MODEL):
    """_summary_

    Args:
        candidates_dicts (_type_): _description_
        key_main_sentences (_type_): _description_
        conditions (_type_): _description_

    Returns:
        _type_: _description_
    """
    strip =  "|\[]_!<>{}—;$€&*‘§—~-'(*):" + '"'
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
        lists_df = pd.read_excel(r"CONFIG\\lists.xlsx", sheet_name=MODEL)
        match_indices, res_seq= list_process(candidates, condition, lists_df)

    if condition[0] == "format": # In this case itertion is over element in the condition list
        keys = condition[1]
        match_indices, res_seq = format_process(candidates, keys)
    
    if condition[0] == "constant":
        match_indices, res_seq = [], [condition[1]]

            
    return match_indices, res_seq

if __name__ == "__main__":
    candidates = [{'text': ['December', '18,', '2023'], 'box': [1723.0, 908.0, 2091.0, 949.0], 'proba': 0.998}]
    condition = ["date"]

    print(condition_filter(candidates, condition))