import os
import dicttoxml
import json
from shutil import copyfile
from copy import deepcopy
from datetime import datetime

def _update_dict(stack_dict, value, keys_path):
    keys_path = keys_path.split(".") if type(keys_path)==type("") else keys_path
    if len(keys_path) == 0:
        return 
    if len(keys_path) == 1:
        stack_dict[keys_path[0]] = value
        return 
    key, *new_keys = keys_path
    if key not in stack_dict:
        stack_dict[key] = {}
    _update_dict(stack_dict[key], value, new_keys)
    return

def _get_dict_value(dict, keys_path):

    value = dict

    for key in keys_path.split("."):

        try:
            value = value[key]
        
        # If the key not in the dict
        except KeyError:
            return False
        
    return value

def runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows):

    analyses, comments = [], []

    for key, items in verif_values.items():

        # Analysis case
        if key[0] == "ana":
            index = key[1]
            if not index in deleted_rows:
                analyses.append(items)
                # comments.append(verif_values[("spec", index)])
            # comments.append(verif_values[("spec", index)])

        # Client case
        elif key[0] == "to_save":
            res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[1]] = items

        # Product Code case
        elif key[0] in ["code_produit", "rush_level"]:
            if items:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"].update({key[0] :  key[1]})

        # Other cases
        elif key[0] in ["zone"]:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key[1]]["sequence"] = items

    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse"]["sequence"] = analyses
    # res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse_specification"] = {"sequence" : comments}

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)

    return res_dict

def keepNeededFields(verified_dict, client_contract, model, to_keep_field=[], added_field=[]):
    """Extract fields to return to the lims from the interface

    Args:
        verified_dict (_type_): _description_
        client_contract (_type_): _description_
        model (_type_): _description_
    """

    # dict for clean fields
    sample_clean_dict = {}
    for key, value  in verified_dict.items():
        if not key in added_field:
            sample_clean_dict[key] = value["sequence"]
        if key in to_keep_field:
            sample_clean_dict[key]=value

    # Add the code_produit
    if "code_produit" in sample_clean_dict.keys():
        sample_clean_dict["code_produit"] = verified_dict["code_produit"]["sequence"].split(": ")[1] if verified_dict["code_produit"]["sequence"] else ""

    # Find the Contract and the quotation according to the data
    clientName, contractName = verified_dict["client_name"], verified_dict["contract_name"]
    corresponding_row = client_contract[(client_contract["ClientName"]==clientName) & (client_contract["ContractName"]==contractName)].fillna(value="")

    if len(corresponding_row)==1:
        CustomerCode, Contractcode, QuotationCode = corresponding_row[["CustomerCode", "ContractCode", "QuotationCode"]].values.tolist()[0]
    else:
        CustomerCode, Contractcode, QuotationCode,  = "", "", ""

    sample_clean_dict["Name"] = clientName
    sample_clean_dict["CustomerCode"] = CustomerCode
    sample_clean_dict["ContractCode"] = Contractcode
    sample_clean_dict["QuotationCode"] = QuotationCode

    return sample_clean_dict

def convertDictToLIMS(stacked_samples_dict, lims_converter, analysis_lims):

    def _get_packages_tests(analysis, analysis_lims):
        """From the list of the analysis and

        Args:
            analysis (_type_): _description_
            analysis_lims (_type_): _description_

        Returns:
            _type_: _description_
        """

        related_test, all_codes = [], []
        for analyse in analysis:
            all_codes += analysis_lims[analysis_lims["Denomination"]==analyse]["Code"].values.tolist()

        # Add related tests
        related_code_lims = analysis_lims.dropna(subset="Related")

        if len(related_code_lims)>0:

            # First all imposed tests
            everytime_list = related_code_lims[related_code_lims["Related"]=="EVERYTIME"]["Code"].values.tolist()
            if everytime_list:
                related_test+=everytime_list

            # All tests that depend on one test
            one_related = related_code_lims[related_code_lims["Related"].str.startswith("ONE")]
            for code in all_codes:
                test = one_related[one_related["Related"].str.contains(code)]["Code"].values.tolist()
                if test:
                    if not test[0] in related_test:
                        related_test.append(test[0])

            # All tests that depend on several tests
            all_related = related_code_lims[related_code_lims["Related"].str.startswith("ALL")]
            if len(all_related)>0:
                all_related.loc[:,["Related"]] =  deepcopy(all_related["Related"]).apply(lambda x: x.split(": ")[1].split(" "))

            for ind in all_related.index:
                if set(all_related["Related"][ind]).issubset(set(all_codes)):
                    related_test.append(all_related["Code"][ind])

            # All tests that create a package
            all_packages = related_code_lims[related_code_lims["Related"].str.startswith("PACKAGE")]
            if len(all_packages)>0:
                all_packages.loc[:,["Related"]] =  deepcopy(all_packages["Related"]).apply(lambda x: x.split(": ")[1].split(" "))
            
            all_packages_index_init = deepcopy(all_packages.index)

            all_packages["len"] = all_packages["Related"].apply(lambda x: len(x))
            all_packages = all_packages.sort_values("len", ascending=False)
            for ind in all_packages_index_init:
                test_in_package, package_code = all_packages["Related"][ind], all_packages["Code"][ind]
                if set(all_codes).intersection(set(test_in_package)) == set(test_in_package):
                    related_test.append(package_code)
                    all_codes = list(set(all_codes) - set(test_in_package))

        related_test = list(set(related_test))
        all_codes += related_test

        # Finally, sort all tests
        customer_package, package_codes, test_codes = [], [], []
        for code in all_codes:
            if len(code)<5:
                customer_package.append(code)
            elif code[0] == "P":
                package_codes.append(code)
            else:
                test_codes.append(code)

        return package_codes, test_codes, customer_package   

    # The return 
    xmls_format_dict = []

    for sample_dict in stacked_samples_dict:
        
        sample_XML_dict = {}

        # Give all new items to the sample dict
        package_code, test_code, customer_package = _get_packages_tests(sample_dict["analyse"], analysis_lims)

        if package_code:
            sample_dict["PackageCode"] = package_code
        if test_code:
            sample_dict["TestCode"] = test_code
        if customer_package:
            sample_dict["CustomerPackage"] = customer_package

        # Add the quotation code to the customer package
        QuotationCode = sample_dict["QuotationCode"]
        sample_dict["CustomerPackage"] = [QuotationCode+"."+pack for pack in customer_package]

        input_keys = list(sample_dict.keys())
        
        for name, convert_dict in lims_converter.items():
            
            value = None
            path, input = convert_dict["path"], convert_dict["input"]
            keys_path = path.split(".")

            if input in input_keys:
                value = sample_dict[input]

            elif type(input) == type([]):
                
                # Implemented cases
                input = [inp if inp!="DATE" else str(datetime.now().date()) for inp in input]
                
                # If the input not in key the input i hardcoded
                sample_dict_keys = list(sample_dict.keys())
                value = convert_dict["join"].join([sample_dict[inp] if inp in sample_dict_keys else inp for inp in input])

            elif input == "HARDCODE":
                value = convert_dict["value"]
            
            if value:
                try:
                    date = datetime.strptime(value, "%d/%m/%Y")
                    value = datetime.strftime(date, "%Y-%m-%d")
                except:
                    pass
                _update_dict(sample_XML_dict, value, keys_path)
        
        # Warning if not client or contract
        if not _get_dict_value(sample_XML_dict, "Order.CustomerCode") or not _get_dict_value(sample_XML_dict, "Order.ContractCode"):
            customerRef =  _get_dict_value(sample_XML_dict, "Order.Samples.Sample.CustomerReference")
            if customerRef:
                print(f"PAS DE CLIENT TROUVE POUR : {customerRef}, A CODER MANUELLEMENT")
            else :
                print(f"PAS DE CLIENT TROUVE")

        else:
            xmls_format_dict.append(sample_XML_dict)
        
    return xmls_format_dict

def arrangeForClientSpecificites(stacked_samples_dict, analysis_lims, model):

    def _split_analysis(stacked_samples_dict, analysis_lims):
        
        res_saple_dict = []

        for sample_dict in stacked_samples_dict:
            # Iterate over the copy of the dict which is not going to change
            sample_dict_copy = deepcopy(sample_dict)
            
            # Make n sample for n type of analysis
            for type in analysis_lims["Type"].unique():
                analysis_by_type = analysis_lims[analysis_lims["Type"]==type]["Denomination"].values.tolist()
                sample_dict_copy["analyse"] = list(set(analysis_by_type) & set(sample_dict["analyse"]))
                # If no analysis of the type
                if sample_dict_copy["analyse"]:
                    res_saple_dict.append(deepcopy(sample_dict_copy))

        return res_saple_dict
    
    def _duplicateSample(stacked_samples_dict, variable="N_d_echantillon"):

        res_saple_dict = []
        for sample_dict in stacked_samples_dict:

            # Iterate over the copy of the dict which is not going to change
            n_start, n_end = sample_dict["n_start"], sample_dict["n_end"]
            with_0 = sample_dict["with_0"]
            if n_start and n_end:
                for i_copy in range(int(n_start),int(n_end)+1):
                    copy_sample_dict = deepcopy(sample_dict)
                    if with_0 and i_copy<10:
                        i_name = f"0{i_copy}"
                    else :
                        i_name = i_copy
                    copy_sample_dict = deepcopy(sample_dict)
                    copy_sample_dict[variable] = copy_sample_dict[variable]+str(i_name)
                    res_saple_dict.append(copy_sample_dict)
            else:
                res_saple_dict.append(sample_dict)

        return res_saple_dict

    if "CU" in model:
        stacked_samples_dict =  _split_analysis(stacked_samples_dict, analysis_lims)
    
    if model in ["Fredon", "SEMAE"]:
        stacked_samples_dict = _duplicateSample(stacked_samples_dict)
        
    return stacked_samples_dict

def mergeOrderSamples(stacked_samples_dict, merge_condition="Order.ContractCode"):
    
    def _merge_bool(merged_dict, sample_dict, merge_condition):

        if type(merge_condition) == type([]):
            return all([_get_dict_value(merged_dict, condi) == _get_dict_value(sample_dict, condi) for condi in merge_condition])
        else :
            return _get_dict_value(merged_dict, merge_condition) == _get_dict_value(sample_dict, merge_condition)

    stacked_merged_dict = []
    added_number = []

    # Case no merging requested
    if not merge_condition:
        return stacked_samples_dict, added_number
    
    # Other case
    for sample_dict in stacked_samples_dict:
        merged = False
        
        for i_dict, merged_dict in enumerate(stacked_merged_dict):

            # Merge samples by common 
            if _merge_bool(merged_dict, sample_dict, merge_condition):
                new_number = len(_get_dict_value(merged_dict, "Order.Samples"))
                # Store to clean the xml after the conversion
                added_number.append(new_number)
                # Generete de new dict path
                new_path = f"Order.Samples.Sample_"+str(new_number)
                _update_dict(stacked_merged_dict[i_dict], _get_dict_value(sample_dict, "Order.Samples.Sample"), new_path)
                merged = True
        if not merged:
            stacked_merged_dict.append(sample_dict)

    return stacked_merged_dict, added_number

def saveToCopyFolder(save_folder, pdf_path, rename="", mode="same"):
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        base, extension = os.path.splitext(os.path.split(pdf_path)[1])

        if rename:
            base=rename
        
        if mode == "same":
            new_name = base+extension

        copyfile(pdf_path, f"{save_folder}/{new_name}")

def finalSaveDict(verified_dict, xmls_save_path, analysis_lims, model, lims_helper, client_contract, xml_name="verified_XML"):
    
    def _rename_sample(xml, added_number):
        xml = xml.decode("UTF-8")
        for number in added_number:
            if f'<Sample_{number} type="dict">' in xml:
                xml = xml.replace(f'<Sample_{number} type="dict">', '<Sample type="dict">')
                xml = xml.replace(f'</Sample_{number}>', '</Sample>')

        xml = xml.encode("UTF-8")
        return xml

    # For all sample to extract from pdfs, keep only relevant fields
    stacked_samples_dict = []
    for pdf_name, sample_dict in verified_dict.items():
        for sample, res_dict in sample_dict.items():
            xml_name = datetime.now().strftime("%Y%m%d%H")
            sample_XML_dict = keepNeededFields(res_dict["EXTRACTION"], client_contract, model, to_keep_field=lims_helper["ADDED_FIELDS"], added_field=lims_helper["ADDED_FIELDS"])
            stacked_samples_dict.append(sample_XML_dict)

    # For all extracted dict, arrage them according to the client/labs needs
    stacked_samples_dict = arrangeForClientSpecificites(stacked_samples_dict, analysis_lims, model)

    # Convert samples dict to the XML format
    xmls_format_dict = convertDictToLIMS(stacked_samples_dict, lims_helper["LIMS_CONVERTER"], analysis_lims)

    xmls_merged_dict, added_number = mergeOrderSamples(xmls_format_dict, lims_helper["SAMPLE_MERGER"])
    
    #######

    # Then convert each dict as XML
    xml_names = []
    for sample_dict in xmls_merged_dict:

        xml_name = "_".join([_get_dict_value(sample_dict, "Order.RecipientLabCode"), pdf_name, datetime.today().strftime('%Y%m%d')])
        # Set the name of the XML : BU_REF_YYMMDD_N
        num = 0
        sample_XML_num = xml_name+f"_{num}"
        while sample_XML_num in xml_names:
            num+=1
            sample_XML_num = xml_name+f"_{num}"
        xml_names.append(sample_XML_num)

        # Create the XML
        xml = dicttoxml.dicttoxml(sample_dict)
        xml = _rename_sample(xml, added_number)
        xml_save_path = os.path.join(xmls_save_path, f"{sample_XML_num}.xml")
        with open(xml_save_path, 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())

if __name__ == "__main__":
    import pandas as pd

    verified_dict = {'Nutricia_2': {'sample_0': {'IMAGE': 'image_0', 'EXTRACTION': {'date_de_commande': {'sequence': '27/11/2023', 'confidence': 1.0, 'area': [0, 435, 979, 870]}, 'N_d_echantillon': {'sequence': '231127029', 'confidence': 1.0, 'area': [0, 435, 979, 870]}, 'type': {'sequence': 'Matiere premiere', 'confidence': 0.957, 'area': [0, 486, 1577, 928]}, 'nature_produit': {'sequence': 'TOURTEAUX DE SOJA EXPELLER PCR 0.9% MAX FR', 'confidence': 0.999, 'area': [0, 713, 2333, 851]}, 'lieu_prelevement': {'sequence': 'SOAL Pomarez', 'confidence': 0.971, 'area': [0, 767, 2448, 898]}, 'classe_produit': {'sequence': 'Inconnu', 'confidence': 1.0, 'area': [0, 824, 1961, 941]}, 'N_de_lot': {'sequence': '0223328004', 'confidence': 0.948, 'area': [0, 870, 2448, 1001]}, 'fournisseur': {'sequence': "Gr D'ALLIANCE", 'confidence': 1.0, 'area': [0, 923, 2448, 1054]}, 'date_de_prelevement': {'sequence': '24/11/2023', 'confidence': 1.0, 'area': [0, 964, 2448, 1120]}, 'analyse': {'sequence': ['ID920', 'ID918', 'AA9H0'], 'confidence': 0.955, 'area': [0, 1207, 2448, 1451]}, 'client_name': 'HORS OQUALIM - Facturation NUTRICIA °', 'contract_name': 'Kevin DAILLEDOUZE'}}}}

    OCR_HELPER = json.load(open("CONFIG\OCR_config.json"))


    client_contract =  pd.read_excel(r"CONFIG\\eLIMS_contract_analysis.xlsx", sheet_name="client_contract", dtype=str)
    xml_save_path = "C:\\Users\\CF6P\\Desktop\\ELPV\\Data\\test1"
    model = "Nutricia"
    analysis_lims = pd.read_excel(r"CONFIG\\eLIMS_contract_analysis.xlsx", sheet_name="analyse Nutricia")
    lims_helper =  json.load(open("CONFIG\LIMS_CONFIG.json"))

    finalSaveDict(verified_dict, xml_save_path, analysis_lims, model, lims_helper, client_contract, xml_name="verified_XML")
