import PySimpleGUI as sg
import sys, os
import shutil
import numpy as np
import io
import json
import pandas as pd
from copy import deepcopy

from screeninfo import get_monitors
from PIL import Image

from LaunchTool import getAllImages, TextCVTool
from SaveDict import runningSave, finalSaveDict, saveToCopyFolder

def is_valid(folderpath, model):

    if len(model) != 1:
        sg.popup_error("Veuillez cocher un model")
        return False
    if not (folderpath and os.path.exists(folderpath)):
        sg.popup_error("Le dossier n'existe pas")
        return False
    
    scan_dict = getAllImages(folderpath)

    if scan_dict == {}:
        sg.popup_ok("Aucun PDF n'est trouvé dans le dossier")
        return False
    
    return scan_dict

def fit_the_screen(X_loc):
    ordered_monitors = sorted([monitor for monitor in get_monitors()], key=lambda mtr: mtr.x)
    for monitor in ordered_monitors:
        x, width, height = monitor.x, monitor.width, monitor.height
        if X_loc<x+width:
            return int(0.95*width), int(0.95*height)

def use_the_tool(folderPath, model):
    scan_dict, pdfs_res_dict = TextCVTool(folderPath, model)
    return scan_dict, pdfs_res_dict

def create_row(row_counter, analyse, spec):
    size = (int(1.5*INPUT_LENGTH)-20, 1)
    row = [
        sg.pin(
            sg.Col([[sg.Push(),
                    
                    sg.I(analyse,
                        key=("ana",row_counter), enable_events=True, expand_y=True, expand_x=True, justification='left', size=size),
                        
                    # sg.I(spec,
                    #     key=("spec", row_counter), enable_events=False, expand_y=True, expand_x=False, justification='left', size=size),

                    sg.B("Suppr.", key=("del", row_counter))]],
                    
                    key=("row", row_counter))
        )
    ]

    return row

def listSuggestionWindow(sugg_values, mainWindow, verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not sugg_values:
        return
    layout = [
        [sg.Listbox(sugg_values, size=(40, 3), enable_events=True, key=('SUGG',),
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def _getFieldsLayout(extract_dict, last_info, model, X_main_dim, Y_main_dim):
    clean_names_dict = LIMS_HELPER["CLEAN_ZONE_NAMES"]
    conf_threshold = int(GUIsettings["TOOL"]["confidence_threshold"])/100   
    lineLayout = []
    global_field = 1

    added_field = LIMS_HELPER["ADDED_FIELDS"]
    product_codes = LIMS_HELPER["PRODUCTCODE_DICT"]
    
    # SPECIFIC : Add filed to find the contract

    if product_codes:
        found_product_code = extract_dict["code_produit"]["sequence"]
        if found_product_code != "":
            found_product_code = found_product_code if found_product_code in product_codes.keys() else [pc for pc in product_codes if product_codes[pc]==found_product_code][0]
        pructcode_combo = [sg.Text("Code produit", s=(25,1)), sg.Combo(list(product_codes.keys()), default_value=found_product_code,
            key=("zone", "code_produit"), expand_y=True, expand_x=False, size=(INPUT_LENGTH, 1))]
        lineLayout.append(pructcode_combo)

    # Add the rush level
    lineLayout.append([sg.T("Rush:"), sg.Push(), sg.Radio('Non', "rush_level", default=True, key=("rush_level", "0")), sg.Radio('Oui', "rush_level", key=("rush_level", "4"), default=False), sg.Push()])


    # Number of detected analysis
    i_ana = 0
    
    # Returned the tool's response for each field
    for zone, landmark_text_dict in extract_dict.items():
        if not zone in added_field: # Field filled by the technicien

            # Add a line when filed pass from global to sample specific
            if OCR_HELPER[model][zone]["global_info"] != global_field:
                lineLayout.append([sg.HorizontalSeparator()])

            global_field = OCR_HELPER[model][zone]["global_info"]
            clean_zone_name = clean_names_dict[zone]
            text = f"{clean_zone_name} : "
            sequence = landmark_text_dict["sequence"]
            
            # Propagate global field smaple by sample
            if last_info and not sequence and global_field:
                sequence = last_info[("zone", zone)]

            conf= landmark_text_dict["confidence"] # If empty must not be red
            if conf < conf_threshold or sequence=="":
                back_color = 'red'
            else: 
                back_color = sg.theme_input_background_color()

            # Special case
            if zone == "analyse":

                frameLayout = []

                frameLayout.append([sg.Push(), sg.T("Analyse", justification="left"), sg.Push()]) #, sg.T("Commentaire"), sg.Push()])

                specs = extract_dict["analyse_specification"]["sequence"] if "analyse_specification" in list(extract_dict.keys()) else ["" for _ in sequence]
                
                col = []

                i = 0
                for i, (analyse, spec) in enumerate(zip(sequence, specs)):
                    analyse = " ".join(analyse) if type(analyse) == type([]) else analyse
                    row = create_row(i, analyse, spec)
                    col.append(row)

                frameLayout.append([sg.Column(col, key="-COL-", vertical_scroll_only=False, scrollable=True, expand_y=False, size=(int(X_main_dim*0.45), int(Y_main_dim*0.3)))])
                frameLayout.append([sg.Push(), sg.B("Ajouter une analyse", key=("add",)), sg.Push()])

                lineLayout.append([sg.Frame("Analyses et commentaires", frameLayout, expand_y=True)])

                i_ana = i+1
            
            else:

                if len(sequence)>=INPUT_LENGTH:
                    lineLayout.append([sg.Text(text, s=(25,1)),
                                    sg.Multiline(sequence, background_color=back_color,
                                    key=("zone", zone), expand_y=False, expand_x=False, size=(INPUT_LENGTH, (len(sequence)//INPUT_LENGTH)+1), justification='left')])
                else :
                    lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.I(sequence, background_color=back_color,
                                key=("zone", zone), expand_y=False, expand_x=False, size=(INPUT_LENGTH, 1), justification='left')])
                                    # sg.Image(data=bio.getvalue(), key=f'image_{zone}')])

    FieldsLayout = lineLayout
    
    return i_ana, FieldsLayout
    
def _getImageLayout(image, X_main_dim, Y_main_dim):

    X_im = int(X_main_dim*0.6)
    
    searching_area = Image.fromarray(image).resize((X_im, int(X_im*1.4)))
    bio = io.BytesIO()
    searching_area.save(bio, format="PNG")
    imageLayout = [[sg.Image(data=bio.getvalue(), key=f'scan')]]
    
    return imageLayout

def _getClientContractLayout(image_dict, last_client_contract):
    
    ClientContractLayout = []
    
    client, contract = "", ""
    if "client_name" in list(image_dict.keys()):
        client = image_dict["client_name"]
    if  "contract_name" in list(image_dict.keys()):
        contract = image_dict["contract_name"]
    
    if not client:
        client = last_client_contract[0]
    if not contract:
        contract = last_client_contract[1]
        
    ClientContractLayout.append([sg.Text('ACF', size=(25, 1)) ,sg.Input(client, size=(INPUT_LENGTH, 1), enable_events=True, key=("to_save", "client_name"))])
    ClientContractLayout.append([sg.Text('Expediteur', size=(25, 1)) ,sg.Input(contract, size=(INPUT_LENGTH, 1), enable_events=True, key=("to_save", "contract_name"), background_color='light gray')])
    ClientContractLayout.append([sg.HorizontalSeparator(key='sep')])
    
    return ClientContractLayout

def ClientSuggestionWindow(values, mainWindow, verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CLIENT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def ContractSuggestionWindow(values, mainWindow,  verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not values:
        return
    layout = [
        [sg.Listbox(values, size=(50, 5), enable_events=True, key='-AUTO_CONTRACT-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def getMainLayout(image_dict, last_info, image, model, X_dim, Y_dim, last_client_contract, add=False):

    X_main_dim, Y_main_dim = int(X_dim*0.9), int(0.85*Y_dim)

    i_ana, FiledsLayout = _getFieldsLayout(image_dict, last_info, model, X_main_dim, Y_main_dim)
    ImageLayout =_getImageLayout(image, X_main_dim, Y_main_dim)
    ClientContractLayout = _getClientContractLayout(image_dict, last_client_contract)

    MainLayout = [
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push(), sg.Button("Consignes d'utilisation", k="-consignes-")],
        [sg.Push(), 
         sg.B("<- Retour", s=10), sg.B("Valider ->", s=10), 
         sg.B("Ajouter une commande manuellement", s=10, size=(35, 1)),
         sg.Push()]
    ]

    if add:
        MainLayout = [
        [sg.Push(), sg.Text("AJOUT MANUEL", font=("Helvetica", 36, "bold")), sg.Push()],
        [sg.Text("Attention : Veuillez bien vérifier les champs proposés"), sg.Push()],
        [sg.Push(), 
         sg.B("Valider ->", s=10),
         sg.Push()]
    ]
    
    MainLayout.append([sg.Column(ClientContractLayout + FiledsLayout , justification="r"), 
            sg.Column(ImageLayout, scrollable=True, justification="l", size=(int(X_dim*0.9), int(0.9*Y_dim)))])
    
    return i_ana, MainLayout

def choice_overwrite_continue_popup(general_text, red_text, green_text):
    layout = [
        [sg.Text(general_text)],
        [sg.B(red_text, k="overwrite", button_color="tomato"), sg.B(green_text, k="continue", button_color="medium sea green")] 

    ]
    window = sg.Window("Predict Risk", layout, use_default_focus=False, finalize=True, modal=True)
    event, values = window.read()
    window.close()
    return event

def get_image(scan_dict, pdf_name, sample_image_name):
    l_image = []
    for image in sample_image_name.split("+"):
        l_image.append(scan_dict[pdf_name][image])
    return np.concatenate(l_image)

def welcomeLayout():

    welcomeLayout = [
        [sg.Text("Dossier contenant les PDFs"), sg.Input(LIMS_HELPER["TOOL_PATH"]["input_folder"], key="-PATH-"), 
         sg.FolderBrowse(button_color="cornflower blue")],

        [sg.Push(), sg.Text("Client :")] + [sg.Radio(model, group_id="model", key=model) for model in LIMS_HELPER["MODELS"]] + [sg.Push()],
        
        [sg.Push(), sg.Exit(button_color="tomato"), sg.Button("Lancer l'algorithme", button_color="medium sea green"), sg.Push()]
    ]
    
    return welcomeLayout

def manually_add_order(MainLayout, res_dict, i_ana_add, MODEL_ANALYSIS, loc, dim, pdf_name):
    sugg_index_add = None
    sample_name = "echantillon_manuel_"
    num = 0
    VerificactionLayout_add = MainLayout
    VerificationWindow_add = sg.Window(f"Fiche {sample_name}",
                                VerificactionLayout_add, use_custom_titlebar=True, location=loc, 
                                size=dim, resizable=True, finalize=True, )
    SuggestionW_add = None
    deleted_rows_add = []

    while True:
        add_window, verif_event_add, verif_values_add = sg.read_all_windows()
        if verif_event_add == sg.WINDOW_CLOSED:
            return None, None, None
        
        if verif_event_add[0] in ["ana", "SUGG", "add", "del"]:
            verif_event_add, deleted_rows_add, i_ana_add, sugg_index_add, SuggestionW_add = suggestion_interaction(add_window, verif_event_add, verif_values_add, SuggestionW_add, VerificationWindow_add, deleted_rows_add, i_ana_add, sugg_index_add, MODEL_ANALYSIS)
        if verif_event_add == "Valider ->":
            # Register the response and go to the following
            sample_name_num = sample_name + str(num)
            while sample_name_num in list(res_dict["RESPONSE"][pdf_name].keys()):
                sample_name_num = sample_name + str(num)
                num+=1
            VerificationWindow_add.close()

            return verif_values_add, sample_name_num, deleted_rows_add
            # If last image

def _process_suggestion(verif_event, verif_values, ANALYSIS_LIST, VerificationWindow, SuggestionW, sugg_index):

        sugg_text = verif_values[verif_event]
        sugg_text = sugg_text[0] if type(sugg_text) == type([]) else sugg_text
        client_sugg = sorted([sugg for sugg in ANALYSIS_LIST if sugg_text.lower() in str(sugg).lower()])

        if sugg_text and client_sugg:
            if SuggestionW :
                SuggestionW.BringToFront()
                SuggestionW[("SUGG",)].update(values=client_sugg)
                SuggestionW.refresh()
            elif len(sugg_text)>1 :
                SuggestionW = listSuggestionWindow(client_sugg, VerificationWindow, verif_event)
                SuggestionW.BringToFront()              
                                                                    
        if verif_event[0] == 'SUGG':
            SuggestionW.close()
            SuggestionW = False
            text = verif_values[("SUGG",)][0]
            VerificationWindow[("ana", sugg_index)].update(value=text)

        return  verif_event, verif_values, VerificationWindow, SuggestionW

def suggestion_interaction(window, verif_event, verif_values, SuggestionW, VerificationWindow, deleted_rows, i_ana, sugg_index, MODEL_ANALYSIS):

# Suggestion
    if verif_event[0] == "del":
        deleted_rows.append(verif_event[1])
        window[("row", verif_event[1])].update(visible=False)
        window.visibility_changed()
        window.refresh()
    
    if verif_event[0] in ["ana", "SUGG"]:

        if verif_event[0] == "ana":
            index = verif_event[1]
            if SuggestionW and sugg_index!=index:
                SuggestionW.close()
                SuggestionW=None
            sugg_index = verif_event[1]

        processed = _process_suggestion(verif_event, verif_values, MODEL_ANALYSIS["Denomination"].dropna().unique(), VerificationWindow, SuggestionW, sugg_index)
        verif_event, verif_values, VerificationWindow, SuggestionW = processed
                                            
    if verif_event[0] == "add":
        row = create_row(i_ana, "", "")
        window.extend_layout(window['-COL-'], [row])
        i_ana+=1
        window['-COL-'].contents_changed()
        # window.visibility_changed()
        window.refresh()
    
    return verif_event, deleted_rows, i_ana, sugg_index, SuggestionW

def process_contract_client_action(verif_event, verif_values, CONTRACT_LIST, VerificationWindow, ClientSuggestionW, contractSuggestionW):
    if verif_event[1] == "client_name":
        client_text = verif_values[("to_save", "client_name")]
        client_sugg = sorted([client for client in CLIENT_LIST if client_text.lower() in client.lower()])
        if client_text and client_sugg:
            try:
                if ClientSuggestionW:
                    ClientSuggestionW.BringToFront()
                    ClientSuggestionW["-AUTO_CLIENT-"].update(values=client_sugg)
                    ClientSuggestionW.refresh()
                elif len(client_text)>2 :
                    ClientSuggestionW = ClientSuggestionWindow(client_sugg, VerificationWindow, verif_event)
                    ClientSuggestionW.BringToFront()             
            except:
                pass 
                                                                
    if verif_event == '-AUTO_CLIENT-':
        ClientSuggestionW.close()
        ClientSuggestionW = False
        text = verif_values['-AUTO_CLIENT-'][0]
        VerificationWindow[("to_save", "client_name")].update(value=text)
        
        CONTRACT_LIST = sorted(list(CLIENT_CONTRACT[CLIENT_CONTRACT["ClientName"]== text]["ContractName"].unique()))
        VerificationWindow[("to_save", "contract_name")].update(disabled=False, background_color=sg.theme_input_background_color())
        VerificationWindow[("to_save", "client_name")].set_focus()
        
    if verif_event[1] == "contract_name":
        contract_text = verif_values[("to_save", "contract_name")]
        contract_sugg = [contract for contract in CONTRACT_LIST if contract_text.lower() in contract.lower()]
        if contract_text and contract_sugg:
            if contractSuggestionW :
                contractSuggestionW.BringToFront()
                contractSuggestionW["-AUTO_CONTRACT-"].update(values=contract_sugg)
                contractSuggestionW.refresh()
            else :
                contractSuggestionW = ContractSuggestionWindow(contract_sugg, VerificationWindow, verif_event)
                contractSuggestionW.BringToFront()

    if verif_event == '-AUTO_CONTRACT-':
        contractSuggestionW.close()
        contractSuggestionW = False
        text = verif_values['-AUTO_CONTRACT-'][0]
        VerificationWindow[("to_save", "contract_name")].update(value=text)
        VerificationWindow[("to_save", "contract_name")].set_focus()
    
    return  CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW

def main():

    window_title = GUIsettings["GUI"]["title"]
    welcomWindow = sg.Window(window_title, welcomeLayout(), use_custom_titlebar=True)
    ClientSuggestionW, contractSuggestionW = None, None

    while True:
        event, values = welcomWindow.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Lancer l'algorithme":

            givenPath = values["-PATH-"]
            model = [model for model in LIMS_HELPER["MODELS"] if values[model]]
            
            MODEL = model[0]
            interface_model = "CU" if "CU" in MODEL else MODEL

            # If everything is valid return the scan dict
            scan_dict = is_valid(givenPath, model)

            if scan_dict:

                # Set the path to load the res.json file
                res_path = os.path.join(givenPath, "RES")
                res_save_path = os.path.join(res_path, "res.json")

                # Set the path to place the XML, if it's not exist then creat it
                xml_save_path = LIMS_HELPER["TOOL_PATH"]["output_folder"] if LIMS_HELPER["TOOL_PATH"]["output_folder"] else os.path.join(res_path, "verified_XML")
                if not os.path.exists(xml_save_path):
                    os.makedirs(xml_save_path)
                
                # Ask if the tool is going to overwrite the found result of tak eit back
                continue_or_smash = "overwrite"
                if os.path.exists(res_save_path):
                    continue_or_smash = choice_overwrite_continue_popup("Il semblerait que l'analyse ai déjà été effectuée.\nEcraser la précédente analyse ?"
                                                                , "Ecraser", "Reprendre la précédente analyse") #Change conditon

                # If the tool take last results back
                if continue_or_smash == "continue":
                    json_file  = open(res_save_path, encoding='utf-8')
                    res_dict = json.load(json_file)
                    welcomWindow.close()
                    if set(list(res_dict["RESPONSE"].keys())) != set(list(scan_dict.keys())):
                        sg.popup_ok(f"Attention : Les images précédemment analysées et celles dans le dossier ne sont pas identiques")
                
                # If the tool have to launch the extraction process
                if continue_or_smash == "overwrite":
                    sg.popup_auto_close("L'agorithme va démarrer !\nSuivez l'évolution dans le terminal", auto_close_duration=2)
                    welcomWindow.close()

                    print("_________ START _________\nAttendez la barre de chargement \nAppuyez sur ctrl+c dans le terminal pour interrompre")
                    scan_dict, res_dict = use_the_tool(givenPath, MODEL)
                    print("_________ DONE _________")

                    # Save the extraction json on RES
                    with open(res_save_path, 'w', encoding='utf-8') as json_file:
                        json.dump(res_dict, json_file,  ensure_ascii=False)
    

                # Set lists to make suggestion on the window
                MODEL_ANALYSIS = pd.read_excel(OCR_HELPER["PATHES"]["contract_analysis_path"], sheet_name="analyse "+MODEL)

                pdfs_res_dict = res_dict["RESPONSE"]

                SuggestionW = None

                is_loop_started = False # Force the verification process to be done or abandonned (no infinit loop)

                while not is_loop_started:
                    is_loop_started = True 
                    is_first_step = True

                    CONTRACT_LIST_0 = list(CLIENT_CONTRACT["ContractName"].unique())
                    CONTRACT_LIST = CONTRACT_LIST_0

                    try:
                        last_client_contract = LIMS_HELPER["BASE_CLIENT_CONTRACT"][MODEL]
                        
                    except:
                        last_client_contract = ["", ""]

                    n_pdf = 0
                    n_sample = 0
                    n_displayed= (-1, -1)
                    X_dim, Y_dim = fit_the_screen(1)
                    X_loc, Y_loc = (10,10)
                    sugg_index = None

                    # Last PDF
                    n_pdf_end = len(list(pdfs_res_dict.keys()))
                    # Last pdf, last sample
                    last_sample = (n_pdf_end-1, len(list(pdfs_res_dict[list(pdfs_res_dict.keys())[-1]].keys()))-1)

                    last_info = None

                    while n_pdf < n_pdf_end:
                        pdf_name, samples_res_dict = list(pdfs_res_dict.items())[n_pdf]
                        # last sample of the pdf
                        n_sample_end = len(samples_res_dict.keys())
                        

                        while n_sample < n_sample_end:
                            sample_name, sample_image_extract = list(samples_res_dict.items())[n_sample]
                            # Identify the current sample
                            n_place = (n_pdf, n_sample)

                            # If new displayed sample is aked
                            if n_place != n_displayed:
                                
                                # Close the window if a new one is going to be display
                                if is_first_step == False :
                                    X_loc, Y_loc = VerificationWindow.current_location()
                                    X_dim, Y_dim = fit_the_screen(10)
                                    VerificationWindow.close()
                                
                                # Get all information to display
                                image =  get_image(scan_dict, pdf_name, sample_image_extract["IMAGE"])
                                extract_dict = sample_image_extract["EXTRACTION"]
                                # Create the new window
                                i_ana, VerificactionLayout = getMainLayout(extract_dict, last_info, image, interface_model, X_dim, Y_dim, last_client_contract=last_client_contract)
                                VerificationWindow = sg.Window(f"PDF {pdf_name} - Commande ({sample_name})", 
                                                            VerificactionLayout, use_custom_titlebar=True, location=(X_loc, Y_loc), 
                                                            size=(X_dim, Y_dim), resizable=True, finalize=True)
                                # Set displayed sample index
                                n_displayed = n_place
                            
                            is_first_step = False
                            deleted_rows = []

                            while n_place == n_displayed:

                                window, verif_event, verif_values = sg.read_all_windows()

                                # print(verif_event, verif_values)

                                if verif_event == sg.WINDOW_CLOSED:
                                    return

                                if verif_event == "-consignes-":
                                    sg.popup_scrolled(GUIsettings["UTILISATION"]["texte"], title="Consignes d'utilisation", size=(50,10))
                                
                                # Client and contract case
                                if verif_event in [("to_save", "client_name"), '-AUTO_CLIENT-', ("to_save", "contract_name"), '-AUTO_CONTRACT-']:

                                    processed = process_contract_client_action(verif_event, verif_values, CONTRACT_LIST, VerificationWindow, ClientSuggestionW, contractSuggestionW)
                                    CONTRACT_LIST, verif_event, verif_values, VerificationWindow, ClientSuggestionW, contractSuggestionW = processed
                                
                                # Analysis case
                                if verif_event[0] in ["ana", "SUGG", "add", "del"]:
                                    verif_event, deleted_rows, i_ana, sugg_index, SuggestionW = suggestion_interaction(window, verif_event, verif_values, SuggestionW, VerificationWindow, deleted_rows, i_ana, sugg_index, MODEL_ANALYSIS)

                                if verif_event == "Ajouter une commande manuellement":
                                    # Disable main sugg windows
                                    VerificationWindow.Disable()
                                    VerificationWindow.SetAlpha(0.5)

                                    # Generate the add layout
                                    i_ana_add, VerfifLayout_add = getMainLayout(extract_dict, last_info, image, interface_model, X_dim, Y_dim, add=True)
                                    loc, dim = (X_loc+30, Y_loc),(X_dim, Y_dim)

                                    verif_values_add, sample_name_add, deleted_rows_add = manually_add_order(VerfifLayout_add, res_dict, i_ana_add, MODEL_ANALYSIS, loc, dim, pdf_name)

                                    if verif_values_add:
                                        res_dict["RESPONSE"][pdf_name][sample_name_add] = deepcopy(res_dict["RESPONSE"][pdf_name][sample_name])
                                        runningSave(res_dict, res_save_path, verif_values_add, pdf_name, sample_name_add, deleted_rows_add)
                                    VerificationWindow.Enable()
                                    VerificationWindow.SetAlpha(1)

                                # Return to the past sample
                                if verif_event == "<- Retour":
                                    # Can go before the firt sample
                                    if n_pdf>0 or n_sample>0:
                                        runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name, deleted_rows)
                                        # Go to the past sample
                                        n_sample-=1
                                        # If it's from the past pdf change the pdf number
                                        if n_pdf>0 and n_sample==-1:
                                            n_pdf-=1
                                            n_sample = len(list(pdfs_res_dict.items())[n_pdf][1].keys())-1
                                            n_place = (n_pdf, n_sample)
                                        n_place = (n_pdf, n_sample)                                         
                                
                                # Go to next
                                if verif_event == "Valider ->":
                                    if SuggestionW : SuggestionW.close()
                                    if ClientSuggestionW : ClientSuggestionW.close()
                                    # Not last image
                                    last_info = verif_values
                                    last_client_contract = [verif_values[("to_save", "client_name")], verif_values[("to_save", "contract_name")]]
                                    if not n_place == last_sample:
                                        runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name, deleted_rows)
                                        n_sample+=1
                                        n_place = (n_pdf, n_sample)
                                    # If last image
                                    else:
                                        final_dict = runningSave(res_dict, res_save_path, verif_values, pdf_name, sample_name, deleted_rows)

                                        choice = sg.popup_ok("Il n'y a pas d'image suivante. Finir l'analyse ?", button_color="dark green")
                                        if choice == "OK":
                                            finalSaveDict(final_dict["RESPONSE"], xml_save_path, analysis_lims=MODEL_ANALYSIS, model=MODEL, lims_helper=LIMS_HELPER,
                                                            client_contract=CLIENT_CONTRACT)
                                            if LIMS_HELPER["TOOL_PATH"]["copy_folder"]:
                                                saveToCopyFolder(LIMS_HELPER["TOOL_PATH"]["copy_folder"], os.path.join(givenPath, pdf_name+".pdf"), rename=pdf_name+"AA")
                                            json_file.close() # Close the file
                                            VerificationWindow.close()
                                            return
                        
                        # If n_sample exceed the n_sample_end, go to the next one from the folowing pdf
                        if verif_event == "Valider ->":
                            n_sample = 0
                            n_pdf+=1

                if VerificationWindow :  VerificationWindow.close()
                if SuggestionW : SuggestionW.close()
    
    welcomWindow.close()               
                         
if __name__ == "__main__":

    print("Attendez quelques instants, une page va s'ouvrir")
    
    # Get the base path if executable or launch directly
    if 'AppData' in sys.executable:
        application_path = os.getcwd()

    else : 
        application_path = os.path.dirname(sys.executable)

    # Load helper
    OCR_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\OCR_CONFIG.json")
    OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH, encoding='utf-8'))

    LIMS_HELPER_JSON_PATH  = os.path.join(application_path, "CONFIG\LIMS_CONFIG.json")
    LIMS_HELPER= json.load(open(LIMS_HELPER_JSON_PATH, encoding='utf-8'))

    CLIENT_CONTRACT = pd.read_excel(os.path.join(application_path, OCR_HELPER["PATHES"]["contract_analysis_path"]), sheet_name='client_contract')
    clientName, contractName = CLIENT_CONTRACT.iloc[0]["ClientName"], CLIENT_CONTRACT.iloc[0]["ContractName"]
    row = CLIENT_CONTRACT[(CLIENT_CONTRACT["ClientName"]==clientName) & (CLIENT_CONTRACT["ContractName"]==contractName)]
    CLIENT_LIST = list(CLIENT_CONTRACT["ClientName"].unique())
    
    # Set interface's graphical settings
    GUIsettings = sg.UserSettings(path=os.path.join(application_path, "CONFIG"), filename="GUI_CONFIG.ini", use_config_file=True, convert_bools_and_none=True)

    theme = GUIsettings["GUI"]["theme"]
    font_family = GUIsettings["GUI"]["font_family"]
    font_size = int(GUIsettings["GUI"]["font_size"])
    help_text = GUIsettings["UTILISATION"]["text"]    
    sg.theme(theme)
    sg.set_options(font=(font_family, font_size))
    INPUT_LENGTH = 45

    main()
