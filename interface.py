import PySimpleGUI as sg
import os
import shutil
import numpy as np
import io
import dicttoxml
import json
import pandas as pd
from copy import deepcopy
from datetime import datetime

from screeninfo import get_monitors
from PIL import Image

OCR_HELPER_JSON_PATH  = r"CONFIG\OCR_config.json"
OCR_HELPER = json.load(open(OCR_HELPER_JSON_PATH))

from LaunchTool import getAllImages, TextCVTool

MODELS = ["Nutricia", "RoyalCanin"]

def is_valid(folderpath, model):

    if len(model) != 1:
        sg.popup_error("Veuillez cocher un model")
        return False
    if not (folderpath and os.path.exists(folderpath)):
        sg.popup_error("Le dossier n'existe pas")
        return False
    
    return True

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
    size = (INPUT_LENGTH-10, 1)
    row = [
        sg.pin(
            sg.Col([[sg.I(analyse,
                        key=f"-ana_{row_counter}-", enable_events=True, expand_y=True, expand_x=False, justification='left', size=size),
                        
                    sg.I(spec,
                        key=f"-spec_{row_counter}-", enable_events=False, expand_y=True, expand_x=False, justification='left', size=size),

                    sg.B("Suppr.", key=f"del_{row_counter}")]],
                    
                    key=f"-row_{row_counter}-")
        )
    ]

    return row

def listSuggestionWindow(sugg_values, mainWindow, verif_event):
    x = mainWindow[verif_event].Widget.winfo_rootx()
    y = mainWindow[verif_event].Widget.winfo_rooty() + 25
    if not sugg_values:
        return
    layout = [
        [sg.Listbox(sugg_values, size=(40, 3), enable_events=True, key='-SUGG-',
            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)]]
    return sg.Window('Title', layout, no_titlebar=True, keep_on_top=True,
        location=(x, y), margins=(0, 0), finalize=True)

def _getFieldsLayout(extract_dict, last_info, MODEL, X_main_dim, Y_main_dim):
    conversion_dict = LIMSsettings["CLEAN_ZONE"]
    conf_threshold = int(GUIsettings["TOOL"]["confidence_threshold"])/100   
    lineLayout = []
    global_field = 1

    # Number of detected analysis
    i_ana = 0

    # Returned the tool's response for each field
    for zone, landmark_text_dict in extract_dict.items():
        if not zone in ["analyse_specification"]: # Field filled by the technicien

            # Add a line when filed pass from global to sample specific
            if OCR_HELPER[MODEL][zone]["global_info"] != global_field:
                lineLayout.append([sg.HorizontalSeparator(key='sep')])

            global_field = OCR_HELPER[MODEL][zone]["global_info"]
            clean_zone_name = conversion_dict[zone]
            text = f"{clean_zone_name} : "
            sequence = landmark_text_dict["sequence"]
            
            # Propagate global field smaple by sample
            if last_info and not sequence and global_field:
                sequence = last_info[f"-{zone}-"]

            conf= landmark_text_dict["confidence"] # If empty must not be red
            if conf < conf_threshold or sequence=="":
                back_color = 'red'
            else: 
                back_color = sg.theme_input_background_color()

            # Special case
            if zone == "analyse":

                frameLayout = []

                frameLayout.append([sg.Push(), sg.T("Analyse", justification="left"), sg.Push(), sg.T("Commentaire"), sg.Push()])

                specs = extract_dict["analyse_specification"]["sequence"] if "analyse_specification" in list(extract_dict.keys()) else ["" for _ in sequence]
                
                col = []

                for i, (analyse, spec) in enumerate(zip(sequence, specs)):
                    row = create_row(i, analyse, spec)
                    col.append(row)

                frameLayout.append([sg.Column(col, key="-COL-", vertical_scroll_only=True, scrollable=True, expand_y=False, size=(int(X_main_dim*0.45), int(Y_main_dim*0.3)))])
                frameLayout.append([sg.Push(), sg.B("Ajouter une analyse"), sg.Push()])

                lineLayout.append([sg.Frame("Analyses et commentaires", frameLayout, expand_y=True)])

                i_ana = i+1
            
            else:

                if len(sequence)>=INPUT_LENGTH:
                    lineLayout.append([sg.Text(text, s=(25,1)),
                                    sg.Multiline(sequence, background_color=back_color,
                                    key=f"-{zone}-", expand_y=False, expand_x=False, size=(INPUT_LENGTH, (len(sequence)//INPUT_LENGTH)+1), justification='left')])
                else :
                    lineLayout.append([sg.Text(text, s=(25,1)),
                                sg.I(sequence, background_color=back_color,
                                key=f"-{zone}-", expand_y=False, expand_x=False, size=(INPUT_LENGTH, 1), justification='left')])
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

def getMainLayout(image_dict, last_info, image, MODEL, X_dim, Y_dim, add=False):

    X_main_dim, Y_main_dim = int(X_dim*0.9), int(0.85*Y_dim)

    i_ana, FiledsLayout = _getFieldsLayout(image_dict, last_info, MODEL, X_main_dim, Y_main_dim)
    ImageLayout =_getImageLayout(image, X_main_dim, Y_main_dim)
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
    
    MainLayout.append([sg.Column(FiledsLayout , justification="r"),
            sg.Column(ImageLayout, scrollable=True, justification="l", size=(X_main_dim, Y_main_dim))])
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

def process_suggestion(verif_event, verif_values, ANALYSIS_LIST, VerificationWindow, SuggestionW, sugg_index):

    sugg_text = verif_values[verif_event]
    sugg_text = sugg_text[0] if type(sugg_text) == type([]) else sugg_text
    client_sugg = sorted([sugg for sugg in ANALYSIS_LIST if sugg_text.lower() in sugg.lower()])

    if sugg_text and client_sugg:
        if SuggestionW :
            SuggestionW.BringToFront()
            SuggestionW["-SUGG-"].update(values=client_sugg)
            SuggestionW.refresh()
        elif len(sugg_text)>2 :
            SuggestionW = listSuggestionWindow(client_sugg, VerificationWindow, verif_event)
            SuggestionW.BringToFront()              
                                                                
    if verif_event == '-SUGG-':
        SuggestionW.close()
        SuggestionW = False
        text = verif_values['-SUGG-'][0]
        VerificationWindow[f"-ana_{sugg_index}-"].update(value=text)

    return  verif_event, verif_values, VerificationWindow, SuggestionW

def runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows):

    analyses, comments = [], []

    for key, items in verif_values.items():
        if key[0] == "-":
            if "ana" in key:
                index = key.strip("-").split("_")[1]
                if not index in deleted_rows and items:

                    analyses.append(items)
                    comments.append(verif_values[f"-spec_{index}-"])

            elif "spec" in key:
                continue

            else:
                res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"][key.strip("-")]["sequence"] = items

    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse"]["sequence"] = analyses
    res_dict["RESPONSE"][pdf_name][sample_name]["EXTRACTION"]["analyse_specification"] = {"sequence" : comments}

    with open(save_path_json, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f,  ensure_ascii=False)
    
    return res_dict

def convertDictToLIMS(verified_dict, code_lims): 
    
    def _package_test(analysis, specs, code_lims):
        package_codes, test_codes = [], []

        for analyse, spec in zip(analysis, specs):
            code = code_lims[code_lims["analyse"]==analyse]["code_lims"].values.tolist()[0]
            if code[0] == "P":
                package_codes.append(code)
            else:
                test_codes.append(code)

        return package_codes, test_codes
    
    # Clean fields
    scan_clean_dict = {}
    for key, value  in verified_dict.items():
        if not key in ["analyse", "analyse_specification"]:
            scan_clean_dict[key] = value["sequence"]

    package_code, test_code = _package_test(verified_dict["analyse"]["sequence"], verified_dict["analyse_specification"]["sequence"], code_lims)
    if package_code:
        scan_clean_dict["packagecode"] = package_code
    if test_code:
        scan_clean_dict["testcode"] = test_code
        
    # Convert client and contract for the LIMS
    stack_dict = {}
    stack_dict.update({"root" : {}})
    stack_dict.update({"Sample" : {}})
    stack_dict.update({"AdditionalField" : {}})

    for key, name_code in LIMSsettings["LIMS_CONVERTER"].items():
        if key in list(scan_clean_dict.keys()):
            dict_name, code = name_code
            stack_dict[dict_name][code] = scan_clean_dict[key]

    stack_dict["Sample"]["AdditionalField"] = stack_dict["AdditionalField"]
    stack_dict["root"]["Sample"] = stack_dict["Sample"]
    return stack_dict["root"]

def finalSaveDict(verified_dict, xml_save_path, code_lims, out_path="", xml_name="verified_XML"):

    def _copy_dict(verified_dict):
        xml_dict = {}
        num = 1
        for pdf_name, sample_dict in verified_dict.items():
            for sample, res_dict in sample_dict.items():
                xml_name = datetime.now().strftime("%Y%m%d%H")
                clean_dict = convertDictToLIMS(res_dict["EXTRACTION"], code_lims)

                sample_num_id = xml_name+f"_{num}"
                while sample_num_id in list((xml_dict).keys()):
                    num+=1
                    sample_num_id = xml_name+f"_{num}"
                xml_dict[sample_num_id] = deepcopy(clean_dict)
                # res_dict[scan_name] = clean_dict

        return xml_dict
    
    if out_path:
        new_xml = os.path.join(out_path, xml_name)
        if not os.path.exists(new_xml):
            os.makedirs(new_xml)
    
    xml_dict = _copy_dict(verified_dict)
    xml_dict = dict(sorted(xml_dict.items()))

    for scan_name, scan_dict in xml_dict.items():
        xml = dicttoxml.dicttoxml(scan_dict)
        with open(os.path.join(xml_save_path, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
            result_file.write(xml.decode())
        if out_path:
            with open(os.path.join(new_xml, f"{scan_name}.xml"), 'w', encoding='utf8') as result_file:
                result_file.write(xml.decode())

def welcomeLayout():

    welcomeLayout = [
        [sg.Text("Dossier contenant les PDFs"), sg.Input(LIMSsettings["TOOL_PATH"]["input_folder"], key="-PATH-"), 
         sg.FolderBrowse(button_color="cornflower blue")],

        [sg.Push(), sg.Text("Client :")] + [sg.Radio(model, group_id="model", key=model) for model in MODELS] + [sg.Push()],
        
        [sg.Push(), sg.Exit(button_color="tomato"), sg.Button("Lancer l'algorithme", button_color="medium sea green"), sg.Push()]

    ]
    
    return welcomeLayout

def manually_add_order(MainLayout, res_dict, i_ana, X_loc, Y_loc, X_dim, Y_dim):
    sample_name = "echantillon_manuel_"
    num = 0
    VerificactionLayout_add = MainLayout
    VerificationWindow_add = sg.Window(f"Fiche {sample_name}",
                                VerificactionLayout_add, use_custom_titlebar=True, location=(X_loc+30, Y_loc), 
                                size=(X_dim, Y_dim), resizable=True, finalize=True, )
    deleted_rows_add = []
    while True:
        add_windows, verif_event_add, verif_values_add = sg.read_all_windows()

        if verif_event_add == sg.WINDOW_CLOSED:
            return None, None, None
        
        if "del" in verif_event_add:
            deleted_rows_add.append(verif_event_add.split("_")[1])
            add_windows[f'-row_{verif_event_add.split("_")[1]}-'].update(visible=False)
            add_windows.visibility_changed()
            add_windows.refresh()
        
        if verif_event_add == "Ajouter une analyse":
            add_windows.extend_layout(add_windows['-COL-'], [create_row(i_ana, "", "")])
            i_ana+=1
            add_windows['-COL-'].contents_changed()
            add_windows.visibility_changed()
            add_windows.refresh()


        if verif_event_add == "Valider ->":
            # Register the response and go to the following
            sample_name_num = sample_name + str(num)
            while sample_name_num in list(res_dict["RESPONSE"].keys()):
                sample_name_num = sample_name + str(num)
                num+=1
            VerificationWindow_add.close()

            return verif_values_add, sample_name_num, deleted_rows_add
            # If last image

def main():

    window_title = GUIsettings["GUI"]["title"]
    welcomWindow = sg.Window(window_title, welcomeLayout(), use_custom_titlebar=True)
    while True:
        event, values = welcomWindow.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Lancer l'algorithme":
            givenPath = values["-PATH-"]
            model = [model for model in MODELS if values[model]]
            if is_valid(givenPath, model):
                MODEL = model[0]
                scan_dict = getAllImages(givenPath)

                if scan_dict == {}:
                    sg.popup_ok("Aucun PDF n'est trouvé dans le dossier")

                else :
                    res_path = os.path.join(givenPath, "RES")
                    save_path_json = os.path.join(res_path, "res.json")
                    xml_res_path = os.path.join(res_path, "verified_XML")
                    start = False # Set to true when scans are processed ; condition to start the verification process
                    end = False # Force the verification process to be done or abandonned (no infinit loop)
                    if os.path.exists(save_path_json):
                        continue_or_smash = choice_overwrite_continue_popup("Il semblerait que l'analyse ai déjà été effectuée.\nEcraser la précédente analyse ?"
                                                                   , "Ecraser", "Reprendre la précédente analyse") #Change conditon
                    else : continue_or_smash = "overwrite"

                    if continue_or_smash==None : pass
                    if continue_or_smash == "continue":
                        json_file  = open(save_path_json, encoding='utf-8')
                        res_dict = json.load(json_file)
                        welcomWindow.close()
                        start = True

                        if set(list(res_dict["RESPONSE"].keys())) != set(list(scan_dict.keys())):
                            sg.popup_ok(f"Attention : Les images précédemment analysées et celles dans le dossier ne sont pas identiques")

                    if continue_or_smash == "overwrite":
                        sg.popup_auto_close("L'agorithme va démarrer !\nSuivez l'évolution dans le terminal", auto_close_duration=2)
                        welcomWindow.close()
                        print("_________ START _________\nAttendez la barre de chargement \nAppuyez sur ctrl+c dans le terminal pour interrompre")

                        scan_dict, res_dict = use_the_tool(givenPath, MODEL)

                        print("_________ DONE _________")

                        with open(save_path_json, 'w', encoding='utf-8') as json_file:
                            json.dump(res_dict, json_file,  ensure_ascii=False) # Save the extraction json on RES
                        if os.path.exists(xml_res_path): # Create or overwrite the verified_XML folder in RES
                            shutil.rmtree(xml_res_path)

                        start = True

                    if not os.path.exists(xml_res_path):
                        os.makedirs(xml_res_path) 
                    
                    MODEL_LISTS = pd.read_excel(r"CONFIG\\lists.xlsx", sheet_name=MODEL)

                    pdfs_res_dict = res_dict["RESPONSE"]

                    SuggestionW = None

                    while start and not end:
                        end = True

                        n_pdf = 0
                        n_sample = 0
                        n_displayed= (-1, -1)
                        X_dim, Y_dim = fit_the_screen(1)
                        X_loc, Y_loc = (10,10)
                        start = False

                        sugg_index = None

                        # Last PDF, last sample
                        n_end = (len(list(pdfs_res_dict.keys()))-1, len(list(pdfs_res_dict[list(pdfs_res_dict.keys())[-1]].keys()))-1)

                        last_info = None

                        while n_pdf < n_end[0]+1:
                            pdf_name, samples_res_dict = list(pdfs_res_dict.items())[n_pdf]

                            while n_sample < n_end[1]+1:
                                sample_name, sample_image_extract = list(samples_res_dict.items())[n_sample]
                                n_place = (n_pdf, n_sample)
                                if n_place != n_displayed:

                                    if start == True :
                                        X_loc, Y_loc = VerificationWindow.current_location()
                                        X_dim, Y_dim = fit_the_screen(10)
                                        
                                    image =  get_image(scan_dict, pdf_name, sample_image_extract["IMAGE"])

                                    extract_dict = sample_image_extract["EXTRACTION"]

                                    i_ana, VerificactionLayout = getMainLayout(extract_dict, last_info, image, MODEL, X_dim, Y_dim)

                                    if start == True :
                                        VerificationWindow.close()
                                    VerificationWindow = sg.Window(f"PDF {pdf_name} - Commande ({sample_name})", 
                                                                VerificactionLayout, use_custom_titlebar=True, location=(X_loc, Y_loc), 
                                                                size=(X_dim, Y_dim), resizable=True, finalize=True)
                                    n_displayed = n_place
                                
                                start = True
                                deleted_rows = []

                                while n_place == n_displayed:

                                    window, verif_event, verif_values = sg.read_all_windows()

                                    # print(verif_event, verif_values)

                                    if verif_event == sg.WINDOW_CLOSED:
                                        return

                                    if verif_event == "-consignes-":
                                        sg.popup_scrolled(GUIsettings["UTILISATION"]["texte"], title="Consignes d'utilisation", size=(50,10))
                                    
                                    if "del" in verif_event:
                                        deleted_rows.append(verif_event.split("_")[1])
                                        window[f'-row_{verif_event.split("_")[1]}-'].update(visible=False)
                                        window.visibility_changed()
                                        window.refresh()
                                    
                                    # Suggestion
                                    if verif_event.startswith(("-ana_", "-SUGG-")):

                                        if verif_event.startswith("-ana"):
                                            index = verif_event.strip("-").split("_")[1]
                                            if SuggestionW and sugg_index!=index:
                                                SuggestionW.close()
                                                SuggestionW=None
                                            sugg_index = verif_event.strip("-").split("_")[1]

                                        processed = process_suggestion(verif_event, verif_values, MODEL_LISTS["analyse"], VerificationWindow, SuggestionW, sugg_index)
                                        verif_event, verif_values, VerificationWindow, SuggestionW = processed
                                                                            
                                    if verif_event == "Ajouter une analyse":
                                        row = create_row(i_ana, "", "")
                                        window.extend_layout(window['-COL-'], [row])
                                        i_ana+=1
                                        window['-COL-'].contents_changed()
                                        # window.visibility_changed()
                                        window.refresh()

                                    if verif_event == "Ajouter une commande manuellement":
                                        VerificationWindow.Disable()
                                        VerificationWindow.SetAlpha(0.5)
                                        i_ana, addLayout = getMainLayout(extract_dict, last_info, image, MODEL, X_dim, Y_dim, add=True)
                                        verif_values_add, sample_name_add, deleted_rows_add = manually_add_order(addLayout, res_dict, X_loc, Y_loc, X_dim, Y_dim)
                                        if verif_values_add:
                                            res_dict["RESPONSE"][pdf_name][sample_name_add] = deepcopy(res_dict["RESPONSE"][pdf_name][sample_name])
                                            runningSave(res_dict, save_path_json, verif_values_add, pdf_name, sample_name_add, deleted_rows_add)
                                        VerificationWindow.Enable()
                                        VerificationWindow.SetAlpha(1)

                                    if verif_event == "<- Retour":
                                        if n_pdf>0 or n_sample>0:
                                            runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name)
                                            n_sample-=1
                                            if n_pdf>0 and n_sample==-1:
                                                n_pdf-=1
                                                n_sample = 0
                                                n_place = (n_pdf, n_sample)                                     

                                    if verif_event == "Valider ->":
                                        if SuggestionW : SuggestionW.close()
                                        # Not last image
                                        last_info = verif_values
                                        if not n_place == n_end:
                                            runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows)
                                            n_sample+=1
                                            n_place = (n_pdf, n_sample)
                                        # If last image
                                        else:
                                            final_dict = runningSave(res_dict, save_path_json, verif_values, pdf_name, sample_name, deleted_rows)
                                            choice = sg.popup_ok("Il n'y a pas d'image suivante. Finir l'analyse ?", button_color="dark green")
                                            if choice == "OK":
                                                finalSaveDict(final_dict["RESPONSE"], xml_res_path, code_lims=MODEL_LISTS, out_path=LIMSsettings["TOOL_PATH"]["output_folder"])
                                                json_file.close() # Close the file
                                                VerificationWindow.close()
                                                return

                            n_sample = 0
                            n_pdf+=1

                VerificationWindow.close()
                if SuggestionW : SuggestionW.close()
    
    welcomWindow.close()               
                         
if __name__ == "__main__":
    print("Attendez quelques instants, une page va s'ouvrir")
    
    SETTINGS_PATH = os.getcwd()
    GUIsettings = sg.UserSettings(path=os.path.join(SETTINGS_PATH, "CONFIG"), filename="GUI_config.ini", use_config_file=True, convert_bools_and_none=True)
    theme = GUIsettings["GUI"]["theme"]
    font_family = GUIsettings["GUI"]["font_family"]
    font_size = int(GUIsettings["GUI"]["font_size"])
    help_text = GUIsettings["UTILISATION"]["text"]    
    sg.theme(theme)
    sg.set_options(font=(font_family, font_size))
    
    # sys.path.append(r"CONFIG")
    OCR_HELPER = json.load(open(r"CONFIG\OCR_config.json"))
    LIMSsettings = json.load(open(r"CONFIG\LIMS_config.json"))
    INPUT_LENGTH = 45

    main()
