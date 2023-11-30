import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import fasttext
from collections import Counter
from engine import Model
def load_data(json_path):
    """
    Load data from a JSON file into a Pandas DataFrame.

    Parameters:
    json_path (str): The file path to the JSON file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the JSON file.
    """
    return pd.read_json(json_path)


def remove_duplicates(text_data):
    """
    Identify and remove duplicate text entries from the dataset, both case-sensitive and case-insensitive.

    Parameters:
    text_data (list): A list of text strings.

    Returns:
    tuple: Two dictionaries containing indices of duplicate entries. The first dictionary is for 
           case-sensitive duplicates and the second for case-insensitive duplicates.
    """
    case_sensitive_duplicates = defaultdict(list)
    case_insensitive_duplicates = defaultdict(list)
    for idx, text in enumerate(text_data):
        case_insensitive_duplicates[text.lower()].append(idx)
        case_sensitive_duplicates[text].append(idx)
    
    return case_sensitive_duplicates, case_insensitive_duplicates


def clean_text(text_data):
    """
    Clean the text data by removing HTML tags, special characters, and extra whitespaces.

    Parameters:
    text_data (list): A list of text strings, possibly containing HTML and other noise.

    Returns:
    list: A list of cleaned text strings.
    """
    cleaned_text_list = []
    
    for text in tqdm(text_data):
        try:
            cleaned_text = BeautifulSoup(text, "html.parser").get_text()
        except:
            cleaned_text = text
        cleaned_text = re.sub(r'\xa0', ' ', cleaned_text)
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)
        cleaned_text = re.sub(r'\t', ' ', cleaned_text)
        cleaned_text = re.sub(r'\r', ' ', cleaned_text)
        cleaned_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        cleaned_text = cleaned_text.lower()
        cleaned_text_list.append(cleaned_text)

    return cleaned_text_list


def identify_languages(text_data, pretrained_lang_model="lid218e.bin"):
    """
    Identify the language of each text string in the dataset using a pre-trained FastText model.

    Parameters:
    text_data (list): A list of text strings.
    pretrained_lang_model (str): File path to the pre-trained FastText language model. Default is "lid218e.bin".

    Returns:
    tuple: Three dictionaries mapping languages to indices, languages to text, and a list of detected languages.
    """
    model = fasttext.load_model(pretrained_lang_model)
    
    language_to_idx_mapping_dict = defaultdict(list)
    language_to_text_mapping_dict = defaultdict(list)
    
    for idx, text in enumerate(tqdm(text_data)):
        predictions = model.predict(text, k=1)
        lang = predictions[0][0].replace('__label__', '')
        language_to_idx_mapping_dict[lang].append(idx)
        language_to_text_mapping_dict[lang].append(text)
    
    return language_to_idx_mapping_dict, language_to_text_mapping_dict



def main():
    """
    Main function to process a dataset from a JSON file. It performs the following operations:
    - Reads data from a JSON file into a DataFrame.
    - Cleans the text data.
    - Identifies languages in the text data.
    - Processes the data for translation and sentiment analysis.
    - Outputs the processed data to a JSON file.
    """
    # Specify the path to your JSON file
    json_file_path = "/home/users/kinjals/work_dir/check/_scratchj_ehealth_Education_raw_data_pendrives_drive2_NO NAME_21.08.2023_20230821161427.pdf.json"
    # Open the file and load its content
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    df=pd.DataFrame()
    l1=[]
    l2=[]
    from tqdm import tqdm
    for key,value in tqdm(data.items()):
        l1.append(key)
        l2.append(value)
    df['fname']=l1
    df['text']=l2
    big_email_data=df
    clean_email_text = clean_text(list(df['text']))
    big_email_data["Cleaned Text"] = clean_email_text
    os.chdir("/home/users/kinjals/work_dir/IndicLID/Inference/ai4bharat")
    IndicLID_model = IndicLID(input_threshold = 0.5, roman_lid_threshold = 0.6)
    batch_size = 1
    languages = []
    for i in tqdm(clean_email_text):
        outputs = IndicLID_model.batch_predict([i], batch_size)
        languages.append(outputs[0][1])
    big_email_data["Language"] = languages
    indices = big_email_data[big_email_data['Language'].str.contains('Latn') & ~big_email_data['Language'].str.contains('eng_Latn')].index
    big_email_data.loc[list(indices), "Language"] = 'eng_Latn'
    lang_str = """
    | Assamese (Bengali script) | asm_Beng |  
    | Assamese (Latin script) | asm_Latn |  
    | Bangla (Bengali script) | ben_Beng |  
    | Bangla (Latin script) | ben_Latn |  
    | Bodo (Devanagari script) | brx_Deva |  
    | Bodo (Latin script) | brx_Latn |  
    | Dogri (Devanagari script) | doi_Deva |  
    | Dogri (Latin script) | doi_Latn | 
    | English (Latin script) | eng_Latn |  
    | Gujarati (Gujarati script) | guj_Gujr |  
    | Gujarati (Latin script) | guj_Latn |  
    | Hindi (Devanagari script) | hin_Deva |  
    | Hindi (Latin script) | hin_Latn |  
    | Kannada (Kannada script) | kan_Knda |  
    | Kannada (Latin script) | kan_Latn |  
    | Kashmiri (Perso_Arabic script) | kas_Arab |  
    | Kashmiri (Devanagari script) | kas_Deva |  
    | Kashmiri (Latin script) | kas_Latn |  
    | Konkani (Devanagari script) | kok_Deva |  
    | Konkani (Latin script) | kok_Latn |  
    | Maithili (Devanagari script) | mai_Deva |  
    | Maithili (Latin script) | mai_Latn |  
    | Malayalam (Malayalam script) | mal_Mlym |  
    | Malayalam (Latin script) | mal_Latn |  
    | Manipuri (Bengali script) | mni_Beng |  
    | Manipuri (Meetei_Mayek script) | mni_Meti |  
    | Manipuri (Latin script) | mni_Latn |  
    | Marathi (Devanagari script) | mar_Deva |  
    | Marathi (Latin script) | mar_Latn |  
    | Nepali (Devanagari script) | nep_Deva |  
    | Nepali (Latin script) | nep_Latn |  
    | Oriya (Oriya script) | ori_Orya |  
    | Oriya (Latin script) | ori_Latn |  
    | Punjabi (Gurmukhi script) | pan_Guru |  
    | Punjabi (Latin script) | pan_Latn |  
    | Sanskrit (Devanagari script) | san_Deva |  
    | Sanskrit (Latin script) | san_Latn |  
    | Santali (Ol_Chiki  script) | sat_Olch |  
    | Sindhi (Perso_Arabic script) | snd_Arab |  
    | Sindhi (Latin script) | snd_Latn |  
    | Tamil (Tamil script) | tam_Tamil |  
    | Tamil (Latin script) | tam_Latn |  
    | Telugu (Telugu script) | tel_Telu |  
    | Telugu (Latin script) | tel_Latn |  
    | Urdu (Perso_Arabic script) | urd_Arab |  
    | Urdu (Latin script) | urd_Latn |  
    | Other | other |
    """
    lang_arr = lang_str.split("|")
    lang_arr = lang_arr[:-1]
    lang_map = dict()
    for i in range(len(lang_arr)//3):
        lang = lang_arr[3*i+1]
        lang_code = lang_arr[3*i+2]
        lang_map[lang_code.strip()] = lang.strip()
    lang_from_code = []
    for i in big_email_data["Language"]:
        lang_from_code.append(lang_map[i])
    big_email_data["Original Language"] = lang_from_code
    model = Model("/scratchj/ehealth/Education/models/indic-en-preprint/fairseq_model", model_type="fairseq")
    languages = big_email_data["Language"]
    clean_feedback_text = big_email_data["Cleaned Text"]
    tgt_lang = "eng_Latn"
    outputs = []
    for idx in tqdm(range(len(clean_feedback_text))):
        if languages[idx] == "eng_Latn":
            outputs.append(clean_feedback_text[idx])
        else:
            text = clean_feedback_text[idx]
            src_lang = languages[idx]
            N = 25  
            words = text.split()
            chunks = []
            for i in range(0, len(words), N):
                chunk = " ".join(words[i:i+N])
                chunks.append(chunk)
            try:
                outputs.append(model.batch_translate(chunks, src_lang, tgt_lang))
            except:
                outputs.append(None)
    for idx, i in enumerate(tqdm(outputs)):
        if i is None:
            outputs[idx] = "None"
        if isinstance(i, list):
            if len(i) != 1:
                st = ""
                for j in i:
                    st += j
                outputs[idx] = st
            else:
                outputs[idx] = i[0]
    big_email_data["Final Text"] = outputs
    device = torch.device('cuda') 
    model = CrossEncoder('cross-encoder/nli-deberta-base', device=device)
    ucc_hypotheses = ["civil code is good",
                    "civil code is beneficial",
                    "support civil code",
                    "civil code is necessary",
                    "implement one civil code",
                    "in favour of civil code",
                    "uniform law is good",
                    "uniform law is beneficial",
                    "support uniform law",
                    "in favour of uniform law",
                    "uniform law is necessary",
                    "implement one law"
                    "ucc is good",
                    "ucc is beneficial",
                    "Support ucc",
                    "ucc is necessary",
                    "implement ucc"
                    "in favour of ucc",
                    "one law for everyone",
                    ]
    text = list(big_email_data["Final Text"])
    big_labels_list = []
    cache = dict()
    for sentence in tqdm(text):
        if sentence:
            sentence = "".join(sentence)
            try:
                cache_hit = cache[sentence]
                big_labels_list.append(cache_hit)
                continue
            except:
                pass
            pairs_list = []
            for hypo in ucc_hypotheses:
                hypotheses_pairs = (sentence.lower(), hypo)
                pairs_list.append(hypotheses_pairs)
            scores = model.predict(pairs_list)
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

            big_labels_list.append(labels)
        else:
            big_labels_list.append(None)
    both_list = []
    entail_list = []
    contradict_list = []
    neutral_list = []
    decision = []
    decision_mapping = dict()
    text = list(text)
    for idx, label in enumerate(big_labels_list):
        if text[idx]:
            sentence = "".join(text[idx])
            if label == None:
                decision_mapping[sentence] = "Uncertain"
                decision.append("Uncertain")
            elif "entailment" in label and "contradiction" in label:
                occurrences = Counter(label)
                if occurrences["entailment"] > occurrences["contradiction"]:
                    entail_list.append(idx)
                    decision_mapping[sentence] = "Support"
                    decision.append("Support")
                elif occurrences["entailment"] < occurrences["contradiction"]:
                    contradict_list.append(idx)
                    decision_mapping[sentence] = "Against"
                    decision.append("Against")
                else:
                    both_list.append(idx)
                    decision_mapping[sentence] = "Uncertain"
                    decision.append("Uncertain")
            elif "entailment" in label and "contradiction" not in label:
                entail_list.append(idx)
                decision_mapping[sentence] = "Support"
                decision.append("Support")
            elif "entailment" not in label and "contradiction" in label:
                contradict_list.append(idx)
                decision_mapping[sentence] = "Against"
                decision.append("Against")
            else:
                neutral_list.append(idx)
                try:
                    decision_mapping[sentence] = "Uncertain"
                    decision.append("Uncertain")
                except:
                    print(sentence)
                    break
        else:
            decision.append(None)
            pass
    big_email_data["Verdict"] = decision
    big_email_data.to_json("/home/users/kinjals/work_dir/check/master_hineng_cleaned_with_sent.json", orient='records')
if __name__ == "__main__":
    main()
