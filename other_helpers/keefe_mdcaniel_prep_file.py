import json
import os
import glob
import spacy
import sys

try:
    nlp = spacy.load("en_core_web_sm")
    print ("spaCy loaded")
except Exception as e:
    print ("failed to load spaCy model")
    sys.exit(1)

def lemmatize_word(word):
    doc = nlp(word)
    return doc[0].lemma_.lower() if doc else word.lower()

def create_keefe_dataset(input_folder_path, output_file_path):
    pattern = os.path.join(input_folder_path, '**', '*.json')
    file_list = glob.glob(pattern, recursive=True)
    
    all_items = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_probe = data.get("probe", "")
            lemma_probe = lemmatize_word(raw_probe)
            item = {
                "id": data.get("id"),
                "target_word": raw_probe,
                "target_lemma": lemma_probe,
            }
            # Extract ONLY first_sentence from control, predictive, explicit
            for cond in ["control", "predictive", "explicit"]:
                if cond in data:
                    first = data[cond].get("first_sentence", "").strip()
                    item[f"{cond}_text"] = first
            if all(f"{cond}_text" in item for cond in ["control", "predictive", "explicit"]):
                all_items.append(item)
            else:
                print(f"Warning: Item {data.get('id')} missing condition")
    
    with open(output_file_path, 'w', encoding='utf-8') as out:
        json.dump(all_items, out, indent=2, ensure_ascii=False)
    print(f"Created {len(all_items)} items for Keefe experiment")

create_keefe_dataset("datasets/potts/", "keefe_ready.json")