import json
import os
import glob

#merges all the files from the Potts et al file into 1 JSON file
def merge_json_files(input_folder_path, output_file_path):
    json_file_pattern = os.path.join(input_folder_path, '**', '*.json')
    file_list = glob.glob(json_file_pattern, recursive=True)

    all_data = []
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_data.append(data)
                elif isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {file_path} contains unsupported type: {type(data)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:
            print(f"Error with file {file_path}: {e}")

    # Save the merged list to a new JSON file
    if all_data:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(all_data, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully merged {len(file_list)} files into {output_file_path}")
    else:
        print("No valid JSON data found to merge.")


input_folder = "datasets/potts/"

output_file = "keith_mcdaniel_merged.json"

# 3. Run the function
merge_json_files(input_folder, output_file)