import re
import os
import pandas as pd


parent_path = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(parent_path, "results")

models = [
    "clipdet_latent10k", "clipdet_latent10k_plus", "Corvi2023",
    "fusion\\[max_logit\\]", "fusion\\[mean_logit\\]", "fusion\\[median_logit\\]",
    "fusion\\[lse_logit\\]", "fusion\\[mean_prob\\]", "fusion\\[soft_or_prob\\]"
]

def further_analysis(file_path, string_videos):
    with open(file_path, 'r') as file:
        content = file.read()

    sections = content.split("\n\n")

    results_data = []

    for section in sections:
        filename_match = re.search(r'/([A-Za-z_]+)_(Real|Fake)\.mp4', section)
        if not filename_match:
            continue
        filename = filename_match.group(0)
        expected_label = "Synthetic" if filename_match.group(2) == "Fake" else "Real"
        
        predictions = {"Filename": filename}
        
        for model in models:
            model_match = re.search(
                rf"{model}.*?\|\s+(Real|Synthetic)\s+\|\s+([\d\.]+)", section
            )
            if model_match:
                output = model_match.group(1)
                confidence = float(model_match.group(2))
                if output == expected_label:
                    predictions[model.replace("\\", "")] = confidence
                else:
                    predictions[model.replace("\\", "")] = 1 - confidence
        
        results_data.append(predictions)

    df_results = pd.DataFrame(results_data)

    avg_row = df_results.drop(columns=["Filename"]).mean().to_dict()
    avg_row["Filename"] = "Average"

    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)

    csv_file_path = os.path.join(RESULTS_PATH, f'final_results_{string_videos}.csv')
    df_results.to_csv(csv_file_path, index=False)

    print(f"Results have been saved to {csv_file_path}")

def main():
    
    path = input("For which file would you like to perform further analysis? \n 1. Luma \n 2. Latte \n 3. CogVideoX-5B \n")

    while path not in ['1', '2', '3']:
        path = input("Invalid input. Please enter 1, 2, or 3: ")
    
    if path == '1':
        path = os.path.join(RESULTS_PATH, "results_luma.txt")
        string_videos = "luma"
    elif path == '2':
        path = os.path.join(RESULTS_PATH, "results_latte.txt")
        string_videos = "latte"
    else:
        path = os.path.join(RESULTS_PATH, "results_cogvideo.txt")
        string_videos = "cogvideo"
    
    further_analysis(path, string_videos)
    
if __name__ == "__main__":
    main()