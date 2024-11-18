import re
import pandas as pd

models = [
    "clipdet_latent10k", "clipdet_latent10k_plus", "Corvi2023",
    "fusion\\[max_logit\\]", "fusion\\[mean_logit\\]", "fusion\\[median_logit\\]",
    "fusion\\[lse_logit\\]", "fusion\\[mean_prob\\]", "fusion\\[soft_or_prob\\]"
]

with open('results.txt', 'r') as file:
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

csv_file_path = 'final_results.csv'
df_results.to_csv(csv_file_path, index=False)

print(f"Results have been saved to {csv_file_path}")
