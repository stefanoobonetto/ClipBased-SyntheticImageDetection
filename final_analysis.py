import re
import pandas as pd

def parse_results(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    videos = re.split(r"/Users/[^\n]+\.mp4", content)
    video_paths = re.findall(r"/Users/[^\n]+\.mp4", content)
    
    results = []
    fusion_methods = ["mean_logit", "max_logit", "median_logit", "lse_logit", "mean_prob", "soft_or_prob"]

    for video, path in zip(videos[1:], video_paths):
        video_name = path.split("/")[-1]
        true_label = "Real" if "Real" in video_name else "Synthetic"
        
        for method in fusion_methods:
            match = re.search(rf"fusion\[{method}\]\s+\|\s+(\w+)\s+\|\s+([\d.]+)", video)
            if match:
                predicted_label, confidence = match.groups()
                results.append({
                    "video": video_name,
                    "true_label": true_label,
                    "method": method,
                    "predicted_label": predicted_label,
                    "confidence": float(confidence),
                    "correct": predicted_label == true_label
                })
    
    return pd.DataFrame(results)

file_path = "results.txt"  
df_results = parse_results(file_path)

accuracy_per_method = df_results.groupby("method")["correct"].mean().sort_values(ascending=False)

print("Fusion Method Accuracies:")
print(accuracy_per_method)
