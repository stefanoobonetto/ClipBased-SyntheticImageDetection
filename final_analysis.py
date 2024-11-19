import re
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    csv_file_path = os.path.join(RESULTS_PATH, f'final_results_{string_videos}.csv')
    
    data = pd.read_csv(csv_file_path)

    filenames = data['Filename']
    models = data.columns[1:]  

    short_filenames = filenames.str.replace(r"^.*[\\/]", "", regex=True)

    # Remove the "Average" row from the data before plotting
    data_without_avg = data[data['Filename'] != "Average"]

    # Generate the plot
    plt.figure(figsize=(14, 8))

    # Assign colors to each model
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # Plot each model's data with its corresponding color
    for model in models:
        if data[model].dtype == float or data[model].dtype == int:
            plt.plot(data_without_avg['Filename'], data_without_avg[model], marker='o', label=model, color=color_map[model])

    # Add average lines using the same colors
    for model in models:
        if data[model].dtype == float or data[model].dtype == int:
            avg_value = data[model].mean()
            plt.axhline(y=avg_value, linestyle='--', label=f"{model} (avg)", color=color_map[model])

    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Filename')
    plt.ylabel('Confidence')
    plt.title('Confidence Scores by Model')

    # Position the legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.tight_layout()

    plot_file_path = os.path.join(RESULTS_PATH, f'confidence_comparison_plot_{string_videos}.png')
    plt.savefig(plot_file_path)
    print(f"Plot has been saved to {plot_file_path}")

    plt.show()

    
if __name__ == "__main__":
    main()