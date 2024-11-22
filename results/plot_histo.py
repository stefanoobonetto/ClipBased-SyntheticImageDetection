import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/stefanobonetto/Documents/GitHub/ClipBased-SyntheticImageDetection/results/comparison_models.csv'
data = pd.read_csv(file_path)

models = data['Model']
columns = data.columns[1:]
x_indices = np.arange(len(columns))  
width = 0.2  

colors = ['#9A75FF', '#00C4B4', '#FF8A33']  

fig, ax = plt.subplots(figsize=(12, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    ax.bar(
        x_indices + i * width,  
        data.iloc[i, 1:],      
        width=width,           
        color=color,           
        label=model,
        edgecolor='black',            
    )

ax.set_xticks(x_indices + width)  
ax.set_xticklabels(columns, rotation=45, ha='right')  

ax.set_xlabel('Detectors and Fusion methods')
ax.set_ylabel('Confidence')
ax.set_title('Confidence Levels of Detectors and Fusion methods across different models')
ax.legend(title='Models')

ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)  
ax.set_yticks(np.arange(0, 1.1, 0.1))  

plt.tight_layout()
plt.show()
