import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# The exact 24 classes sorted alphabetically
classes = [
    "BitTorrent", "Cridex Malware", "FTP", "Facetime", "Geodo Malware",
    "Gmail", "Htbot Malware", "Miuref Malware", "MySQL", "Neris Malware",
    "Nsis-ay Malware", "Outlook", "SMB-1", "SMB-2", "Shifu Malware",
    "Skype", "Tinba Malware", "Virut Malware", "Weibo-1", "Weibo-2",
    "Weibo-3", "Weibo-4", "WorldOfWarcraft", "Zeus Malware"
]

print("Loading Ground Truth and Predictions...")

with open("Datasets/test_dataset.tsv", "r") as f:
    true_lines = f.readlines()[1:]

with open("Datasets/prediction.tsv", "r") as f:
    pred_lines = f.readlines()[1:]

y_true = []
y_pred = []

for true_line, pred_line in zip(true_lines, pred_lines):
    y_true.append(int(true_line.split('\t')[0]))
    y_pred.append(int(pred_line.strip()))

# Generate the raw mathematical matrix
cm = confusion_matrix(y_true, y_pred)

print("Drawing the Heatmap...")
# Set up the matplotlib figure (made it very large so all 24 classes fit perfectly)
plt.figure(figsize=(22, 18))
sns.set_theme(style="white")

# Draw the heatmap using Seaborn
# annot=True puts the actual numbers inside the boxes
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='gray', square=True, 
            cbar_kws={"shrink": .75})

plt.title('ET-BERT Traffic Classification: Confusion Matrix', fontsize=26, pad=20, fontweight='bold')
plt.ylabel('True Traffic Family (What it actually is)', fontsize=18, labelpad=15)
plt.xlabel('Predicted Traffic Family (What the AI guessed)', fontsize=18, labelpad=15)

# Rotate the labels so they don't overlap
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
graph_filename = 'Metrics/confusion_matrix_heatmap.png'
plt.savefig(graph_filename, dpi=300)
print(f"--> Success: Heatmap saved as '{graph_filename}'")
