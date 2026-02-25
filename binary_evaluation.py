import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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

y_true_binary = []
y_pred_binary = []

# Grouping Logic: If the word "Malware" is in the class name, it goes to the Malware bucket.
for true_line, pred_line in zip(true_lines, pred_lines):
    true_id = int(true_line.split('\t')[0])
    pred_id = int(pred_line.strip())

    true_category = "Malware" if "Malware" in classes[true_id] else "Benign"
    pred_category = "Malware" if "Malware" in classes[pred_id] else "Benign"

    y_true_binary.append(true_category)
    y_pred_binary.append(pred_category)

# Calculate overall binary accuracy
correct_count = sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == p)
accuracy = (correct_count / len(y_true_binary)) * 100

print(f"\n--- FIREWALL SECURITY POSTURE (Benign vs Malware) ---")
print(f"Total Packets Evaluated: {len(y_true_binary)}")
print(f"Successfully Handled: {correct_count}")
print(f"New Real-World Accuracy: {accuracy:.2f}%\n")

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_true_binary, y_pred_binary, labels=["Benign", "Malware"]))

# Generate the 2x2 Confusion Matrix
print("\nDrawing the Binary Heatmap...")
labels = ["Benign", "Malware"]
cm = confusion_matrix(y_true_binary, y_pred_binary, labels=labels)

plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

# Using a green colormap to signify firewall success
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels,
            linewidths=1, linecolor='gray', square=True, annot_kws={"size": 16})

plt.title('ET-BERT Firewall: Benign vs. Malware', fontsize=18, pad=15, fontweight='bold')
plt.ylabel('True Traffic Type (Reality)', fontsize=14, labelpad=10)
plt.xlabel('Predicted Traffic Type (AI Decision)', fontsize=14, labelpad=10)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13, rotation=0)

plt.tight_layout()
graph_filename = 'Metrics/binary_confusion_matrix.png'
plt.savefig(graph_filename, dpi=300)
print(f"--> Success: Heatmap saved as '{graph_filename}'")
