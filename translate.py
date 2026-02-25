import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The exact 24 classes sorted alphabetically
classes = [
    "BitTorrent", "Cridex Malware", "FTP", "Facetime", "Geodo Malware",
    "Gmail", "Htbot Malware", "Miuref Malware", "MySQL", "Neris Malware",
    "Nsis-ay Malware", "Outlook", "SMB-1", "SMB-2", "Shifu Malware",
    "Skype", "Tinba Malware", "Virut Malware", "Weibo-1", "Weibo-2",
    "Weibo-3", "Weibo-4", "WorldOfWarcraft", "Zeus Malware"
]

print("Reading the Answer Key (test_dataset.tsv)...")
with open("Datasets/test_dataset.tsv", "r") as f:
    # Skip the first line (header)
    true_lines = f.readlines()[1:]

print("Reading the AI's Guesses (prediction.tsv)...")
with open("Datasets/prediction.tsv", "r") as f:
    # Skip the first line (header)
    pred_lines = f.readlines()[1:]

results = []
correct_count = 0

for index, (true_line, pred_line) in enumerate(zip(true_lines, pred_lines)):
    # The true dataset has the label and the hex text separated by a tab
    true_label_id = int(true_line.split('\t')[0])

    # The prediction file just has the number
    pred_label_id = int(pred_line.strip())

    # Check if the AI got it right
    is_correct = (true_label_id == pred_label_id)
    if is_correct:
        correct_count += 1

    results.append({
        "Packet_ID": index + 1,
        "True_Family": classes[true_label_id],
        "Predicted_Family": classes[pred_label_id],
        "Is_Correct": "✅ YES" if is_correct else "❌ NO"
    })

# Save the graded test to a new CSV
df = pd.DataFrame(results)
output_file = "Metrics/graded_predictions.csv"
df.to_csv(output_file, index=False)

# Calculate final grade
accuracy = (correct_count / len(pred_lines)) * 100

print(f"\n--- FINAL GRADE ---")
print(f"Total Packets Checked: {len(pred_lines)}")
print(f"Correct Guesses: {correct_count}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Saved full report to: {output_file}")

# ==========================================
# NEW: GENERATE THE CLASS-WISE BAR CHART
# ==========================================
print("\nGenerating class-wise accuracy graph...")

# Group by the true class to get total packets and correctly predicted packets
class_stats = df.groupby('True_Family').apply(
    lambda x: pd.Series({
        'Total': len(x),
        'Correct': (x['Is_Correct'] == '✅ YES').sum()
    })
).reset_index()

# Enforce the original alphabetical order on the x-axis
class_stats['True_Family'] = pd.Categorical(class_stats['True_Family'], categories=classes, ordered=True)
class_stats = class_stats.sort_values('True_Family')

# Initialize the matplotlib figure
plt.figure(figsize=(18, 8))
sns.set_theme(style="whitegrid")

# Plot 1: The Total Packets (Background bar - Light Gray)
sns.barplot(x="True_Family", y="Total", data=class_stats, color="#d3d3d3", label="Total Ground Truth Packets")

# Plot 2: The Correctly Predicted Packets (Foreground bar - Dark Blue/Green)
sns.barplot(x="True_Family", y="Correct", data=class_stats, color="#2ca02c", label="Correctly Predicted")

# Add informative labels and title
plt.title('ET-BERT Prediction Accuracy by Traffic Class', fontsize=20, pad=20, fontweight='bold')
plt.xlabel('Traffic Family', fontsize=14, labelpad=15)
plt.ylabel('Number of Packets', fontsize=14, labelpad=15)

# Rotate the x labels 45 degrees so all 24 names are legible without overlapping
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# Add a legend at the top right
plt.legend(ncol=1, loc="upper right", frameon=True, fontsize=12)

# Adjust layout and save
plt.tight_layout()
graph_filename = 'Metrics/class_accuracy_bar_chart.png'
plt.savefig(graph_filename, dpi=300)
plt.show()

print(f"--> Success: Graph saved as '{graph_filename}'")
