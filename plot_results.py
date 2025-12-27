import pandas as pd
import matplotlib.pyplot as plt

# Load results.csv
df = pd.read_csv('runs/detect/train4/results.csv')

# Plot metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Box loss
axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
axes[0, 0].set_title('Box Loss')
axes[0, 0].legend()

# Class loss
axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss')
axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss')
axes[0, 1].set_title('Class Loss')
axes[0, 1].legend()

# mAP50
axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
axes[1, 0].set_title('mAP50')
axes[1, 0].legend()

# mAP50-95
axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
axes[1, 1].set_title('mAP50-95')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('runs/detect/train4/training_results.png')
print("График сохранён в runs/detect/train4/training_results.png")