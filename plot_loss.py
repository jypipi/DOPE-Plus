import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSVs
run1 = pd.read_csv('original_cookies.csv')
run2 = pd.read_csv('vit_cookies_4.csv')
run3 = pd.read_csv('original_block.csv')
run4 = pd.read_csv('vit_block.csv')

# Name the runs for labeling
ORIGINAL = "original"
VIT = "vit"
OBJECT = None
# name3 = "original, block"
# name4 = "vit, block"

# Choose the metrics to compare
# metrics_to_plot = [
#     'train_accuracy', 'train_ADD', 'train_loss', 'train_mAP', 'train_loss_affinities', 'train_loss_belief',
#     'val_mAP', 'val_accuracy', 'val_ADD', 'val_loss'
# ]

metrics_to_plot = [
    'train_loss_affinities',
    'train_loss_belief',
    'val_loss',
]
plot_cookies = 1
plot_block = 0

train_loss_1 = []
train_loss_2 = []

plt.figure()

# Plot each metric
for metric in metrics_to_plot:
    metric_label = metric.replace('_', ' ').title()

    if plot_cookies:
        df1 = run1.dropna(subset=['_step', metric])
        df2 = run2.dropna(subset=['_step', metric])
        df1['_step'] = df1['_step'] * 0.5
        df2['_step'] = df2['_step'] * 0.5

        if metric == 'val_loss':
            plt.plot(df1['_step'], df1[metric], label=metric_label + f' ({ORIGINAL})')
            plt.plot(df2['_step'], df2[metric], label=metric_label + f' ({VIT})')
        else:
            train_loss_1.append(df1[metric])
            train_loss_2.append(df2[metric])

    elif plot_block:
        df3 = run3.dropna(subset=['_step', metric])
        df4 = run4.dropna(subset=['_step', metric])
        df3['_step'] = df3['_step'] * 0.5
        df4['_step'] = df4['_step'] * 0.5

        if metric == 'val_loss':
            plt.plot(df3['_step'], df3[metric], label=metric_label + f' ({ORIGINAL})')
            plt.plot(df4['_step'], df4[metric], label=metric_label + f' ({VIT})')
        else:
            train_loss_1.append(df3[metric])
            train_loss_2.append(df4[metric])

combined_train_loss_1 = train_loss_1[0] + train_loss_1[1]
combined_train_loss_2 = train_loss_2[0] + train_loss_2[1]

if plot_cookies:
    plt.plot(df1['_step'], combined_train_loss_1, label='Train Loss ' + f'({ORIGINAL})')
    plt.plot(df2['_step'], combined_train_loss_2, label='Train Loss ' + f'({VIT})')
elif plot_block:
    plt.plot(df3['_step'], combined_train_loss_1, label='Train Loss ' + f'({ORIGINAL})')
    plt.plot(df4['_step'], combined_train_loss_2, label='Train Loss ' + f'({VIT})')

item = "Loss"
if plot_cookies:
    OBJECT = 'Cookies'
    save_path = f"Figures/{item}_cookies_4.png"
elif plot_block:
    OBJECT = 'Block'
    save_path = f"Figures/{item}_block.png"


plt.xlabel('Epoch')
plt.ylabel(item)
title = f'Training and Validation {item} [{OBJECT}]'
plt.title(title)
plt.legend()
plt.grid(True)

# plt.savefig(save_path)

plt.show()
