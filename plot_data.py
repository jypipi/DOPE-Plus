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

 # TODO
metrics_to_plot = [
    'train_mAP',
    'val_mAP',
]
plot_cookies = 1 # TODO
plot_block = 0 # TODO


plt.figure()

# Plot each metric
for metric in metrics_to_plot:
    if plot_cookies and metric == 'val_ADD':
        for i in range(len(run2[metric])):
            if run2[metric][i] > 80 and i > 5:
                print("Outlier found at index", i, "with value", run2[metric][i])
                run2[metric][i] = run2[metric][i-1]

    metric_label = metric.replace('_', ' ').title() # TODO

    if plot_cookies:
        df1 = run1.dropna(subset=['_step', metric])
        df2 = run2.dropna(subset=['_step', metric])
        df1['_step'] = df1['_step'] * 0.5
        df2['_step'] = df2['_step'] * 0.5

        plt.plot(df1['_step'], df1[metric], label=metric_label + f' ({ORIGINAL})')
        plt.plot(df2['_step'], df2[metric], label=metric_label + f' ({VIT})')

    elif plot_block:
        df3 = run3.dropna(subset=['_step', metric])
        df4 = run4.dropna(subset=['_step', metric])
        df3['_step'] = df3['_step'] * 0.5
        df4['_step'] = df4['_step'] * 0.5

        plt.plot(df3['_step'], df3[metric], label=metric_label + f' ({ORIGINAL})')
        plt.plot(df4['_step'], df4[metric], label=metric_label + f' ({VIT})')


item = "mAP" # TODO
if plot_cookies:
    OBJECT = 'Cookies'
    save_path = f"Figures/{item}_cookies_2.png"
elif plot_block:
    OBJECT = 'Block'
    save_path = f"Figures/{item}_block.png"


plt.xlabel('Epoch')
plt.ylabel(item)
title = f'Training and Validation {item} [{OBJECT}]'
plt.title(title)
plt.legend()
plt.grid(True)

plt.savefig(save_path) # TODO

plt.show()
