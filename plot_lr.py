import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSVs
run1 = pd.read_csv('original_cookies.csv')
run2 = pd.read_csv('vit_cookies.csv')
run3 = pd.read_csv('original_block.csv')
run4 = pd.read_csv('vit_block.csv')

# Name the runs for labeling
ORIGINAL = "original"
VIT = "vit"
OBJECT = None

metrics_to_plot = [
    'current_lr',
]

plt.figure()

# Plot each metric
for metric in metrics_to_plot:

    df1 = run1.dropna(subset=['_step', metric])
    df2 = run2.dropna(subset=['_step', metric])
    df1['_step'] = df1['_step'] * 0.5
    df2['_step'] = df2['_step'] * 0.5

    plt.plot(df1['_step'], df1[metric], label=f'Cookies ({ORIGINAL})')
    plt.plot(df2['_step'], df2[metric], label=f'Cookies ({VIT})')

    df3 = run3.dropna(subset=['_step', metric])
    df4 = run4.dropna(subset=['_step', metric])
    df3['_step'] = df3['_step'] * 0.5
    df4['_step'] = df4['_step'] * 0.5

    plt.plot(df3['_step'], df3[metric], label=f'Block ({ORIGINAL})')
    plt.plot(df4['_step'], df4[metric], label=f'Block ({VIT})')

plt.xlabel('Epoch')
plt.ylabel("Learning Rate")
title = f'Cosine Annealing Learning Rate'
plt.title(title)
plt.legend()
plt.grid(True)
save_path = f"Figures/Learning_rates.png"
# plt.savefig(save_path)

plt.show()
