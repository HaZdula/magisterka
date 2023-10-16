import os
import pandas as pd
import matplotlib.pyplot as plt

dfs = []

results_dir = 'results'
for file in os.listdir(results_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(results_dir, file), sep=';')
        df['flavor'] = file.split('.')[0]
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)


# print(df)


def create_plot(df, ood_column_name, ood_label, best_only=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    flavor_colors = {'LP': 'red', 'LPFT': 'purple', 'FT': 'blue'}
    df['color'] = df['flavor'].map(flavor_colors)

    if best_only:
        df = df.groupby('flavor').agg({ood_column_name: 'max'}).reset_index().merge(df, on=['flavor', ood_column_name], how='left')
        print(df)
        x_ft = df['cifar10_acc'][df['flavor']=='FT'].iloc[0]
        y_ft = df[ood_column_name][df['flavor']=='FT'].iloc[0]
        x_lp = df['cifar10_acc'][df['flavor']=='LP'].iloc[0]
        y_lp = df[ood_column_name][df['flavor']=='LP'].iloc[0]
        x_lpft = df['cifar10_acc'][df['flavor']=='LPFT'].iloc[0]
        y_lpft = df[ood_column_name][df['flavor']=='LPFT'].iloc[0]

        lr_lp = df['lr'][df['flavor']=='LP'].iloc[0]
        lr_lpft = df['lr'][df['flavor']=='LPFT'].iloc[0]
        lr_ft = df['lr'][df['flavor']=='FT'].iloc[0]

        ax.annotate("", xy=(x_lpft, y_lpft),
                    xytext=(x_lp, y_lp),
                    arrowprops=dict(arrowstyle="->", color='red', alpha=0.5))
        ax.annotate("", xy=(x_lpft, y_lpft),
                    xytext=(x_ft, y_ft),
                    arrowprops=dict(arrowstyle="->", color='blue', alpha=0.5))
        ax.annotate(f"lr = {lr_lp}", xy=(x_lpft, y_lpft),
                    xytext=(x_lp, y_lp),)
        ax.annotate(f"lr = {lr_ft}", xy=(x_lpft, y_lpft),
                    xytext=(x_ft, y_ft),)
        ax.annotate(f"lr = {lr_lpft}", xy=(x_lpft, y_lpft),
                    xytext=(x_lpft, y_lpft), )

    else:
        # add arrow from LP to LPFT and from FT to LPFT
        for lr in df['lr'].unique():
            df_lr = df[df['lr'] == lr]
            df_lp = df_lr[df_lr['flavor'] == 'LP']
            df_lpft = df_lr[df_lr['flavor'] == 'LPFT']
            df_ft = df_lr[df_lr['flavor'] == 'FT']

            ax.annotate("", xy=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        xytext=(df_lp['cifar10_acc'].iloc[0], df_lp[ood_column_name].iloc[0]),
                        arrowprops=dict(arrowstyle="->", color='red', alpha=0.8),
                        bbox=dict(edgecolor=(0, 0, 0, 0), facecolor=(1, 1, 1, 0)))
            ax.annotate("", xy=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        xytext=(df_ft['cifar10_acc'].iloc[0], df_ft[ood_column_name].iloc[0]),
                        arrowprops=dict(arrowstyle="->", color='blue', alpha=0.8),
                        bbox=dict(edgecolor=(0, 0, 0, 0), facecolor=(1, 1, 1, 0)))

            ax.annotate(lr, xy=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        xytext=(df_lp['cifar10_acc'].iloc[0], df_lp[ood_column_name].iloc[0]),
                        bbox=dict(edgecolor=(0, 0, 0, 0), facecolor=(1, 1, 1, 0)))
            ax.annotate(lr, xy=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        xytext=(df_ft['cifar10_acc'].iloc[0], df_ft[ood_column_name].iloc[0]),
                        bbox=dict(edgecolor=(0, 0, 0, 0), facecolor=(1, 1, 1, 0)))
            ax.annotate(lr, xy=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        xytext=(df_lpft['cifar10_acc'].iloc[0], df_lpft[ood_column_name].iloc[0]),
                        bbox=dict(edgecolor=(0, 0, 0, 0), facecolor=(1, 1, 1, 0)))

    plt.xlabel('CIFAR10 Accuracy')
    plt.ylabel(ood_label)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for flavor in df['flavor'].unique():
        df_flavor = df[df['flavor'] == flavor]
        ax.scatter(data=df_flavor, x='cifar10_acc', y=ood_column_name, c='color', alpha=0.5, label=flavor)
    plt.legend(flavor_colors)
    plt.show()
    return


#create_plot(df, 'cifar101_acc', 'CIFAR10.1 Accuracy', best_only=True)
#create_plot(df, 'stl_acc', 'STL Accuracy', best_only=True)

create_plot(df, 'cifar101_acc', 'CIFAR10.1 Accuracy', best_only=False)
create_plot(df, 'stl_acc', 'STL Accuracy', best_only=False)