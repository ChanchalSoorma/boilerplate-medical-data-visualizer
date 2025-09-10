import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column
# Calculate BMI = weight (kg) / (height (m))^2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalize data: cholesterol and gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # 4. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 5. Group and reformat the data to show counts of each feature split by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')


   # 6. Draw the categorical plot with seaborn catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                      data=df_cat, kind='bar').fig

    # 7
    fig.savefig('catplot.png')
    return fig


# 8
def draw_heat_map():
    # 9. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 10. Calculate the correlation matrix
    corr = df_heat.corr()

    # 11. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 12. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 13. Draw the heatmap with seaborn
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, square=True, linewidths=.5, ax=ax, center=0)

    # 14
    fig.savefig('heatmap.png')
    return fig
