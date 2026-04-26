import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


MAPPING_FIXES_JULY_24 = {
    'LN_3': 'HN_9',
    'HN_8': 'NN_1',
    'HN_7': 'NN_2',
    'LP_1': 'HP_9',
    'LP_2': 'HP_10',
    'LP_5': 'HP_11',
    'LP_8': 'HP_13',
    'HP_5': 'NP_1',
    'HP_8': 'LP_9'
}

MAPPING_FIXES_14_01_25 = {
    'LN_8': 'LN_9',
    'HN_8': 'HN_9',
    'LP_8': 'LP_9',
    'HP_8': 'HP_9',
    'HP_7_L': 'LP_HP_7',
    'HP_3_L': 'LP_HP_3',
    'HP_1_L': 'LP_HP_1',
    'HN_3_L': 'LN_HN_3',
    'HN_2_L': 'LN_HN_2',
    'LN_7_P': 'LP_LN_7',
}


def plot_individual_ground_truths(individual_ground_truth_path: str, plots_folder_base_path: str):
    plots_folder_path = os.path.join(plots_folder_base_path, 'individual_plots')

    # Create a folder to save images
    if not os.path.exists(plots_folder_path):
        os.makedirs(plots_folder_path)
    # Read data from CSV
    data = pd.read_csv(individual_ground_truth_path)

    # data['Valence'] = (data['Valence'] - data['Valence'].mean()) / data['Valence'].std()
    # data['Arousal'] = (data['Arousal'] - data['Arousal'].mean()) / data['Arousal'].std()

    # Group data by respondent
    grouped_data = data.groupby('Participant')  # Assuming the first column is 'Participant'

    # Create scatter plots for each respondent
    for participant, group in grouped_data:

        lns_df = group[group['Stimulus_Name'].str.startswith('LN')]
        hns_df = group[group['Stimulus_Name'].str.startswith('HN')]
        lps_df = group[group['Stimulus_Name'].str.startswith('LP')]
        hps_df = group[group['Stimulus_Name'].str.startswith('HP')]

        # Plot the normalized data points and label them
        plt.figure(figsize=(10, 8))
        plt.scatter(lns_df['Valence'], lns_df['Arousal'], color='b', marker='o', label='Low Arousal Negative Valence (LN)')
        plt.scatter(hns_df['Valence'], hns_df['Arousal'], color='r', marker='o', label='High Arousal Negative Valence (HN)')
        plt.scatter(lps_df['Valence'], lps_df['Arousal'], color='g', marker='o', label='Low Arousal Positive Valence (LP)')
        plt.scatter(hps_df['Valence'], hps_df['Arousal'], color='y', marker='o', label='High Arousal Positive Valence (HP)')
        plt.legend(fontsize=14)

        for i, row in group.iterrows():
            plt.annotate(row['Stimulus_Name'], (row['Valence'], row['Arousal']), textcoords="offset points", xytext=(0, 10), ha='center',
                         fontsize=14)

        plt.xlabel('Valence', fontsize=18)
        plt.ylabel('Arousal', fontsize=18)
        plt.title(f"Ground truth for Respondent {participant}", fontsize=18)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlim(-1.1, 1.1)  # Set x-axis range to -1.1 to 1.1
        plt.ylim(-1.1, 1.1)  # Set y-axis range to -1.1 to 1.1
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        # Save the plot as an image in the folder
        plt.savefig(os.path.join(plots_folder_path, f'plot_{participant}.png'))
        plt.show()
        plt.close()  # Close the figure to free up memory


def plot_grouped_ground_truth_from_csv(group_space_data_path: str, plots_folder_base_path: str,
                                       stimuli_names: list[str], transformations: dict[str, bool]):
    # Read the CSV file
    df = pd.read_csv(group_space_data_path, header=None)
    plot_grouped_ground_truth(df, plots_folder_base_path, stimuli_names, transformations)


def plot_grouped_ground_truth(df: pd.DataFrame, plots_folder_base_path: str, stimuli_names: list[str],
                              transformations: dict[str, bool]):
    plots_folder_path = os.path.join(plots_folder_base_path, 'group_space_plot')
    if not os.path.exists(plots_folder_path):
        os.makedirs(plots_folder_path)

    if transformations['mirror_y']:
        # Mirror along Y axis (invert positive and negative)
        df.iloc[:, 0] = df.iloc[:, 0] * -1

    if transformations['mirror_x']:
        # Mirror along X axis (invert high and low)
        df.iloc[:, 1] = df.iloc[:, 1] * -1

    if transformations['rotate_90_deg']:
        # Rotate 90deg clockwise
        rot_matrix = np.array([[0, -1], [1 , 0]])
        df = pd.DataFrame(np.dot(df, rot_matrix))

    if transformations['rotate_180_deg']:
        # Rotate 180deg clockwise
        rot_matrix = np.array([[-1, 0], [0, -1]])
        df = pd.DataFrame(np.dot(df, rot_matrix))

    df.columns = ['x', 'y']

    # Normalisation: Centered MinMax normalisation using max(|min(x) - 0.1 * |min(x)||, |max(x) + 0.1 * |max(x)||)
    normalised_individual_maps = pd.DataFrame({'x': df.x, 'y': df.y})
    max_val = df.iloc[:, 0].max()
    min_val = df.iloc[:, 0].min()
    max_val = max(
        abs(min_val - 0.1 * abs(min_val)),
        abs(max_val + 0.1 * abs(max_val))
    )
    min_val = -max_val

    normalised_individual_maps.iloc[:, 0] = ((df.iloc[:, 0] - min_val) / (max_val - min_val)) * 2 - 1

    max_ar = df.iloc[:, 1].max()
    min_ar = df.iloc[:, 1].min()
    max_ar = max(
        abs(min_ar - 0.1 * abs(min_ar)),
        abs(max_ar + 0.1 * abs(max_ar))
    )
    min_ar = -max_ar
    normalised_individual_maps.iloc[:, 1] = ((df.iloc[:, 1] - min_ar) / (max_ar - min_ar)) * 2 - 1
    df = normalised_individual_maps

    # Rename stimuli
    df['names'] = stimuli_names
    df['names'] = df['names'].replace(MAPPING_FIXES_14_01_25)

    lns_df = df[df['names'].str.startswith('LN')]
    hns_df = df[df['names'].str.startswith('HN')]
    lps_df = df[df['names'].str.startswith('LP')]
    hps_df = df[df['names'].str.startswith('HP')]

    # Plot the normalized data points and label them
    plt.figure(figsize=(15, 12))
    plt.scatter(lns_df['x'], lns_df['y'], color='b', marker='o', label='Low Arousal Negative Valence (LN)')
    plt.scatter(hns_df['x'], hns_df['y'], color='r', marker='o', label='High Arousal Negative Valence (HN)')
    plt.scatter(lps_df['x'], lps_df['y'], color='g', marker='o', label='Low Arousal Positive Valence (LP)')
    plt.scatter(hps_df['x'], hps_df['y'], color='y', marker='o', label='High Arousal Positive Valence (HP)')
    plt.legend(fontsize=16)

    for i, row in df.iterrows():
        plt.annotate(row['names'], (row['x'], row['y']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=18)

    plt.xlabel('Valence', fontsize=22)
    plt.ylabel('Arousal', fontsize=22)
    plt.title('Group space with INDSCAL', fontsize=24)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.grid(True)
    # plt.savefig(os.path.join(plots_folder_path, 'pca_group_space_plot.pdf'), format='pdf')
    plt.savefig(os.path.join(plots_folder_path, 'group_space_plot.png'))
    plt.show()
    plt.close()  # Close the figure to free up memory


def plot_fa_group_space(fa_folder_path: str, merged_data_path: str, participants_to_exclude: list[str]):
    base_fa_folder_path = Path(fa_folder_path)
    valence_arousal_fa_matrix = pd.read_csv(base_fa_folder_path / "fa_valence_arousal_matrix.csv")

    # Import the data from a CSV file
    df = pd.read_excel(merged_data_path, sheet_name="Combined_Sheet")
    df_filtered = df[~df['respondent'].isin(participants_to_exclude)]
    # Remove meditation stimuli
    df_no_meditations = df_filtered[~df_filtered['stimulus'].str.contains("meditation")]
    stimuli = df_no_meditations['stimulus'].drop_duplicates().to_list()
    participants = df_no_meditations['respondent'].drop_duplicates().to_list()

    df_part = pd.DataFrame(participants)
    df_part = df_part.loc[df_part.index.repeat(len(stimuli))]
    df_part = df_part.reset_index().drop(columns=['index'], axis=1)
    valence_arousal_fa_matrix.loc[:, 'Participant'] = df_part.iloc[:, 0]
    valence_arousal_fa_matrix = valence_arousal_fa_matrix.rename(columns={"stimulus": "Stimulus_Name"})

    avg_space = valence_arousal_fa_matrix[['Stimulus_Name', 'Valence', 'Arousal']].groupby('Stimulus_Name').mean()
    avg_space.columns = ['x', 'y']
    avg_space['names'] = avg_space.index

    # Rename stimuli
    avg_space['names'] = avg_space['names'].replace(MAPPING_FIXES_14_01_25)

    lns_df = avg_space[avg_space['names'].str.startswith('LN')]
    hns_df = avg_space[avg_space['names'].str.startswith('HN')]
    lps_df = avg_space[avg_space['names'].str.startswith('LP')]
    hps_df = avg_space[avg_space['names'].str.startswith('HP')]

    # Plot the normalized data points and label them
    plt.figure(figsize=(15, 12))
    plt.scatter(lns_df['x'], lns_df['y'], color='b', marker='o', label='Low Arousal Negative Valence (LN)')
    plt.scatter(hns_df['x'], hns_df['y'], color='r', marker='o', label='High Arousal Negative Valence (HN)')
    plt.scatter(lps_df['x'], lps_df['y'], color='g', marker='o', label='Low Arousal Positive Valence (LP)')
    plt.scatter(hps_df['x'], hps_df['y'], color='y', marker='o', label='High Arousal Positive Valence (HP)')
    plt.legend(fontsize=16)

    for i, row in avg_space.iterrows():
        plt.annotate(row['names'], (row['x'], row['y']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=18)

    plt.xlabel('Valence', fontsize=22)
    plt.ylabel('Arousal', fontsize=22)
    plt.title('Group space with Factor Analysis', fontsize=24)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.grid(True)
    # plt.savefig(base_fa_folder_path / 'fa_group_space_plot.pdf', format='pdf')
    plt.savefig(base_fa_folder_path / 'fa_group_space_plot.png')
    plt.show()
    plt.close()  # Close the figure to free up memory

    ind_gt = valence_arousal_fa_matrix[['Participant', 'Stimulus_Name', 'Valence', 'Arousal']]
    ind_gt.loc[:, 'Stimulus_Name'] = ind_gt.loc[:, 'Stimulus_Name'].replace(MAPPING_FIXES_14_01_25)
    ind_gt.to_csv(base_fa_folder_path / "fa_individual_ground_truth.csv", index=False)
