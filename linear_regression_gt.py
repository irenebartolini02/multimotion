from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

EMOTIONS_DIMENSIONS = {
    'Amused': {'angle': 20, 'magnitude': 0.59},
    'Angry': {'angle': 117, 'magnitude': 0.88},
    'Bored': {'angle': 247, 'magnitude': 0.87},
    'Calm': {'angle': 317, 'magnitude': 0.98},
    'Content': {'angle': 325, 'magnitude': 0.99},
    'Disgust': {'angle': 144, 'magnitude': 0.82},
    'Excited': {'angle': 46.0, 'magnitude': 1},
    'Fearful': {'angle': 98, 'magnitude': 0.79},
    'Happy': {'angle': 10, 'magnitude': 0.91},
    'Negative': {'angle': 180, 'magnitude': 1},
    'Positive': {'angle': 0, 'magnitude': 1},
    'Sad': {'angle': 215, 'magnitude': 0.91}
}


def extract_vectorial_gt(input_file: str, output_file: str, participants_to_exclude: list[str], emotions_to_use: list[str]):
    output_file = Path(output_file)
    output_folder = output_file.parent
    if not output_folder.exists():
        output_folder.mkdir()

    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    # Filter patients
    participants_to_exclude = participants_to_exclude.copy()
    df_filtered = df[~df['respondent'].isin(participants_to_exclude)]

    # Remove meditation stimuli
    df_no_meditations = df_filtered[~df_filtered['stimulus'].str.contains("meditation")]

    # Get unique respondents and stimuli (using drop_duplicates to ensure uniqueness)
    respondents = df_no_meditations['respondent'].drop_duplicates()
    stimuli = df_no_meditations['stimulus'].drop_duplicates()

    # Initialize an empty dictionary to store individual similarity matrices for each respondent
    individual_vectorial_gts_x = []
    individual_vectorial_gts_y = []

    emotions_df = pd.DataFrame(EMOTIONS_DIMENSIONS).T

    # Imputation: replace nans with the average value across participants
    for stimulus in stimuli:
        ratings = df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use]
        ratings = ratings.fillna(ratings.mean())
        df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use] = ratings

    # Iterate over each respondent
    for i, respondent in enumerate(respondents):
        respondent_df = df_no_meditations[df_no_meditations['respondent'] == respondent]
        ratings = respondent_df[emotions_to_use]
        ratings.columns = ratings.columns.str.replace("*", "")
        ratings = ratings / 10

        x, y = polar_to_cartesian(ratings, emotions_df, stimuli)

        # Normalise to have a circle of radius 1
        x = x / np.sqrt(x ** 2 + y ** 2)
        y = y / np.sqrt(x ** 2 + y ** 2)

        x['Participant'] = respondent
        y['Participant'] = respondent
        individual_vectorial_gts_x.append(x)
        individual_vectorial_gts_y.append(y)

    # All participants
    all_x = pd.concat(individual_vectorial_gts_x)
    all_y = pd.concat(individual_vectorial_gts_y)

    ind_gt = (pd.merge(pd.melt(all_x, id_vars='Participant', var_name='Stimulus_Name', value_name='Valence'),
                       pd.melt(all_y, id_vars='Participant', var_name='Stimulus_Name', value_name='Arousal'),
                       on=['Participant', 'Stimulus_Name'])
              .sort_values(['Participant', 'Stimulus_Name']))

    ind_gt.to_csv(output_file, index=False)

    lopo_path = Path(output_file).parent / Path("leave-one-out-gt")
    if not lopo_path.exists():
        lopo_path.mkdir()

    # Leave-one-out
    for participant in all_y.Participant.unique():
        lopo_x = all_x[all_x['Participant'] != participant]
        lopo_y = all_y[all_y['Participant'] != participant]

        ind_gt = (pd.merge(pd.melt(lopo_x, id_vars='Participant', var_name='Stimulus_Name', value_name='Valence'),
                           pd.melt(lopo_y, id_vars='Participant', var_name='Stimulus_Name', value_name='Arousal'),
                           on=['Participant', 'Stimulus_Name'])
                  .sort_values(['Participant', 'Stimulus_Name']))

        ind_gt.to_csv(lopo_path / Path(output_file.stem + f"_no_{participant}").with_suffix(output_file.suffix), index=False)

    group_x = all_x.drop('Participant', axis=1).mean(axis=0)
    group_y = all_y.drop('Participant', axis=1).mean(axis=0)

    df = pd.concat([group_x, group_y], axis=1)
    df.columns = ['x', 'y']
    df['names'] = df.index

    # Rename stimuli
    # df['names'] = df['names'].replace(MAPPING_FIXES_14_01_25)

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
        plt.annotate(row['names'], (row['x'], row['y']), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=18)

    plt.xlabel('Valence', fontsize=22)
    plt.ylabel('Arousal', fontsize=22)
    plt.title('Group space with Vectorial Sum', fontsize=24)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.grid(True)
    # plt.savefig(os.path.join(plots_folder_path, 'group_space_plot.pdf'), format='pdf')
    # plt.savefig(os.path.join(plots_folder_path, 'group_space_plot.png'))
    plt.show()
    plt.close()  # Close the figure to free up memory


def polar_to_cartesian(magnitudes, emotions_df, stimuli):
    emotion_angles = emotions_df['angle']
    emotion_magnitudes = emotions_df['magnitude']

    xs = emotion_magnitudes.multiply(magnitudes.multiply(np.cos(np.radians(emotion_angles)).values))
    ys = emotion_magnitudes.multiply(magnitudes.multiply(np.sin(np.radians(emotion_angles)).values))

    xs_sum_abs_squared = xs.abs().sum(axis=1)**2
    ys_sum_abs_squared = ys.abs().sum(axis=1)**2

    weighted_xs = xs * (xs.abs() ** 2).div(xs_sum_abs_squared, axis=0)
    weighted_ys = ys * (ys.abs() ** 2).div(ys_sum_abs_squared, axis=0)

    # weighted_xs = np.sign(xs) * np.pow(2, xs.abs())
    # weighted_ys = np.sign(ys) * np.pow(2, ys.abs())

    x = pd.DataFrame(weighted_xs.sum(axis=1)).T
    x.columns = stimuli
    y = pd.DataFrame(weighted_ys.sum(axis=1)).T
    y.columns = stimuli

    """
    x = pd.DataFrame(np.dot(magnitudes, np.cos(np.radians(angles)).T).T, columns=stimuli)
    y = pd.DataFrame(np.dot(magnitudes, np.sin(np.radians(angles)).T).T, columns=stimuli)
    """
    return x, y


if __name__ == '__main__':
    extract_vectorial_gt("survey_data/SurveyDataJuly24/exp-2025-05-14-with-lopo/merged_data.xlsx",
                         "survey_data/SurveyDataJuly24/exp-2025-05-14-with-lopo/vectorial/individual_ground_truth.csv",
                         ["F3YZw", "VGCNk", "3hxsz", "EMX7o", "iHMUL", "SaqCY", "hit94", "pV9tu", "4AxqV", "agWdu", "CAJBM", "CEMKM", "RYXuu", "6CCj", "8C6iE", "cdVZZ", "dDLFD", "dU5uF", "er6ZP", "fA8yJ", "FS48s", "jJUMU", "KSPEO", "twq7m", "uu9cj", "VksBp", "hMHEK2", "KfBii", "2LBuk", "8DhRbY", "WaWrWi", "Yfg4l", "SbBNV", "A5TLF", "Oa4hap", "oJxBj", "VBss6", "Cjddt", "Lh5dT", "L2Lzw", "L2Lzw_2", "L9TG7", "Fg2ghyT", "FmN3", "FDsxx", "SDJkl1"],
                         ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
                          'Happy*', 'Negative*', 'Positive*', 'Sad*']
                         )