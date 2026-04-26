# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:42
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : distance_matrix.py
# @Software: PyCharm
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from typing_extensions import deprecated


def process_and_combine_similarities(input_file: str, output_file: str, participants_to_exclude: list[str],
                                     emotions_to_use: list[str], fa_folder_path: str):
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
    print(stimuli)
    # Initialize an empty dictionary to store individual similarity matrices for each respondent
    individual_similarity_matrices = {}

    # check_unexpectedly_similar_videos(df_no_meditations, respondents)

    # Imputation: replace nans with the average value across participants
    for stimulus in stimuli:
        ratings = df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use]
        ratings = ratings.fillna(ratings.mean())
        df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use] = ratings

    extract_matrix_for_fa(df_no_meditations, emotions_to_use, fa_folder_path)

    # Iterate over each respondent
    for i, respondent in enumerate(respondents):
        respondent_df = df_no_meditations[df_no_meditations['respondent'] == respondent]
        ratings = respondent_df[emotions_to_use]

        # Calculate the similarity matrix using Euclidean distance
        similarity_matrix = squareform(pdist(ratings, metric='euclidean'))

        # Store the similarity matrix in the dictionary with respondent's identifier as the key
        individual_similarity_matrices[respondent] = similarity_matrix

    combined_similarity_df = pd.DataFrame(np.vstack(list(individual_similarity_matrices.values())), columns=stimuli)

    # Create an ExcelWriter to save all the individual matrices into a single Excel file
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Save the combined DataFrame to a new sheet in the Excel file
        combined_similarity_df.to_excel(writer, sheet_name='Combined_Sheet', index=False)

        # Save individual similarity matrices to separate sheets in the Excel file
        for respondent, similarity_matrix in individual_similarity_matrices.items():
            # Create a DataFrame for the similarity matrix with proper indices and column names
            respondent_similarity_df = pd.DataFrame(similarity_matrix, index=stimuli, columns=stimuli)

            # Save the DataFrame to the Excel file with a sheet name based on the respondent
            sheet_name = f'Respondent_{respondent}'
            respondent_similarity_df.to_excel(writer, sheet_name=sheet_name)

            # Apply conditional formatting to highlight response patterns
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            yellow_format = workbook.add_format({'bg_color': '#FFFF99', 'font_color': '#996600'})
            (max_row, max_col) = respondent_similarity_df.shape
            stimuli_block_dim = max_row // 4

            for i in range(1, max_row, stimuli_block_dim):
                # Format blocks that are on the main diagonal
                worksheet.conditional_format(i, i, i + stimuli_block_dim - 1, i + stimuli_block_dim - 1,
                                             {'type': 'cell',
                                              'criteria': '>',
                                              'value': 11,
                                              'format': yellow_format})
                
                # Format blocks that are not on the main diagonal
                if i < max_row - stimuli_block_dim:
                    worksheet.conditional_format(i, i + stimuli_block_dim, i + stimuli_block_dim - 1, max_col,
                                                 {'type': 'cell',
                                                  'criteria': '<',
                                                  'value': 11,
                                                  'format': yellow_format})

                    worksheet.conditional_format(i + stimuli_block_dim, i, max_row, i + stimuli_block_dim - 1,
                                                 {'type': 'cell',
                                                  'criteria': '<',
                                                  'value': 11,
                                                  'format': yellow_format})

            worksheet.conditional_format(1, 1, max_row, max_col,
                                         {'type': '3_color_scale',
                                          'min_color': '#FF6666',
                                          'mid_color': '#FFFFFF',
                                          'max_color': '#336699'})

    # Notify that the saving process is completed
    print(f"All individual similarity matrices have been combined and saved to '{output_file}' on a single sheet.")


def lopo_process_and_combine_similarities(input_file: str, output_file: str, participants_to_exclude: list[str],
                                          emotions_to_use: list[str], fa_folder_path: str):

    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    # Filter patients
    participants_to_exclude = participants_to_exclude.copy()
    respondents = df[~df['respondent'].isin(participants_to_exclude)].loc[:, 'respondent'].unique()

    for respondent in respondents:
        lopo_participants_to_exclude = participants_to_exclude + [respondent]
        lopo_output_file = Path(output_file).parent / Path(f"{Path(output_file).stem}_no_{respondent}{Path(output_file).suffix}")
        process_and_combine_similarities(input_file, str(lopo_output_file), lopo_participants_to_exclude, emotions_to_use, fa_folder_path)


def l2po_process_and_combine_similarities(input_file: str, output_file: str, participants_to_exclude: list[str],
                                          emotions_to_use: list[str], fa_folder_path: str):

    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    # Filter patients
    participants_to_exclude = participants_to_exclude.copy()
    respondents = df[~df['respondent'].isin(participants_to_exclude)].loc[:, 'respondent'].unique()

    for i in range(len(respondents)):
        for j in range(i+1, len(respondents)):
            respondent1, respondent2 = respondents[i], respondents[j]
            lopo_participants_to_exclude = participants_to_exclude + [respondent1, respondent2]
            lopo_output_file = Path(output_file).parent / Path(f"{Path(output_file).stem}_no_{respondent1}_no_{respondent2}{Path(output_file).suffix}")
            process_and_combine_similarities(input_file, str(lopo_output_file), lopo_participants_to_exclude, emotions_to_use, fa_folder_path)


@deprecated("Used only once for analysis purposes, no need to use it systematically")
def check_unexpectedly_similar_videos(df_no_meditations: pd.DataFrame, respondents: list[str]):
    """
    Function used to control unexpectedly similar videos that should be far from each other in the graphical representation
    """
    emotions_to_use_reordered = ['Bored*', 'Calm*', 'Content*',
                                 'Amused*', 'Positive*', 'Happy*', 'Excited*',
                                 'Angry*', 'Disgust*', 'Fearful*', 'Negative*', 'Sad*']

    for stimulus_pair in [['HP_3', 'LP_8'], ['HP_1', 'LP_2'], ['HP_6', 'LP_1'], ['HP_7', 'LP_7'], ['HP_2', 'LP_5']]:
        df_pair = df_no_meditations[df_no_meditations['stimulus'].isin(stimulus_pair)]
        individual_ratings = {}
        differences = []

        # Iterate over each respondent
        for i, respondent in enumerate(respondents):
            respondent_df = df_pair[df_pair['respondent'] == respondent]

            ratings = respondent_df[emotions_to_use_reordered]
            individual_ratings[respondent] = respondent_df[emotions_to_use_reordered]

            differences.append(ratings.diff().abs().iloc[1])

        mean_diff_df = pd.DataFrame(np.mean(list(differences), axis=0).reshape(1, -1),
                                    columns=emotions_to_use_reordered).T
        mean_diff_df.sort_values(by=[0], ascending=True, inplace=True)

        mean_df = pd.DataFrame(np.mean(list(individual_ratings.values()), axis=0), columns=emotions_to_use_reordered).T
        mean_df.columns = stimulus_pair

        std_df = pd.DataFrame(np.std(list(individual_ratings.values()), axis=0), columns=emotions_to_use_reordered).T
        std_df.columns = stimulus_pair

        with pd.ExcelWriter(
                f"survey_data/SurveyDataJuly24/exp-2024-10-11/emotions_distances/{'-'.join(stimulus_pair)}.xlsx",
                engine='xlsxwriter') as writer:
            mean_diff_df.to_excel(writer, sheet_name='diff')
            mean_df.to_excel(writer, sheet_name='mean')
            std_df.to_excel(writer, sheet_name='std')


def extract_matrix_for_fa(df_no_meditations: pd.DataFrame, emotions_to_use: list[str], fa_folder_path: str):
    fa_base_folder_path = Path(fa_folder_path)
    if not fa_base_folder_path.exists():
        fa_base_folder_path.mkdir()

    matrices = []
    clean_df_fa = df_no_meditations[['respondent', 'stimulus'] + emotions_to_use]
    melted_df = clean_df_fa.melt(id_vars=['respondent', 'stimulus'],
                                 var_name='Emotion',
                                 value_name="Value")
    for emotion, matrix in melted_df.groupby('respondent'):
        df_for_fa = matrix.pivot(index='stimulus', columns='Emotion', values='Value')
        # print(f"{emotion}: {round(float(alpha), 3)}")
        matrices.append(df_for_fa)
    fa_matrix = pd.concat(matrices, axis=0)
    fa_matrix.to_excel(fa_base_folder_path / "fa_matrix.xlsx")


def pca_group_space(input_file: str, patients_to_exclude: list[str], emotions_to_use: list[str]):
    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    # Filter patients
    df_filtered = df[~df['respondent'].isin(patients_to_exclude)]

    # Remove meditation stimuli
    df_no_meditations = df_filtered[~df_filtered['stimulus'].str.contains("meditation")]
    stimuli = df_no_meditations['stimulus'].drop_duplicates()

    # Imputation: replace nans with the average value across participants
    for stimulus in stimuli:
        ratings = df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use]
        ratings = ratings.fillna(ratings.mean())
        df_no_meditations.loc[df_no_meditations['stimulus'] == stimulus, emotions_to_use] = ratings

    # Get unique respondents and stimuli (using drop_duplicates to ensure uniqueness)
    respondents = df_no_meditations['respondent'].drop_duplicates()

    matrix = group_pca(df_no_meditations, respondents, emotions_to_use)
    rotated_matrix = np.dot(matrix.T, np.array([[-1, 0], [0, -1]]))
    matrix_df = pd.DataFrame(rotated_matrix)

    return matrix_df


def group_pca(df, respondents, emotions_to_use):
    covariances = []
    for i, respondent in enumerate(respondents):
        respondent_df = df[df['respondent'] == respondent]
        ratings = respondent_df[emotions_to_use]

        C = covariance_matrix(ratings.to_numpy())
        covariances.append(C)

    avg_covariances = np.mean(covariances, axis=0)

    U, s, Vh = np.linalg.svd(avg_covariances)
    P = U[:, 0:2]
    DP = np.dot(P.T, avg_covariances)
    return DP


def covariance_matrix(D):
    mu = D.mean(axis=1)
    mu = mu.reshape((mu.size, 1))

    DC = D - mu
    return np.dot(DC, DC.T) / D.shape[1]
