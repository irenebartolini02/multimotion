# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:11
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : combine_data.py
# @Software: PyCharm


import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def concatenate_dataframes(new_dataframe_paths: list[str], new_dataframe_path_2: str, updated_dataframe_path: str,
                           emotions_to_use: list[str], positive_emotions: list[str], negative_emotions: list[str],
                           participants_to_exclude: list[str]):
    dataframes_to_concat = []

    out_file_folder, _ = os.path.split(updated_dataframe_path)
    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)

    if new_dataframe_paths is not None:
        for new_dataframe_path in new_dataframe_paths:
            if new_dataframe_path is not None and os.path.exists(new_dataframe_path):
                new_df = pd.read_excel(new_dataframe_path, sheet_name="Combined_Sheet")
                dataframes_to_concat.append(new_df)
    new_df_2 = pd.read_excel(new_dataframe_path_2, sheet_name="Combined_Sheet")
    dataframes_to_concat.append(new_df_2)

    merged_experiments = pd.concat(dataframes_to_concat)

    # Create an ExcelWriter to save all the individual matrices into a single Excel file
    # Apply conditional formatting to each mark and count the mismatches
    with pd.ExcelWriter(updated_dataframe_path, engine='xlsxwriter') as writer:
        combined_normalised_matrices = list()

        # Get unique respondents and stimuli (using drop_duplicates to ensure uniqueness)
        respondents = merged_experiments['respondent'].drop_duplicates()
        for respondent in respondents:
            respondent_df = merged_experiments.loc[merged_experiments['respondent'] == respondent]

            values_to_normalize = respondent_df[emotions_to_use].values
            flattened_matrix = values_to_normalize.flatten().reshape(-1, 1)

            # Apply MinMaxScaler to the flattened matrix
            scaler = MinMaxScaler(feature_range=(0, 9))
            scaled_flattened_matrix = scaler.fit_transform(flattened_matrix)

            # Reshape the scaled data back to the original matrix shape
            scaled_matrix = scaled_flattened_matrix.reshape(values_to_normalize.shape)

            # Assign the scaled matrix back to the DataFrame, with rounding
            respondent_df.loc[:, emotions_to_use] = np.round(scaled_matrix, 1)

            combined_normalised_matrices.append(respondent_df)

            # Save the DataFrame to the Excel file with a sheet name based on the respondent
            respondent_df_clean = respondent_df.drop(columns=['respondent'])
            sheet_name = f'Respondent_{respondent}'
            respondent_df_clean.to_excel(writer, sheet_name=sheet_name, index=False)

            # Detect mismatches in emotions and survey responses
            n_mismatches = detect_mismatching_emotions_in_surveys(respondent_df_clean, positive_emotions, negative_emotions)

            # Apply conditional formatting to highlight response patterns
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            worksheet.autofit()

            red_format = workbook.add_format({'bg_color': '#FF4444'})
            green_format = workbook.add_format({'bg_color': '#CCDD99'})
            blue_format = workbook.add_format({'bg_color': '#6699CC'})

            if n_mismatches < 4:
                worksheet.write(0, 0, f'mismatches: {n_mismatches}', green_format)
            elif n_mismatches < 8:
                worksheet.write(0, 0, f'mismatches: {n_mismatches}', blue_format)
            else:
                worksheet.write(0, 0, f'mismatches: {n_mismatches}', red_format)

            (max_row, max_col) = respondent_df_clean.shape
            worksheet.conditional_format(1, 1, max_row, max_col,
                                         {'type': 'cell',
                                          'criteria': 'between',
                                          'minimum': 1,
                                          'maximum': 4,
                                          'format': green_format})

            worksheet.conditional_format(1, 1, max_row, max_col,
                                         {'type': 'cell',
                                          'criteria': '>=',
                                          'value': 6,
                                          'format': red_format})

            worksheet.conditional_format(1, 1, max_row, max_col,
                                         {'type': 'cell',
                                          'criteria': '=',
                                          'value': 5,
                                          'format': blue_format})

            if respondent in participants_to_exclude:
                worksheet.write(max_row + 1, 0, f'Excluded Participant', red_format)
            else:
                worksheet.write(max_row + 1, 0, f'Accepted Participant', green_format)

        normalised_merged_experiments = pd.concat(combined_normalised_matrices, ignore_index=True)

        # Save the combined DataFrame to a new sheet in the Excel file
        normalised_merged_experiments.to_excel(writer, sheet_name='Combined_Sheet', index=False)


def detect_mismatching_emotions_in_surveys(data: pd.DataFrame, positive_emotions: list[str], negative_emotions: list[str]):
    positive_answers = data[['stimulus'] + positive_emotions]
    negative_answers = data[['stimulus'] + negative_emotions]

    positive_answers_and_negative_stimuli = positive_answers[positive_answers['stimulus'].str.contains("N_")]
    negative_answers_and_positive_stimuli = negative_answers[negative_answers['stimulus'].str.contains("P_")]

    mismatches_pa_ns = int((positive_answers_and_negative_stimuli[positive_emotions] > 5).sum().sum())
    mismatches_na_ps = int((negative_answers_and_positive_stimuli[negative_emotions] > 5).sum().sum())

    return mismatches_pa_ns + mismatches_na_ps
