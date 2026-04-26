# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:38
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : final_exp_data_converter.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler

MAPPING_JULY_24 = {
    'survey_LN_1_1': 'LN_1',
    'survey_LN_1_2': 'LN_2',
    'survey_LN_1_3': 'LN_3',
    'survey_LN_1_4': 'LN_4',
    'survey_LN_1_5': 'LN_5',
    'survey_LN_1_6': 'LN_6',
    'survey_LN_1_7': 'LN_7',
    'survey_LN_1_8': 'LN_8',
    'survey_HN_1_7': 'HN_7',
    'survey_HP_7.1': 'HP_7',
    'survey_HP_1-1': 'HP_1',
    'survey_HP_6.1': 'HP_6',
    'survey_LP_1_2': 'LP_2',
    'survey_LP_1_3': 'LP_3',
    'survey_LP_1_4': 'LP_4',
    'survey_LP_6.1': 'LP_6',
    'survey_LP_1_7': 'LP_7',
    'survey_LP_1_7-1': 'LP_8',
    'meditation_survey_B': 'meditation_start',
    'meditation_survey_B-1': 'meditation_end'
}

MAPPING_OLD_VERSION_1 = {
    "Survey_A": "A_HN",
    "Survey_A1": "A1_LP",
    "Survey_AA2": "A2_LP",
    "Survey_AA3": "A3_LP",
    "Survey_B": "B_HN",
    "Survey_C": "C_LN",
    "Survey_A-1": "A4_LP",
    "Survey_F": "F_HN",
    "Survey_G": "G_HP",
    "Survey_H": "H_HP",
    "Survey_J": "J_Ne",
    "Survey_K": "K_Ne",
    "Survey_M": "M_LN",
    "Survey_N": "N_LN",
    "Survey_O": "O_LN",
    "Survey_P": "P_HP",
    "Survey_U": "U_Ne",
    "Survey_V": "V_Ne",
    "Survey_W": "W_HN",
    "Survey_q": "Q_HP"
}

MAPPING_OLD_VERSION_FINAL = {
    'Survey_A': 'A_HN',
    'Survey_A1': 'A1_LP',
    'Survey_A2': 'A2_LP',
    'Survey_A3': 'A3_LP',
    'Survey_A4': 'A4_LP',
    'Survey_B': 'B_HN',
    'Survey_C': 'C_LN',
    'Survey_F': 'F_HN',
    'Survey_G': 'G_HP',
    'Survey_H': 'H_HP',
    'Survey_J': 'J_Ne',
    'Survey_K': 'K_Ne',
    'Survey_M': 'M_LN',
    'Survey_N': 'N_LN',
    'Survey_O': 'O_LN',
    'Survey_P': 'P_HP',
    'Survey_U': 'U_Ne',
    'Survey_V': 'V_Ne',
    'Survey_W': 'W_HN',
    'Survey_q': 'Q_HP'
}

EMOTIONAL_SPLIT_MAPPING = {
    "HP_1": {"target_emotion": "Amused*", "new_names": ("HP_1_H", "HP_1_L")},
    "HP_3": {"target_emotion": "Excited*", "new_names": ("HP_3_H", "HP_3_L")},
    "HP_7": {"target_emotion": "Excited*", "new_names": ("HP_7_H", "HP_7_L")},
    "HN_2": {"target_emotion": "Fearful*", "new_names": ("HN_2_H", "HN_2_L")},
    "HN_3": {"target_emotion": "Fearful*", "new_names": ("HN_3_H", "HN_3_L")},
    # "HN_5": {"target_emotion": "Negative*", "new_names": ("HN_5_N", "HN_5_P")},
    "LN_7": {"target_emotion": "Negative*", "new_names": ("LN_7_N", "LN_7_P")},
}

# Negative
"""
HIGH_AROUSAL_CENTROID = {
    'Amused*': 6.6942,
    'Angry*': 6.62434,
    'Bored*': -0.90217,
    'Calm*': 1.04836,
    'Content*': 3.63248,
    'Disgust*': 1.21311,
    'Excited*': 5.7598,
    'Fearful*': 3.83367,
    'Happy*': 7.06334,
    'Negative*': 2.17718,
    'Positive*': 3.95502,
    'Sad*': -0.84642
}

LOW_AROUSAL_CENTROID = {
    'Amused*': 0.06986,
    'Angry*': -1.14852,
    'Bored*': 3.41109,
    'Calm*': 7.05522,
    'Content*': 3.63248,
    'Disgust*': 1.21311,
    'Excited*': -0.125640,
    'Fearful*': -1.17083,
    'Happy*': 3.60787,
    'Negative*': 2.17718,
    'Positive*': 3.95502,
    'Sad*': -0.846412
}

POSITIVE_VALENCE_CENTROID = {
    'Amused*': 6.6942,
    'Angry*': -1.14852,
    'Bored*': 1.25446,
    'Calm*': 7.05522,
    'Content*': 6.74709,
    'Disgust*': -1.18779,
    'Excited*': 5.7598,
    'Fearful*': -1.17083,
    'Happy*': 7.06334,
    'Negative*': -0.70275,
    'Positive*': 7.34928,
    'Sad*': -0.84642
}

NEGATIVE_VALENCE_CENTROID = {
    'Amused*': 0.06986,
    'Angry*': 3.2637,
    'Bored*': 1.25446,
    'Calm*': 1.04836,
    'Content*': 0.51787,
    'Disgust*': 3.61401,
    'Excited*': -0.12564,
    'Fearful*': 3.83367,
    'Happy*': 0.15240,
    'Negative*': 5.05711,
    'Positive*': 0.56076,
    'Sad*': 5.28046
}
"""

# Zero-capped
"""
HIGH_AROUSAL_CENTROID = {
    'Amused*': 6.6942,
    'Angry*': 6.62434,
    'Bored*': 0,
    'Calm*': 1.04836,
    'Content*': 3.63248,
    'Disgust*': 1.21311,
    'Excited*': 5.7598,
    'Fearful*': 3.83367,
    'Happy*': 7.06334,
    'Negative*': 2.17718,
    'Positive*': 3.95502,
    'Sad*': 0
}

LOW_AROUSAL_CENTROID = {
    'Amused*': 0.06986,
    'Angry*': 0,
    'Bored*': 3.41109,
    'Calm*': 7.05522,
    'Content*': 3.63248,
    'Disgust*': 1.21311,
    'Excited*': 0,
    'Fearful*': 0,
    'Happy*': 3.60787,
    'Negative*': 2.17718,
    'Positive*': 3.95502,
    'Sad*': 0
}

POSITIVE_VALENCE_CENTROID = {
    'Amused*': 6.6942,
    'Angry*': 0,
    'Bored*': 1.25446,
    'Calm*': 7.05522,
    'Content*': 6.74709,
    'Disgust*': 0,
    'Excited*': 5.7598,
    'Fearful*': 0,
    'Happy*': 7.06334,
    'Negative*': 0,
    'Positive*': 7.34928,
    'Sad*': 0
}

NEGATIVE_VALENCE_CENTROID = {
    'Amused*': 0.06986,
    'Angry*': 3.2637,
    'Bored*': 1.25446,
    'Calm*': 1.04836,
    'Content*': 0.51787,
    'Disgust*': 3.61401,
    'Excited*': 0,
    'Fearful*': 3.83367,
    'Happy*': 0.15240,
    'Negative*': 5.05711,
    'Positive*': 0.56076,
    'Sad*': 5.28046
}
"""

# 1/2 std
"""
HIGH_AROUSAL_CENTROID = {'Amused*': 5.038115, 'Angry*': 4.968255, 'Bored*': 0.176145, 'Calm*': 2.550075, 'Content*': 3.63248, 'Disgust*': 1.21311, 'Excited*': 4.28844, 'Fearful*': 2.582545, 'Happy*': 5.335605, 'Negative*': 2.17718, 'Positive*': 3.95502, 'Sad*': 0.68530}

LOW_AROUSAL_CENTROID = {'Amused*': 1.725945, 'Angry*': 1.656085, 'Bored*': 2.332775, 'Calm*': 2.550075, 'Content*': 3.63248, 'Disgust*': 1.21311, 'Excited*': 1.34572, 'Fearful*': 0.080295, 'Happy*': 1.8801350, 'Negative*': 2.17718, 'Positive*': 3.95502, 'Sad*': 3.74874}

POSITIVE_VALENCE_CENTROID = {'Amused*': 5.038115, 'Angry*': -0.045465, 'Bored*': 1.25446, 'Calm*': 5.553505, 'Content*': 5.1897850, 'Disgust*': 0.01266, 'Excited*': 4.28844, 'Fearful*': 0.080295, 'Happy*': 5.335605, 'Negative*': 0.737215, 'Positive*': 5.652150, 'Sad*': 0.68530}

NEGATIVE_VALENCE_CENTROID = {'Amused*': 1.725945, 'Angry*': 2.160645, 'Bored*': 1.25446, 'Calm*': 2.550075, 'Content*': 2.075175, 'Disgust*': 2.41356, 'Excited*': 1.34572, 'Fearful*': 2.582545, 'Happy*': 1.8801350, 'Negative*': 3.617145, 'Positive*': 2.25789, 'Sad*': 3.74874}
"""

# HN_4 as centroid
HIGH_AROUSAL_CENTROID = {
    'Amused*': 3.102272727272727,
    'Angry*': 1.725,
    'Bored*': 0.6295454545454545,
    'Calm*': 2.4727272727272727,
    'Content*': 2.4727272727272727,
    'Disgust*': 1.7704545454545455,
    'Excited*': 1.4909090909090907,
    'Fearful*': 2.3545454545454545,
    'Happy*': 1.7272727272727273,
    'Negative*': 3.440909090909091,
    'Positive*': 2.0818181818181816,
    'Sad*': 2.765909090909091
}

# LN_5 as centroid
LOW_AROUSAL_CENTROID = {
    'Amused*': 1.7954545454545454,
    'Angry*': 0.6613636363636364,
    'Bored*': 2.1772727272727277,
    'Calm*': 3.45,
    'Content*': 2.5454545454545454,
    'Disgust*': 0.9590909090909091,
    'Excited*': 1.1272727272727272,
    'Fearful*': 0.4113636363636364,
    'Happy*': 1.5045454545454546,
    'Negative*': 2.584090909090909,
    'Positive*': 1.9977272727272728,
    'Sad*': 2.8227272727272723
}

# HP_6 as centroid
POSITIVE_VALENCE_CENTROID = {
    'Amused*': 5.9840909090909085,
    'Angry*': 0.1431818181818182,
    'Bored*': 1.2477272727272728,
    'Calm*': 5.286363636363637,
    'Content*': 5.381818181818182,
    'Disgust*': 0.19090909090909092,
    'Excited*': 5.045454545454546,
    'Fearful*': 0.2545454545454545,
    'Happy*': 6.2749999999999995,
    'Negative*': 0.2659090909090909,
    'Positive*': 6.434090909090909,
    'Sad*': 0.16818181818181818
}

# HN_6 as centroid
NEGATIVE_VALENCE_CENTROID = {
    'Amused*': 1.3136363636363637,
    'Angry*': 3.8863636363636362,
    'Bored*': 0.7272727272727273,
    'Calm*': 2.0704545454545453,
    'Content*': 1.5090909090909093,
    'Disgust*': 4.861363636363636,
    'Excited*': 0.6681818181818181,
    'Fearful*': 3.0568181818181817,
    'Happy*': 0.3977272727272727,
    'Negative*': 6.070454545454546,
    'Positive*': 0.8545454545454546,
    'Sad*': 5.20909090909091
}


def convert_experiment(survey_path: str, df_out_path: str, version: str, mapping: dict, emotions_to_use: list[str],
                       export_csv=False):
    def prepare_survey_data():
        survey_data_raw = []
        f = open(survey_path)
        counter = 0
        index_questions, index_responses, index_responses_headers = -1, -1, -1
        questions = None

        for line in f:
            line = line.strip()
            if len(line) > 0:
                line = line.replace(' \t ', ';').replace('\t', ';')
                if line.find("Matrix contains") != -1:
                    index_questions = counter
                elif line.find("Response Matrix") != -1:
                    index_responses = counter
                elif line.find("STUDY") != -1:
                    index_responses_headers = counter
            line = line.split(';')
            survey_data_raw.append(line)
            counter += 1

        f.close()

        if index_questions != -1:
            questions = pd.DataFrame(survey_data_raw[index_questions + 3:index_responses - 1])
            no_columns_needed = len(survey_data_raw[index_questions + 3]) - len(survey_data_raw[index_questions + 2])
            questions.columns = survey_data_raw[index_questions + 2] + ["Label-info"] * no_columns_needed

        responses = pd.DataFrame(survey_data_raw[index_responses_headers + 1:])
        responses.columns = survey_data_raw[index_responses_headers]

        return questions, responses

    _, responses = prepare_survey_data()

    slides = sorted(
        set(i[i.find("_") + 1: i.rfind("_")] for i in responses.loc[:, ['LABELID' in i for i in responses.columns]]))

    # Remove elements from old_data
    if version in ['version_1', 'version_5', 'version_final']:
        elements_to_remove = ["Survey_L", "Survey_I", "Survey_Y"]
        slides = [slide for slide in slides if slide not in elements_to_remove]

    filtered_columns = [col for col in responses.columns if any(slide in col for slide in slides)]
    filtered_responses = responses.loc[:, filtered_columns]
    emotions = sorted(set(
        i[i.find('"') + 1: i.rfind('"')] for i in filtered_responses
    ))

    # FYI: A-1 survey is actually for A4 video

    df = pd.DataFrame()
    if version in ['version_july_24']:
        meditation_slides = [s for s in slides if "meditation" in s]
        meditation_emotions = [e for e in emotions if "relaxation" in e]
        slides = [s for s in slides if "meditation" not in s]
        emotions = [e for e in emotions if e not in meditation_emotions]

        # Meditations have to be treated separately
        for respondent in responses["RESPONDENT"]:
            df_stimuli = pd.DataFrame({"respondent": respondent, "stimulus": meditation_slides})
            responses_respondent = responses.loc[responses["RESPONDENT"] == respondent]
            for emotion in meditation_emotions:
                values = []
                for slide in meditation_slides:
                    value = responses_respondent.loc[:,
                            responses.columns.str.contains(fr"LABELVALUE_{slide}_[\w\W]*{emotion}", regex=True)
                            ].values[0][0]

                    if value == "" or value is None:
                        values.append(0)
                    else:
                        values.append(int(value) - 1)
                df_emotion = pd.DataFrame({emotion: values})
                df_stimuli = pd.concat([df_stimuli, df_emotion], axis=1)
            df = pd.concat([df, df_stimuli], ignore_index=True)

    for respondent in responses["RESPONDENT"]:
        df_stimuli = pd.DataFrame({"respondent": respondent, "stimulus": slides})
        responses_respondent = responses.loc[responses["RESPONDENT"] == respondent]
        for emotion in emotions:
            values = []
            for slide in slides:
                value = responses_respondent.loc[:,
                        responses.columns.str.contains(fr"LABELVALUE_{slide}_[\w\W]*{emotion}", regex=True)
                        ].values[0][0]

                if value == "" or value is None:
                    values.append(0)
                else:
                    values.append(int(value) - 1)
            df_emotion = pd.DataFrame({emotion: values})
            df_stimuli = pd.concat([df_stimuli, df_emotion], axis=1)
        df = pd.concat([df, df_stimuli], ignore_index=True)

    df.sort_values(by=['respondent', 'stimulus'], inplace=True)
    # Assuming your dataframe is named 'df' and the column with the old stimulus names is named 'stimulus'
    df['stimulus'] = df['stimulus'].replace(mapping).str.replace("survey_", "").str.replace("-", "_")

    # Split videos that have two distinct emotional parts
    if version in ['version_july_24']:
        positive_centroid_df = pd.DataFrame([POSITIVE_VALENCE_CENTROID])
        negative_centroid_df = pd.DataFrame([NEGATIVE_VALENCE_CENTROID])
        high_centroid_df = pd.DataFrame([HIGH_AROUSAL_CENTROID])
        low_centroid_df = pd.DataFrame([LOW_AROUSAL_CENTROID])

        for video, mapping_info in EMOTIONAL_SPLIT_MAPPING.items():
            old_rows = df.loc[df['stimulus'] == video, :].copy()
            emotion_idxs = [old_rows.columns.get_loc(col) for col in emotions_to_use]

            new_rows = old_rows.copy()
            new_empty_rows = old_rows.copy()
            new_empty_rows.loc[:, emotions_to_use] = np.nan

            for i in range(old_rows.shape[0]):
                if video == 'LN_7':
                    distances_to_positive = pdist(
                        np.vstack([positive_centroid_df.values, old_rows.iloc[i, emotion_idxs].values]).astype(float),
                        metric='euclidean')
                    distances_to_negative = pdist(
                        np.vstack([negative_centroid_df.values, old_rows.iloc[i, emotion_idxs].values]).astype(float),
                        metric='euclidean')

                    if distances_to_positive > distances_to_negative:
                        new_rows.iloc[i, new_rows.columns.get_loc('stimulus')] = 'LN_7_N'
                        new_empty_rows.iloc[i, new_empty_rows.columns.get_loc('stimulus')] = 'LN_7_P'
                    else:
                        new_rows.iloc[i, new_rows.columns.get_loc('stimulus')] = 'LN_7_P'
                        new_empty_rows.iloc[i, new_empty_rows.columns.get_loc('stimulus')] = 'LN_7_N'

                else:
                    distance_to_high = pdist(
                        np.vstack([high_centroid_df.values, old_rows.iloc[i, emotion_idxs].values]).astype(float),
                        metric='euclidean')
                    distance_to_low = pdist(
                        np.vstack([low_centroid_df.values, old_rows.iloc[i, emotion_idxs].values]).astype(float),
                        metric='euclidean')

                    if distance_to_high > distance_to_low:
                        new_rows.iloc[i, new_rows.columns.get_loc('stimulus')] = new_rows.iloc[
                                                                                     i, new_rows.columns.get_loc(
                                                                                         'stimulus')] + '_L'
                        new_empty_rows.iloc[i, new_empty_rows.columns.get_loc('stimulus')] = new_empty_rows.iloc[
                                                                                                 i, new_rows.columns.get_loc(
                                                                                                     'stimulus')] + '_H'
                    else:
                        new_rows.iloc[i, new_rows.columns.get_loc('stimulus')] = new_rows.iloc[
                                                                                     i, new_rows.columns.get_loc(
                                                                                         'stimulus')] + '_H'
                        new_empty_rows.iloc[i, new_empty_rows.columns.get_loc('stimulus')] = new_empty_rows.iloc[
                                                                                                 i, new_rows.columns.get_loc(
                                                                                                     'stimulus')] + '_L'

            df = pd.concat([df, new_rows], ignore_index=True)
            df = pd.concat([df, new_empty_rows], ignore_index=True)

            """
            old_rows.loc[:, 'above_threshold'] = old_rows.loc[:, target_emotion] >= 5
            for idx, old_row in old_rows.iterrows():
                new_row_above_threshold = pd.DataFrame(old_row, dtype='object').T
                new_row_above_threshold.drop(columns=['above_threshold'], inplace=True)
                new_row_above_threshold.iloc[:, df.columns.get_loc('stimulus')] = new_names[0]

                new_row_below_threshold = pd.DataFrame(old_row, dtype='object').T
                new_row_below_threshold.drop(columns=['above_threshold'], inplace=True)
                new_row_below_threshold.iloc[:, df.columns.get_loc('stimulus')] = new_names[1]

                if np.all(old_row['above_threshold']):
                    new_row_below_threshold.loc[:, emotions_to_use] = np.nan
                else:
                    new_row_above_threshold.loc[:, emotions_to_use] = np.nan

                df = pd.concat([df, new_row_above_threshold], ignore_index=True)
                df = pd.concat([df, new_row_below_threshold], ignore_index=True)
            """

            df = df.loc[df['stimulus'] != video, :]

    if export_csv:
        df.to_csv(df_out_path, index=False)

    return df


def convert_multiple_experiments(survey_paths: list[str], df_out_path: str, emotions_to_use: list[str],
                                 version: str, export_single_csvs=False):
    out_file_without_extension, out_file_extension = os.path.splitext(df_out_path)
    out_file_folder, out_file_name_without_extension = os.path.split(out_file_without_extension)

    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)

    single_files_out_folder = os.path.join(out_file_folder, 'single_converted_data')

    if version in ['version_5', 'version_final']:
        mapping = MAPPING_OLD_VERSION_FINAL
    elif version in ['version_1']:
        mapping = MAPPING_OLD_VERSION_1
    else:
        mapping = MAPPING_JULY_24

    emotions = emotions_to_use.copy()
    emotions.append("relaxation*")

    if export_single_csvs:
        if not os.path.exists(single_files_out_folder):
            os.makedirs(single_files_out_folder)

    experiments = []
    for i, survey_path in enumerate(survey_paths):
        if not os.path.exists(survey_path):
            raise FileNotFoundError(f'{survey_path} does not exist')

        single_out_file_path = os.path.join(single_files_out_folder,
                                            f"{out_file_name_without_extension}_exp_{i}{out_file_extension}")
        df = convert_experiment(survey_path, single_out_file_path, version, mapping, emotions_to_use,
                                export_single_csvs)
        if version in ['version_july_24']:
            df.sort_values(by=['respondent', 'stimulus'], inplace=True)
        experiments.append(df)

    merged_experiments = pd.concat(experiments, ignore_index=True)

    # Create an ExcelWriter to save all the individual matrices into a single Excel file
    with pd.ExcelWriter(df_out_path) as writer:
        combined_normalised_matrices = list()

        # Get unique respondents and stimuli (using drop_duplicates to ensure uniqueness)
        respondents = merged_experiments['respondent'].drop_duplicates()
        for respondent in respondents:
            respondent_df = merged_experiments.loc[merged_experiments['respondent'] == respondent]

            values_to_normalize = respondent_df[emotions].values

            flattened_matrix = values_to_normalize.flatten().reshape(-1, 1)

            # Apply MinMaxScaler to the flattened matrix
            scaler = MinMaxScaler(feature_range=(0, 9))
            scaled_flattened_matrix = scaler.fit_transform(flattened_matrix)

            # Reshape the scaled data back to the original matrix shape
            scaled_matrix = scaled_flattened_matrix.reshape(values_to_normalize.shape)

            # Assign the scaled matrix back to the DataFrame, with rounding
            respondent_df.loc[:, emotions] = np.round(scaled_matrix, 1)

            combined_normalised_matrices.append(respondent_df)

            # Save the DataFrame to the Excel file with a sheet name based on the respondent
            respondent_df_clean = respondent_df.drop(columns=['respondent'])
            sheet_name = f'Respondent_{respondent}'
            respondent_df_clean.to_excel(writer, sheet_name=sheet_name, index=False)

        normalised_merged_experiments = pd.concat(combined_normalised_matrices, ignore_index=True)

        # Save the combined DataFrame to a new sheet in the Excel file
        normalised_merged_experiments.to_excel(writer, sheet_name='Combined_Sheet', index=False)
