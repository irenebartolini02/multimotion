# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:27
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : individual_map.py
# @Software: PyCharm
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plot_utils import plot_grouped_ground_truth_from_csv
from subject_weights_converter import parse_and_save_to_csv_spss_data

MAPPING_FIXES_14_01_25 = {
    'LN_8': 'LN_9',
    'HN_8': 'HN_9',
    'LP_8': 'LP_9',
    'HP_8': 'HP_9'
}

MIN = -1.6860734220000002
MAX = 1.6860734220000002


def compute_and_save_individual_ground_truth(output_file: str, merged_xlsx_data_path: str,
                                             participants_to_exclude: list[str], group_space_data_path: str,
                                             subject_weights_data_path: str, stimuli_names: list[str],
                                             transformations: dict[str, bool],):

    group_space = np.genfromtxt(group_space_data_path, delimiter=',')
    subject_weights = np.genfromtxt(subject_weights_data_path, delimiter=',')

    # Get participants codes
    df = pd.read_excel(merged_xlsx_data_path, sheet_name="Combined_Sheet")
    df_filtered = df[~df['respondent'].isin(participants_to_exclude)]
    respondents = df_filtered['respondent'].drop_duplicates()

    # Create empty array to store individual maps
    individual_maps = np.empty((subject_weights.shape[0], group_space.shape[0], group_space.shape[1]))

    # Calculate individual maps
    for i in range(subject_weights.shape[0]):
        individual_map = group_space * subject_weights[i]

        if transformations['mirror_y']:
            # Mirror along Y axis (invert positive and negative)
            individual_map[:, 0] = individual_map[:, 0] * -1

        if transformations['mirror_x']:
            # Mirror along X axis (invert high and low)
            individual_map[:, 1] = individual_map[:, 1] * -1

        if transformations['rotate_90_deg']:
            # Rotate 90deg clockwise
            rot_matrix = np.array([[0, -1], [1, 0]])
            individual_map = np.dot(individual_map, rot_matrix)

        if transformations['rotate_180_deg']:
            # Rotate 180deg clockwise
            rot_matrix = np.array([[-1, 0], [0, -1]])
            individual_map = np.dot(individual_map, rot_matrix)

        individual_maps[i] = individual_map

    # Save the individual maps to CSV
    headers = ["Valence", "Arousal"]
    reshaped_individual_maps = individual_maps.reshape(-1, 2)

    """
    # Normalisation: Centered MinMax normalisation using max(|min(x) - 0.1 * |min(x)||, |max(x) + 0.1 * |max(x)||)
    normalised_individual_maps = np.empty(reshaped_individual_maps.shape)
    max_val = reshaped_individual_maps[:, 0].max()
    min_val = reshaped_individual_maps[:, 0].min()
    max_val = max(
        abs(min_val - 0.1 * abs(min_val)),
        abs(max_val + 0.1 * abs(max_val))
    )
    min_val = -max_val

    max_ar = reshaped_individual_maps[:, 1].max()
    min_ar = reshaped_individual_maps[:, 1].min()
    max_ar = max(
        abs(min_ar - 0.1 * abs(min_ar)),
        abs(max_ar + 0.1 * abs(max_ar))
    )
    min_ar = -max_ar
    max_norm = max(max_ar, max_val)
    min_norm = -max_norm
    print(f"Min: {min_norm}, Max: {max_norm}")
    normalised_individual_maps[:, 0] = ((reshaped_individual_maps[:, 0] - min_norm) / (max_norm - min_norm)) * 2 - 1
    normalised_individual_maps[:, 1] = ((reshaped_individual_maps[:, 1] - min_norm) / (max_norm - min_norm)) * 2 - 1
    """
    # normalised_individual_maps = reshaped_individual_maps

    # """
    normalised_individual_maps = np.empty(reshaped_individual_maps.shape)
    normalised_individual_maps[:, 0] = ((reshaped_individual_maps[:, 0] - MIN) / (MAX - MIN)) * 2 - 1
    normalised_individual_maps[:, 1] = ((reshaped_individual_maps[:, 1] - MIN) / (MAX - MIN)) * 2 - 1
    # """

    data = pd.DataFrame(normalised_individual_maps.round(decimals=6), columns=headers)

    # Calculate participant numbers
    participant_name_times_stimuli_number = []

    for respondent in respondents:
        participant_name_times_stimuli_number += [respondent] * len(stimuli_names)

    # Add participant codes as a new column, assuming that the order of the participants is the same as in combine_data
    data['Participant'] = participant_name_times_stimuli_number

    # Create a new column with repeated values from "Stimulus Name"
    data["Stimulus_Name"] = stimuli_names * len(respondents)
    new_column_order_index = [2, 3, 0, 1]  # Assuming the order: Valence, Arousal, Participant, Stimulus_Name

    # Rename stimuli
    data['Stimulus_Name'] = data['Stimulus_Name'].replace(MAPPING_FIXES_14_01_25)

    # Use the iloc property to reorder the columns based on their index positions
    data = data.iloc[:, new_column_order_index]
    # Save the modified data to the same CSV file
    data.to_csv(output_file, index=False)


def compute_and_save_lopo_individual_ground_truth(output_file: str | Path,
                                                  merged_xlsx_data_path: str | Path,
                                                  participants_to_exclude: list[str],
                                                  stimuli_names: list[str],
                                                  transformations: dict[str, bool],
                                                  lopo_folder_path: str | Path,
                                                  group_space_data_path: str | Path,
                                                  subject_weights_data_path: str | Path,
                                                  plots_folder_path: str | Path):

    for participant_file in Path(lopo_folder_path).iterdir():
        try:
            with open(participant_file, 'r') as file:
                print(participant_file)
                content = file.read()
        except UnicodeDecodeError:
            with open(participant_file, 'r', encoding="utf-16") as file:
                print(participant_file)
                content = file.read()

        cur_part = participant_file.stem
        cur_participants_to_exclude = participants_to_exclude + [cur_part]

        cur_individual_ground_truth_path = (
            str(Path(output_file).parent / Path(f"{Path(output_file).stem}_no_{cur_part}{Path(output_file).suffix}"))
        )

        weights_df = None
        stimulus_df = None

        # Identifica le tabelle nel file
        # Cerco pattern di righe con numeri e spazi che formano tabelle
        # Tabella 1: Stimulus data
        stimulus_pattern = r'(Stimulus\s+Stimulus\s+1\s+2\s+Number\s+Name.*?)(?=Subject weights|$)'
        stimulus_match = re.search(stimulus_pattern, content, re.DOTALL)
        if stimulus_match:
            stimulus_text = stimulus_match.group(1)
            # Rimuovo le prime righe di intestazione e le righe vuote
            data_lines = [line for line in stimulus_text.split('\n') if re.search(r'^\s*\d+\s+\w+', line)]
            stimulus_data = '\n'.join(data_lines)
            stimulus_df = pd.read_fwf(io.StringIO(stimulus_data),
                                    decimal=",",
                                    names=['Stimulus_Number', 'Stimulus_Name', 'Dimension_1', 'Dimension_2'])
            # print("Tabella degli stimoli:")
            # print(stimulus_df)

        # Tabella 2: Subject Weights
        weights_pattern = r'(Subject\s+Weird-\s+1\s+2\s+Number\s+ness.*?)(?=Overall importance|$)'
        weights_match = re.search(weights_pattern, content, re.DOTALL)
        if weights_match:
            weights_text = weights_match.group(1)
            # Estraggo solo le righe con i dati numerici
            data_lines = [line for line in weights_text.split('\n') if re.search(r'^\s*\d+\s+,\d+', line)]
            weights_data = '\n'.join(data_lines)
            weights_df = pd.read_fwf(io.StringIO(weights_data),
                                    decimal=",",
                                    names=['Subject_Number', 'Weirdness', 'Dimension_1', 'Dimension_2'])

        if weights_df is None or stimulus_df is None:
            raise ValueError

        hn_1_valence = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HN_1'].loc[:, 'Dimension_1'].iloc[0])
        hn_1_arousal = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HN_1'].loc[:, 'Dimension_2'].iloc[0])

        hp_1_valence = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HP_1_H'].loc[:, 'Dimension_1'].iloc[0])
        hp_1_arousal = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HP_1_H'].loc[:, 'Dimension_2'].iloc[0])

        if (hn_1_valence > 0 and hn_1_arousal > 0) and (hp_1_valence < 0 and hp_1_arousal > 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": True,
                "mirror_x": False
            }
        elif (hn_1_valence < 0 and hn_1_arousal < 0) and (hp_1_valence > 0 and hp_1_arousal < 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": True
            }
        elif (hn_1_valence > 0 and hn_1_arousal < 0) and (hp_1_valence < 0 and hp_1_arousal < 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": True,
                "mirror_y": False,
                "mirror_x": False
            }
        elif (hn_1_valence < 0 and hn_1_arousal > 0) and (hp_1_valence > 0 and hp_1_arousal > 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": False
            }
        else:
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": True
            }

        group_space_spss_io = io.StringIO()
        stimulus_df.to_csv(group_space_spss_io, index=False, header=False, sep='\t')
        group_space_spss = group_space_spss_io.getvalue()

        subject_weights_spss_io = io.StringIO()
        weights_df.to_csv(subject_weights_spss_io, index=False, header=False, sep='\t')
        subject_weights_spss = subject_weights_spss_io.getvalue()

        parse_and_save_to_csv_spss_data(group_space_spss, group_space_data_path, subject_weights_spss,
                                        subject_weights_data_path)

        # plot_grouped_ground_truth_from_csv(group_space_data_path, plots_folder_path, stimuli_names, transformations)

        compute_and_save_individual_ground_truth(cur_individual_ground_truth_path, merged_xlsx_data_path,
                                                 cur_participants_to_exclude, group_space_data_path,
                                                 subject_weights_data_path, stimuli_names, transformations)


def compute_and_save_l2po_individual_ground_truth(output_file: str | Path,
                                                  merged_xlsx_data_path: str | Path,
                                                  participants_to_exclude: list[str],
                                                  stimuli_names: list[str],
                                                  transformations: dict[str, bool],
                                                  lopo_folder_path: str | Path,
                                                  group_space_data_path: str | Path,
                                                  subject_weights_data_path: str | Path,
                                                  plots_folder_path: str | Path):

    for participants_file in Path(lopo_folder_path).iterdir():
        try:
            with open(participants_file, 'r') as file:
                print(participants_file)
                content = file.read()
        except UnicodeDecodeError:
            with open(participants_file, 'r', encoding="utf-16") as file:
                print(participants_file)
                content = file.read()

        cur_file = participants_file.stem
        cur_part1 = cur_file.split('_')[0]
        cur_part2 = cur_file.split('_')[1]
        cur_participants_to_exclude = participants_to_exclude + [cur_part1, cur_part2]

        cur_individual_ground_truth_path = (
            str(Path(output_file).parent / Path(f"{Path(output_file).stem}_no_{cur_part1}_no_{cur_part2}{Path(output_file).suffix}"))
        )

        weights_df = None
        stimulus_df = None

        # Identifica le tabelle nel file
        # Cerco pattern di righe con numeri e spazi che formano tabelle
        # Tabella 1: Stimulus data
        stimulus_pattern = r'(Stimulus\s+Stimulus\s+1\s+2\s+Number\s+Name.*?)(?=Subject weights|$)'
        stimulus_match = re.search(stimulus_pattern, content, re.DOTALL)
        stimulus_pattern = r'(Stimulus\s+Stimulus\s+1\s+2\s+Number\s+Name.*?)(?=Subject weights|$)'
        stimulus_match = re.search(stimulus_pattern, content, re.DOTALL)
        if stimulus_match:
            stimulus_text = stimulus_match.group(1)
            # Rimuovo le prime righe di intestazione e le righe vuote
            data_lines = [line for line in stimulus_text.split('\n') if re.search(r'^\s*\d+\s+\w+', line)]
            stimulus_data = '\n'.join(data_lines)
            stimulus_df = pd.read_fwf(io.StringIO(stimulus_data),
                                    decimal=",",
                                    names=['Stimulus_Number', 'Stimulus_Name', 'Dimension_1', 'Dimension_2'])
            # print("Tabella degli stimoli:")
            # print(stimulus_df)

        # Tabella 2: Subject Weights
        weights_pattern = r'(Subject\s+Weird-\s+1\s+2\s+Number\s+ness.*?)(?=Overall importance|$)'
        weights_match = re.search(weights_pattern, content, re.DOTALL)
        if weights_match:
            weights_text = weights_match.group(1)
            # Estraggo solo le righe con i dati numerici
            data_lines = [line for line in weights_text.split('\n') if re.search(r'^\s*\d+\s+,\d+', line)]
            weights_data = '\n'.join(data_lines)
            weights_df = pd.read_fwf(io.StringIO(weights_data),
                                    decimal=",",
                                    names=['Subject_Number', 'Weirdness', 'Dimension_1', 'Dimension_2'])

        if weights_df is None or stimulus_df is None:
            raise ValueError

        hn_1_valence = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HN_1'].loc[:, 'Dimension_1'].iloc[0])
        hn_1_arousal = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HN_1'].loc[:, 'Dimension_2'].iloc[0])

        hp_1_valence = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HP_1_H'].loc[:, 'Dimension_1'].iloc[0])
        hp_1_arousal = float(stimulus_df[stimulus_df['Stimulus_Name'] == 'HP_1_H'].loc[:, 'Dimension_2'].iloc[0])

        if (hn_1_valence > 0 and hn_1_arousal > 0) and (hp_1_valence < 0 and hp_1_arousal > 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": True,
                "mirror_x": False
            }
        elif (hn_1_valence < 0 and hn_1_arousal < 0) and (hp_1_valence > 0 and hp_1_arousal < 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": True
            }
        elif (hn_1_valence > 0 and hn_1_arousal < 0) and (hp_1_valence < 0 and hp_1_arousal < 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": True,
                "mirror_y": False,
                "mirror_x": False
            }
        elif (hn_1_valence < 0 and hn_1_arousal > 0) and (hp_1_valence > 0 and hp_1_arousal > 0):
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": False
            }
        else:
            transformations = {
                "rotate_90_deg": False,
                "rotate_180_deg": False,
                "mirror_y": False,
                "mirror_x": True
            }

        group_space_spss_io = io.StringIO()
        stimulus_df.to_csv(group_space_spss_io, index=False, header=False, sep='\t')
        group_space_spss = group_space_spss_io.getvalue()

        subject_weights_spss_io = io.StringIO()
        weights_df.to_csv(subject_weights_spss_io, index=False, header=False, sep='\t')
        subject_weights_spss = subject_weights_spss_io.getvalue()

        parse_and_save_to_csv_spss_data(group_space_spss, group_space_data_path, subject_weights_spss,
                                        subject_weights_data_path)

        # plot_grouped_ground_truth_from_csv(group_space_data_path, plots_folder_path, stimuli_names, transformations)

        compute_and_save_individual_ground_truth(cur_individual_ground_truth_path, merged_xlsx_data_path,
                                                 cur_participants_to_exclude, group_space_data_path,
                                                 subject_weights_data_path, stimuli_names, transformations)