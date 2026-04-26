# -*- coding: utf-8 -*-
# @Time    : 2023/8/2 16:39
# @Author  : Xiaoyi Sun
# @Site    : 
# @File    : subject_weights_converter.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import io


def parse_and_save_to_csv_spss_data(group_space_data: str, group_space_output_file: str,
                                    subject_weights_data: str, subject_weights_output_file: str):
    # Read the group_space data into a DataFrame
    df_group_space = pd.read_csv(io.StringIO(group_space_data), sep='\s+', header=None)
    df_group_space.drop(df_group_space.columns[:2], axis=1, inplace=True)

    # Save the group_space DataFrame to CSV file without the index
    df_group_space.to_csv(group_space_output_file, index=False, header=False)

    # Read the subject_weights data into a DataFrame
    df_subject_weights = pd.read_csv(io.StringIO(subject_weights_data), sep='\s+', header=None)
    df_subject_weights.drop(df_subject_weights.columns[[0, 1]], axis=1, inplace=True)

    # Save the subject_weights DataFrame to CSV file without the index
    df_subject_weights.to_csv(subject_weights_output_file, index=False, header=False)


def parse_and_save_to_csv_rotated_group_space(group_space_data: str, group_space_output_file: str,
                                              transformations: dict[str, bool]):
    # Read the group_space data into a DataFrame
    df_group_space = pd.read_csv(io.StringIO(group_space_data), sep='\s+', header=None)
    stimuli = df_group_space.iloc[:, 1]
    df_group_space.drop(df_group_space.columns[:2], axis=1, inplace=True)

    if transformations['mirror_y']:
        # Mirror along Y axis (invert positive and negative)
        df_group_space.iloc[:, 0] = df_group_space.iloc[:, 0] * -1

    if transformations['mirror_x']:
        # Mirror along X axis (invert high and low)
        df_group_space.iloc[:, 1] = df_group_space.iloc[:, 1] * -1

    if transformations['rotate_90_deg']:
        # Rotate 90deg clockwise
        rot_matrix = np.array([[0, -1], [1, 0]])
        df_group_space = pd.DataFrame(np.dot(df_group_space, rot_matrix))

    if transformations['rotate_180_deg']:
        # Rotate 180deg clockwise
        rot_matrix = np.array([[-1, 0], [0, -1]])
        df_group_space = pd.DataFrame(np.dot(df_group_space, rot_matrix))

    df_group_space.columns = ['Valence', 'Arousal']
    df_group_space['Stimuli_Names'] = stimuli
    df_group_space = df_group_space[['Stimuli_Names', 'Valence', 'Arousal']]

    # Save the group_space DataFrame to CSV file without the index
    df_group_space.to_csv(group_space_output_file, index=False)
