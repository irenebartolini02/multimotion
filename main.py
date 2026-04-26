# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:38:34 2024

@author: zeelp, ed0sh
"""

import yaml

from combine_data import concatenate_dataframes
from distance_matrix import process_and_combine_similarities, pca_group_space, lopo_process_and_combine_similarities, \
    l2po_process_and_combine_similarities
# from old_exp_data_converter import convert_old_experiment
from final_exp_data_converter import convert_multiple_experiments
from individual_map import compute_and_save_individual_ground_truth, compute_and_save_lopo_individual_ground_truth, \
    compute_and_save_l2po_individual_ground_truth
from participant_analysis import compute_survey_correlation, compute_survey_kripp_alpha
from plot_utils import plot_individual_ground_truths, plot_grouped_ground_truth_from_csv, plot_grouped_ground_truth, \
    plot_fa_group_space
from subject_weights_converter import parse_and_save_to_csv_spss_data, parse_and_save_to_csv_rotated_group_space

SURVEY_VERSION = "survey_version"
RAW_DATA = "raw_data"
PROCESSED_XLSX_DATA = "processed_xlsx_data"
OLD_XLSX_DATA = "old_xlsx_data"
MERGED_XLSX_DATA = "merged_xlsx_data"
DISTANCE_MATRIX = "distance_matrix"
GROUP_SPACE_DATA = "group_space_data"
SUBJECT_WEIGHTS_DATA = "subject_weights_data"
INDIVIDUAL_GROUND_TRUTH = "individual_ground_truth"
PLOTS_FOLDER = "plots_folder"
FA_FOLDER = "fa_folder"
LOPO_FOLDER = "lopo_folder"
L2PO_FOLDER = "l2po_folder"
STIMULI_NAMES = "stimuli_names"
PARTICIPANTS_TO_EXCLUDE = "participants_to_exclude"
TRANSFORMATIONS = "transformations"

# Transfer SPSS data to csv file, replace when needed
group_space_spss = """\
    1      HN_1      -1.8956    .3075
    2      HN_2_H    -1.2992    .6964
    3      HN_2_L     -.4195  -1.3489
    4      HN_3_H    -1.2550    .6806
    5      HN_3_L     -.2346  -1.2921
    6      HN_4      -1.1828    .6975
    7      HN_5      -1.3179    .3777
    8      HN_6      -1.6433    .0560
    9      HN_7      -1.2881   -.5102
   10      HN_8      -1.3216    .1558
   11      HP_1_H      .6653   1.2075
   12      HP_1_L      .9047   -.3274
   13      HP_2       1.0748    .8734
   14      HP_3_H      .8900   1.2954
   15      HP_3_L      .7330   -.6835
   16      HP_4        .8687    .9190
   17      HP_5        .6492   1.0740
   18      HP_6        .8273   1.0619
   19      HP_7_H      .8557   1.2567
   20      HP_7_L      .9868   -.1951
   21      HP_8        .6073   1.0609
   22      LN_1        .0226  -1.6816
   23      LN_2       -.7629  -1.3326
   24      LN_3      -1.9245    .1572
   25      LN_4       -.6084  -1.5076
   26      LN_5       -.1932  -1.3833
   27      LN_6       -.3049  -1.5122
   28      LN_7_N     -.8879  -1.4387
   29      LN_7_P      .2450  -1.0552
   30      LN_8       -.6756  -1.3987
   31      LP_1        .9773    .9550
   32      LP_2        .9573   1.0176
   33      LP_3       1.1328    .0878
   34      LP_4        .9596    .8970
   35      LP_5       1.0429    .9575
   36      LP_6        .6929  -1.1836
   37      LP_7       1.1013    .1221
   38      LP_8       1.0205    .9363
"""


subject_weights_spss = """\
      1     .0732    .6380    .4154
      2     .2211    .4091    .3384
      3     .0534    .5806    .3097
      4     .0743    .7065    .3647
      5     .0873    .4370    .2910
      6     .2067    .6323    .2636
      7     .1538    .6462    .4786
      8     .2066    .2983    .2408
      9     .1499    .4584    .3373
     10     .0746    .6425    .3315
     11     .1179    .6224    .2998
     12     .1777    .7603    .3325
     13     .1778    .3634    .2798
     14     .1462    .5749    .4205
     15     .2292    .5760    .4829
     16     .4402    .8377    .2286
     17     .3736    .7837    .2437
     18     .3082    .3742    .3595
     19     .1331    .4090    .2929
     20     .0576    .5431    .3450
     21     .2466    .5521    .4767
     22     .2592    .3426    .3022
     23     .0985    .6119    .3039
     24     .0437    .5593    .3476
     25     .0610    .5586    .3567
     26     .2874    .3355    .3107
     27     .2441    .4804    .4130
     28     .1098    .6260    .3054
     29     .1279    .5754    .4087
     30     .1548    .7431    .3373
     31     .4851    .8508    .2113
     32     .1378    .4862    .3509
     33     .0790    .7294    .3737
     34     .1896    .2805    .2202
     35     .0210    .7320    .4390
     36     .1195    .6545    .3144
     37     .2106    .3226    .2622
     38     .1540    .5152    .3816
     39     .1832    .5044    .3918
     40     .0117    .7381    .4362
     41     .1862    .5606    .4376
     42     .1457    .4554    .3329
     43     .3560    .2991    .3134
     44     .1797    .7332    .3195
     45     .1088    .4491    .3094
     46     .2909    .6485    .2343
     47     .3615    .8331    .2650
     48     .0769    .5662    .3708
     49     .2499    .7658    .2970
     50     .0003    .5986    .3472
     51     .1938    .6852    .2918
     52     .3371    .7185    .2391
     53     .1246    .6176    .2943
"""


def main():
    # Read yaml configuration file
    with open('repo.yml', 'r') as file:
        repository_data = yaml.safe_load(file)

    survey_version: str = repository_data[SURVEY_VERSION]
    raw_data_paths: list[str] = repository_data[RAW_DATA]
    processed_xlsx_data_path: str = repository_data[PROCESSED_XLSX_DATA]
    old_xlsx_data_paths: list[str] = repository_data[OLD_XLSX_DATA]
    merged_xlsx_data_path: str = repository_data[MERGED_XLSX_DATA]
    distance_matrix_path: str = repository_data[DISTANCE_MATRIX]
    group_space_data_path: str = repository_data[GROUP_SPACE_DATA]
    subject_weights_data_path: str = repository_data[SUBJECT_WEIGHTS_DATA]
    individual_ground_truth_path: str = repository_data[INDIVIDUAL_GROUND_TRUTH]
    plots_folder_path: str = repository_data[PLOTS_FOLDER]
    fa_folder_path: str = repository_data[FA_FOLDER]
    lopo_folder_path: str = repository_data[LOPO_FOLDER]
    l2po_folder_path: str = repository_data[L2PO_FOLDER]
    stimuli_names: list[str] = repository_data[STIMULI_NAMES]
    participants_to_exclude: list[str] = repository_data[PARTICIPANTS_TO_EXCLUDE]
    transformations: dict[str, bool] = repository_data[TRANSFORMATIONS]

    if survey_version in ['version_1', 'version_5', 'version_final']:
        emotions_to_use = ['Amused*', 'Angry*', 'Anxious*', 'Bored*', 'Calm*', 'Content*', 'Excited*', 'Fearful*',
                           'Happy*', 'Negative*', 'Positive*', 'Sad*']

        positive_emotions = ['Happy*', 'Positive*']
        negative_emotions = ['Angry*', 'Fearful*', 'Negative*', 'Sad*', 'Anxious*']

    else:
        emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
                           'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Content
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Calm
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Calm and Content
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Bored
        # emotions_to_use = ['Amused*', 'Angry*', 'Calm*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Bored, Calm and Content
        # emotions_to_use = ['Amused*', 'Angry*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Content and Excited
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Disgust*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Amused, Excited and Content
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Disgust*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Amused, Excited, Content and Calm
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Disgust*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Without Amused
        # emotions_to_use = ['Angry*', 'Bored*', 'Calm*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Negative*', 'Positive*', 'Sad*']

        # Only few
        # emotions_to_use = ['Negative*', 'Positive*']

        # Without positive and negative
        # emotions_to_use = ['Amused*', 'Angry*', 'Bored*', 'Calm*', 'Content*', 'Disgust*', 'Excited*', 'Fearful*',
        #                    'Happy*', 'Sad*']

        # Without Positive, Negative, Bored, Calm and Content
        # emotions_to_use = ['Amused*', 'Angry*', 'Disgust*', 'Excited*', 'Fearful*', 'Happy*', 'Sad*']

        positive_emotions = ['Happy*', 'Positive*']
        negative_emotions = ['Angry*', 'Fearful*', 'Negative*', 'Sad*', 'Disgust*']

    while True:
        print("\nChoose a function:")
        print("1. convert survey from txt to csv")  # if you haven't had the .csv file for those survey file you will run this.
        print("2. merge new data to existing dataframe")  # when you have more than one .csv survey file you can use this to merge it.
        print("3. compute distance matrix")  # compute distance of each emotions by using Euclidean theorem
        print("4. convert group space and subject weights from string to csv")
        print("5. compute individual ground_truth")
        print("6. compute individual ground_truth maps")
        print("7. compute group space map")
        print("8. compute group space using PCA")
        print("9. compute analyses on surveys")
        print("10. compute factor analysis plot and individual ground truth")
        print("11. save rotated group space csv")
        print("12. compute distance matrix with LOPO")
        print("13. compute individual gt matrix with LOPO")
        print("14. compute distance matrix with L2PO")
        print("15. compute individual gt matrix with L2PO")
        print("0. Exit")

        choice = input("Enter the number of the function you want to choose: ")

        if choice == "1":
            convert_multiple_experiments(raw_data_paths,
                                         processed_xlsx_data_path,
                                         emotions_to_use,
                                         version=survey_version,
                                         export_single_csvs=False)

        elif choice == "2":
            concatenate_dataframes(old_xlsx_data_paths, processed_xlsx_data_path, merged_xlsx_data_path,
                                   emotions_to_use, positive_emotions, negative_emotions, participants_to_exclude)

        elif choice == "3":
            process_and_combine_similarities(merged_xlsx_data_path, distance_matrix_path, participants_to_exclude,
                                             emotions_to_use, fa_folder_path)

        # need SPSS
        elif choice == "4":
            parse_and_save_to_csv_spss_data(group_space_spss, group_space_data_path, subject_weights_spss,
                                            subject_weights_data_path)

        elif choice == "5":
            compute_and_save_individual_ground_truth(individual_ground_truth_path, merged_xlsx_data_path,
                                                     participants_to_exclude, group_space_data_path,
                                                     subject_weights_data_path, stimuli_names, transformations)

        elif choice == "6":
            plot_individual_ground_truths(individual_ground_truth_path, plots_folder_path)

        elif choice == "7":
            plot_grouped_ground_truth_from_csv(group_space_data_path, plots_folder_path, stimuli_names, transformations)

        elif choice == "8":
            matrix = pca_group_space(merged_xlsx_data_path, participants_to_exclude, emotions_to_use)
            plot_grouped_ground_truth(matrix, plots_folder_path, stimuli_names, transformations)

        elif choice == "9":
            compute_survey_correlation(merged_xlsx_data_path, participants_to_exclude, emotions_to_use)
            compute_survey_kripp_alpha(merged_xlsx_data_path, participants_to_exclude)

        elif choice == "10":
            plot_fa_group_space(fa_folder_path, merged_xlsx_data_path, participants_to_exclude)

        elif choice == "11":
            parse_and_save_to_csv_rotated_group_space(group_space_spss, group_space_data_path, transformations)

        elif choice == "12":
            lopo_process_and_combine_similarities(merged_xlsx_data_path, distance_matrix_path, participants_to_exclude,
                                                  emotions_to_use, fa_folder_path)

        elif choice == "13":
            compute_and_save_lopo_individual_ground_truth(individual_ground_truth_path, merged_xlsx_data_path,
                                                          participants_to_exclude, stimuli_names, transformations,
                                                          lopo_folder_path, group_space_data_path,
                                                          subject_weights_data_path, plots_folder_path)
        elif choice == "14":
            l2po_process_and_combine_similarities(merged_xlsx_data_path, distance_matrix_path, participants_to_exclude,
                                                  emotions_to_use, fa_folder_path)

        elif choice == "15":
            compute_and_save_l2po_individual_ground_truth(individual_ground_truth_path, merged_xlsx_data_path,
                                                          participants_to_exclude, stimuli_names, transformations,
                                                          l2po_folder_path, group_space_data_path,
                                                          subject_weights_data_path, plots_folder_path)

        elif choice == "0":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice.")


if __name__ == '__main__':
    main()
