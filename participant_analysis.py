from typing import Tuple, List, Dict

from krippendorff import krippendorff
from typing_extensions import deprecated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pin


def compute_survey_correlation(input_file: str, participants_to_exclude, emotions_to_use):
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    df_no_meditations = df[~df['stimulus'].str.contains("meditation")]
    df_clean = df_no_meditations.drop(['relaxation*', 'Are you familiar with this video?'], axis=1)
    stimuli = df_clean['stimulus'].drop_duplicates()
    print(stimuli)

    # Imputation: replace nans with the average value across participants
    for stimulus in stimuli:
        ratings = df_clean.loc[df_clean['stimulus'] == stimulus, emotions_to_use]
        ratings = ratings.fillna(ratings.mean())
        df_clean.loc[df_clean['stimulus'] == stimulus, emotions_to_use] = ratings

    min_corr = 0
    part_min_corr = None
    excl_parts = participants_to_exclude.copy()
    respondent_col = 'respondent'
    stimulus_col = 'stimulus'
    target = 'Emotion'

    min_corrs = []
    total_stats = []
    iter = 0
    while min_corr < 0.4 or np.isnan(min_corr):
        if part_min_corr is not None:
            print(f"Discarding {part_min_corr} with {min_corr} correlation")
            excl_parts.append(part_min_corr)
        df_filt = df_clean.loc[~df_clean['respondent'].isin(excl_parts)]
        min_corr, part_min_corr, stats_df = compute_correlation(df_filt, respondent_col, stimulus_col, target)
        stats_df['iter'] = iter
        total_stats.append(stats_df)
        iter += 1
        min_corrs.append(min_corr)

    total_stats_df = pd.concat(total_stats)
    grouped_sats = total_stats_df.groupby('emotion')

    """
    plt.figure()
    plt.title("mean")
    for emotion, emotion_stats in grouped_sats:
        plt.plot(emotion_stats['mean'].values, label=emotion, marker='o')
    plt.legend(ncols=3)
    plt.show()

    plt.figure()
    plt.title("std")
    for emotion, emotion_stats in grouped_sats:
        plt.plot(emotion_stats['std'].values, label=emotion, marker='o')
    plt.legend(ncols=3)
    plt.show()

    plt.figure()
    plt.title("min")
    for emotion, emotion_stats in grouped_sats:
        plt.plot(emotion_stats['min'].values, label=emotion, marker='o')
    plt.legend(ncols=3)
    plt.show()

    plt.figure()
    plt.title("max")
    for emotion, emotion_stats in grouped_sats:
        plt.plot(emotion_stats['max'].values, label=emotion, marker='o')
    plt.legend(ncols=3)
    plt.show()
    """

    avg_mean = total_stats_df.groupby('iter')['mean'].mean()
    min_mean = total_stats_df.groupby('iter')['min'].mean()
    max_mean = total_stats_df.groupby('iter')['max'].mean()
    plt.figure()
    plt.title("Overall Average Participant Correlation per Iteration")
    plt.plot(avg_mean, label='Average', marker='o')
    plt.plot(min_mean, label='Minimum', marker='o')
    plt.plot(max_mean, label='Maximum', marker='o')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Minimum Participant Correlation per Iteration")
    plt.plot(min_corrs, marker='o')
    plt.show()

    melted_df = df_filt.melt(id_vars=[respondent_col, stimulus_col],
                        var_name=target,
                        value_name="Value")

    corrs = []
    for emotion, matrix in melted_df.groupby(target):
        df_for_cronbach = matrix.pivot(index=stimulus_col, columns=respondent_col, values='Value')
        corr = df_for_cronbach.corr(method='spearman')
        corrs.append(corr)

    mean_corr = pd.concat(corrs).mean()
    print(mean_corr.mean(), len(mean_corr), mean_corr, len(excl_parts), set(excl_parts) - set(participants_to_exclude))


def compute_ar_val_correlation(input_file: str):
    # Import the data from a CSV file
    df = pd.read_csv(input_file)

    min_corr = 0
    part_min_corr = None
    excl_parts = []
    respondent_col = 'Participant'
    stimulus_col = 'Stimulus_Name'
    target = 'Target'

    while min_corr < 0.4:
        if part_min_corr is not None:
            excl_parts.append(part_min_corr)
        df_filt = df[~df[respondent_col].isin(excl_parts)]
        min_corr, part_min_corr, stats = compute_correlation(df_filt, respondent_col, stimulus_col, target)

    melted_df = df_filt.melt(id_vars=[respondent_col, stimulus_col],
                        var_name=target,
                        value_name="Value")

    corrs = []
    for emotion, matrix in melted_df.groupby(target):
        df_for_cronbach = matrix.pivot(index=stimulus_col, columns=respondent_col, values='Value')
        corr = df_for_cronbach.corr(method='spearman')
        corrs.append(corr)

    mean_corr = pd.concat(corrs).mean()
    print(mean_corr.mean(), len(mean_corr), mean_corr, len(excl_parts), excl_parts)


def compute_correlation(df: pd.DataFrame, respondent_col: str, stimulus_col: str, target: str) -> Tuple[float, float, pd.DataFrame]:
    melted_df = df.melt(id_vars=[respondent_col, stimulus_col],
                                 var_name=target,
                                 value_name="Value")

    emotion_corrs = {}
    stats = []
    for emotion, matrix in melted_df.groupby(target):
        df_for_cronbach = matrix.pivot(index=stimulus_col, columns=respondent_col, values='Value')
        corr = df_for_cronbach.corr(method='spearman')
        emotion_corrs[emotion] = corr

    itemlist = sorted(emotion_corrs.items(), reverse=True, key=lambda C: C[1].mean().mean())
    emotion_corrs = dict(itemlist)

    for emotion, corr in emotion_corrs.items():
        cur_stats = {}
        cur_stats['mean'] = round(corr.mean().mean(), 3)
        cur_stats['std'] = round(corr.mean().std(), 3)
        cur_stats['min'] = round(corr.mean().min(), 3)
        cur_stats['max'] = round(corr.mean().max(), 3)
        cur_stats['emotion'] = emotion
        stats.append(cur_stats)

        """
        print(f"{emotion}")
        print(f"\tMean: {cur_stats['mean']}")
        print(f"\tStd: {cur_stats['std']}")
        print(f"\tLowest: {cur_stats['min']} <-> {corr.mean().idxmin()}")
        print(f"\t\tDistance from the average: {abs(cur_stats['mean'] - cur_stats['min'])}")
        print(f"\tHighest: {cur_stats['max']} <-> {corr.mean().idxmax()}")
        print(f"\t\tDistance from the average: {abs(cur_stats['mean'] - cur_stats['max'])}")
        """

    mean_corr = pd.concat(emotion_corrs.values()).mean()

    if mean_corr.hasnans:
        most_uncorr_part = mean_corr.index[mean_corr.isnull()][0]
        min_corr = np.nan
    else:
        most_uncorr_part = mean_corr.index[mean_corr.argmin()]
        min_corr = mean_corr.min()

    return min_corr, most_uncorr_part, pd.DataFrame(stats)


def compute_survey_kripp_alpha(input_file: str, participants_to_exclude):
    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    df_no_meditations = df[~df['stimulus'].str.contains("meditation")]
    df_clean = df_no_meditations.drop(['relaxation*', 'Are you familiar with this video?'], axis=1)
    df_clean = df_clean[~df_clean['respondent'].isin(participants_to_exclude)]

    respondent_col = 'respondent'
    stimulus_col = 'stimulus'
    target = 'Emotion'

    mean_alpha = compute_krippendorff_alpha(df_clean, respondent_col, stimulus_col, target)
    print(mean_alpha)

    best_alpha = 0.0
    best_included_participants = df_clean[respondent_col].unique().tolist()
    best_alpha, best_included_participants = \
        find_best_combination_recursively_greedy_kripp_alpha(df_clean, best_included_participants.copy(),
                                                             best_alpha, best_included_participants,
                                                             respondent_col, stimulus_col, target)

    print(best_alpha, len(best_included_participants), best_included_participants)


def find_best_combination_recursively_greedy_kripp_alpha(df: pd.DataFrame, included_participants: list[str],
                                                         best_alpha: float, best_included_participants: list[str],
                                                         respondent_col: str, stimulus_col: str, target: str
                                                         ) -> Tuple[float, List[str]]:
    if 0 < best_alpha < 0.4:
        return best_alpha, best_included_participants

    alphas_and_participants = []
    for participant in included_participants:
        included_participants.remove(participant)
        df_filtered = df[df[respondent_col].isin(included_participants)]

        if all(x == y for x, y in zip(included_participants, best_included_participants)):
            continue

        mean_alpha = compute_krippendorff_alpha(df_filtered, respondent_col, stimulus_col, target)
        alphas_and_participants.append((mean_alpha, included_participants.copy()))
        included_participants.append(participant)

    # Sort by alpha descending
    alphas_and_participants.sort(reverse=True, key=lambda x: x[0])

    # Select the branch with the highest alpha
    if alphas_and_participants:
        alpha, included_participants = alphas_and_participants[0]

        if alpha > best_alpha:
            best_alpha = alpha
            best_included_participants = included_participants
            print(best_alpha, len(best_included_participants), df[~df[respondent_col].isin(included_participants)][respondent_col].unique())

        df_filtered = df[df[respondent_col].isin(included_participants)]
        if df_filtered.shape[0] < 32 * 3:
            return best_alpha, best_included_participants

        best_alpha, best_included_participants = \
            find_best_combination_recursively_greedy_kripp_alpha(df_filtered, included_participants.copy(),
                                                                 best_alpha, best_included_participants,
                                                                 respondent_col, stimulus_col, target)

    return best_alpha, best_included_participants


def compute_krippendorff_alpha(df: pd.DataFrame, respondent_col: str, stimulus_col: str, target: str) -> float:
    melted_df = df.melt(id_vars=[respondent_col, stimulus_col],
                        var_name=target,
                        value_name="Value")

    alphas = []
    for emotion, matrix in melted_df.groupby(target):
        df_for_kripp = matrix.pivot(index=respondent_col, columns=stimulus_col, values='Value')
        alpha = krippendorff.alpha(reliability_data=df_for_kripp)
        # print(f"{emotion}: {round(float(alpha), 3)}")
        alphas.append(alpha)

    return float(np.mean(alphas))


@deprecated("Cronbach alpha is meaningless for a high number of items (i.e. participants)")
def compute_survey_cronbach_alpha(input_file: str, participants_to_exclude):
    # Import the data from a CSV file
    df = pd.read_excel(input_file, sheet_name="Combined_Sheet")

    df_no_meditations = df[~df['stimulus'].str.contains("meditation")]
    df_clean = df_no_meditations.drop(['relaxation*', 'Are you familiar with this video?'], axis=1)
    df_clean = df_clean[~df_clean['respondent'].isin(participants_to_exclude)]

    mean_alpha = compute_cronbach_alpha(df_clean, 'respondent', 'stimulus', 'Emotion')

    best_alpha = 0.0
    best_included_participants = df_clean['respondent'].unique().tolist()
    best_alpha, best_included_participants = find_best_combination_recursively_greedy_alpha(df_clean,
                                                                                                best_included_participants.copy(),
                                                                                                best_alpha,
                                                                                                best_included_participants)
    print(best_alpha, len(best_included_participants), best_included_participants)

@deprecated("Cronbach alpha is meaningless for a high number of items (i.e. participants)")
def find_best_combination_recursively_greedy_alpha(df: pd.DataFrame, included_participants: list[str],
                                             best_alpha: float, best_included_participants: list[str]) -> Tuple[float, List[str]]:
    if 0 < best_alpha < 0.97:
        return best_alpha, best_included_participants

    alphas_and_participants = []
    for participant in included_participants:
        included_participants.remove(participant)
        df_filtered = df[df['respondent'].isin(included_participants)]

        if all(x == y for x, y in zip(included_participants, best_included_participants)):
            continue

        mean_alpha = compute_cronbach_alpha(df_filtered, 'respondent', 'stimulus', 'Emotion')
        alphas_and_participants.append((mean_alpha, included_participants.copy()))
        included_participants.append(participant)

    # Sort by alpha descending
    alphas_and_participants.sort(reverse=True, key=lambda x: x[0])

    # Select the branch with the highest alpha
    if alphas_and_participants:
        alpha, included_participants = alphas_and_participants[0]

        if alpha > best_alpha:
            best_alpha = alpha
            best_included_participants = included_participants
            print(best_alpha, len(best_included_participants), best_included_participants)

        df_filtered = df[df['respondent'].isin(included_participants)]
        if df_filtered.shape[0] < 32 * 3:
            return best_alpha, best_included_participants

        best_alpha, best_included_participants = find_best_combination_recursively_greedy_alpha(df_filtered,
                                                                                                included_participants.copy(),
                                                                                                best_alpha,
                                                                                                best_included_participants)

    return best_alpha, best_included_participants

@deprecated("Cronbach alpha is meaningless for a high number of items (i.e. participants)")
def compute_cronbach_alpha(df: pd.DataFrame, respondent_col: str, stimulus_col: str, target: str) -> float:
    melted_df = df.melt(id_vars=[respondent_col, stimulus_col],
                                 var_name=target,
                                 value_name="Value")

    alphas = []
    for emotion, matrix in melted_df.groupby(target):
        df_for_cronbach = matrix.pivot(index=stimulus_col, columns=respondent_col, values='Value')
        ca = pin.cronbach_alpha(df_for_cronbach, ci=.99)
        # print(f"{emotion}: {round(float(ca[0]), 3)}, confidence interval: {ca[1]}")
        alphas.append(ca[0])

    return float(np.mean(alphas))


@deprecated("Cronbach alpha is meaningless for a high number of items (i.e. participants)")
def compute_arousal_valence_cronbach_alpha(input_file: str, participants_to_exclude: list[str]):
    # Import the data from a CSV file
    df = pd.read_csv(input_file)

    # Filter patients
    df_filtered = df[~df['respondent'].isin(participants_to_exclude)]

    mean_alpha = compute_cronbach_alpha(df_filtered, "Participant", "Stimulus_Name", "Target")
    print(mean_alpha)

