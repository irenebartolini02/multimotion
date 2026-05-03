from pathlib import Path
import pandas as pd
import numpy as np
from parfac_model_gt import scale_and_center_tensor


CORRECT_PARTICIPANTS = ['GFGzP', 'KWgkc', 'EXAMPLE', 'tMGNS', 'urJvc', 'yb3T2', '4FoNM', '5KB3V', 'CoQmx',
 'Cr1sTi', 'F5tXL', 'F8mDn', 'LR96S', 'PPjCX', 'VckNZ', 'hMHEK1', 'wohkw', '6GSd4',
 '8vDRG', 'Fjil72', 'Fyt7d', 'G4Egk', 'Gftw5', 'KoG5ii', 'M4t7k', 'O1pGR',
 'XhsN0o', 'kH4Dd', 'GHft8', 'WYuvqk', 'XPI3pA', 'h2yD5h', '7yqP3', 'G5XzoP',
 'NMy2s', 'Xd3foP9', 'gf5X', 'A5feTy', 'CVgs2', 'Dwf5T', 'ERqc8', 'G4dLl', 'Gr33d',
 'Hx7dO', 'Kgh4P3', 'SI3pa2', 'V9D5x', 'sop2d', '6k6gw', 'D7GcxxFf3', 'FWvvb87',
 'FcXX1', 'K34LOp', 'KLO33', 'Lso2g', 'Ou4tR', 'P67Dftt', 'SxRtt99', 'V11GHjjk',
 'VBr4ssd', 'XCv32m', 'XfdEE1', 'rt8Ye0r']

def select_green_participants(emotions_to_use, normalization='MinMax', preprocessed_data_file="preprocessed_data.csv", correct_participants=CORRECT_PARTICIPANTS):
    """
    Use this function after "convert_multiple_experiments" to load the data in only one csv file:
     1.  filter only the correct participants and the stimuli (without meditation, 
     2.  normalize the emotion ratings to [-1, 1] range 
     3.  impute missing values with the average rating for each stimulus. 
    
    Finally, it saves the preprocessed data to a new csv file and returns the preprocessed dataframe.

    Args:
        emotions_to_use (list): List of emotion column names to process.
        normalization (str): The normalization method to use ('MinMax' or 'Overall').
        preprocessed_data_file (str): Path to save the preprocessed data CSV file.
        correct_participants (list): List of participant IDs to include in the analysis.
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for tensor construction.
    """

    # Load all Excel files from the specified folder and concatenate them into a single dataframe
    single_converted_data_folder = Path('survey_data') / 'SurveyDataJuly24' / 'exp-2025-11-02-with-lopo_v2' / 'single_converted_data'
    all_files = sorted(single_converted_data_folder.glob('*.xlsx'))

    if not all_files:
        raise FileNotFoundError(f'No Excel files found in {single_converted_data_folder}')

    df_list = []
    failed_files = []

    for file in all_files:
        try:
            df_list.append(pd.read_csv(file))
        except Exception as e:
            failed_files.append((file.name, str(e)))
            print(f"Skipping invalid file {file.name}: {e}")

    if not df_list:
        raise RuntimeError(
            f"No valid Excel files could be read from {single_converted_data_folder}. "
            f"Failures: {failed_files}"
        )

    if failed_files:
        print(f"Loaded {len(df_list)} file(s), skipped {len(failed_files)} invalid file(s).")
    df_initial = pd.concat(df_list, ignore_index=True)

    # Check if the required columns are present in the concatenated dataframe
    required_columns = ['respondent', 'stimulus', *emotions_to_use]
    missing_columns = [col for col in required_columns if col not in df_initial.columns]
    if missing_columns:
        raise KeyError(f'Missing columns in concatenated dataframe: {missing_columns}')

    # Filter the dataframe to keep only the required columns
    df_initial = df_initial[required_columns].copy()
    
    # Filter out rows where 'stimulus' contains "meditation"
    df_no_meditations = df_initial[~df_initial['stimulus'].str.contains("meditation")]
    
    # Filter the dataframe to include only the correct (green) participants
    df_filtered = df_no_meditations[df_no_meditations['respondent'].isin(correct_participants)] 

    stimuli = df_filtered['stimulus'].drop_duplicates()
    print(stimuli)

    # Imputation: replace nans with the average value across participants
    for stimulus in stimuli:
        ratings = df_filtered.loc[df_filtered['stimulus'] == stimulus, emotions_to_use]
        ratings = ratings.fillna(ratings.mean())
        df_filtered.loc[df_filtered['stimulus'] == stimulus, emotions_to_use] = ratings
    # rinomino colono respondent in participant e stimulus con Stimulus_Name per coerenza con il resto del codice
    df_for_tensor = df_filtered.rename(columns={'respondent': 'Participant', 'stimulus': 'Stimulus_Name'})

    # add normalized columns for each emotion betweem -1 and 1


    if normalization == 'MinMax':
        # Normalization MinMax for each participant score individually (min is the minimum score applied an Max is the maximum score applied by a certain participant across all stimuli)
        for row in df_for_tensor.itertuples(index=False):
            participant = row.Participant
            participant_mask = df_for_tensor['Participant'] == participant
            for emotion in emotions_to_use:
                norm_col = f'{emotion}_normalized'
                min_score = df_for_tensor.loc[participant_mask, emotion].min()
                max_score = df_for_tensor.loc[participant_mask, emotion].max()
                if max_score > min_score:  # avoid division by zero
                    df_for_tensor.loc[participant_mask, norm_col] = (df_for_tensor.loc[participant_mask, emotion] - min_score) / (max_score - min_score) * 2 - 1
                else:
                    df_for_tensor.loc[participant_mask, norm_col] = 0  # if all scores are the same, set normalized to 0

    
    else: 
        # Normalization Overall: each score is normalized using the global min and max (0 and 9)
        min=0
        max=9
        for emotion in emotions_to_use:
            norm_col = f'{emotion}_normalized'
            df_for_tensor[norm_col] = (df_for_tensor[emotion] - min) / (max - min) * 2 - 1

    df_for_tensor.to_csv(preprocessed_data_file, index=False)

    return df_for_tensor



def prepare_tensor_from_dataframe(df: pd.DataFrame, emotions_to_use: list, scale_participants: bool = False, center_participants: bool = False, center_stimuli: bool = False) -> tuple:
    """
    Prepare a 3D tensor from survey responses: (participants × stimuli × emotions).
    """
    participants = sorted(df['Participant'].unique())
    stimuli = sorted(df['Stimulus_Name'].unique())

    participant_map = {i: p for i, p in enumerate(participants)}
    stimulus_map = {i: s for i, s in enumerate(stimuli)}

    n_participants = len(participants)
    n_stimuli = len(stimuli)
    n_emotions = len(emotions_to_use)

    tensor = np.zeros((n_participants, n_stimuli, n_emotions))

    for idx, row in df.iterrows():
        p_idx = participants.index(row['Participant'])
        s_idx = stimuli.index(row['Stimulus_Name'])
        tensor[p_idx, s_idx, :] = row[emotions_to_use].values
    
    if scale_participants or center_participants or center_stimuli:
        tensor, scale_factors = scale_and_center_tensor(tensor, scale_participants=scale_participants, center_participants=center_participants, center_stimuli=center_stimuli)

    return tensor, participant_map, stimulus_map



def scale_and_center_tensor(tensor, 
                             scale_participants=True,
                             center_participants=True, 
                             center_stimuli=True  ):
    """
    Preprocessing corretto secondo Bro (1997) sezione 5.
    
    Ordine: scaling PRIMA della centratura.
    
    Scaling per partecipante (slab scaling sul modo 0):
        Normalizza ogni slice tensor[i,:,:] alla stessa norma.
        Questo equalizza il "metro di giudizio" tra partecipanti
        che usano la scala in modo molto diverso (compressa vs espansa).
    
    Centratura sequenziale (un modo alla volta):
        - Modo 0 (partecipanti): rimuove il bias di risposta individuale
        - Modo 1 (stimoli): rimuove il livello medio per stimolo
    """
    tensor_out = tensor.copy().astype(float)
    scale_factors = np.ones(tensor.shape[0])  # salva i fattori per interpretazione
    
    # STEP 1: Scaling per partecipante (slab scaling modo 0)
    # Scala ogni slice tensor[i,:,:] in modo che abbia norma unitaria
    if scale_participants:
        for i in range(tensor_out.shape[0]):
            slab = tensor_out[i, :, :]  # shape (n_stimuli, n_emotions)
            norm = np.sqrt(np.sum(slab ** 2))
            if norm > 1e-10:
                tensor_out[i, :, :] = slab / norm
                scale_factors[i] = norm
            # se norm ~ 0 (partecipante ha risposto tutto uguale) lascia a zero
    
    # STEP 2: Centratura sequenziale (dopo scaling)
    if center_participants:
        tensor_out -= np.mean(tensor_out, axis=0, keepdims=True)
    
    if center_stimuli:
        tensor_out -= np.mean(tensor_out, axis=1, keepdims=True)
    
    return tensor_out, scale_factors

 