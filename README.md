# Ground Truth

## Goal

*Dimensionality reduction of participant responses about their feelings* - In 2 dimension space that can be interpeted as Valence and Arousal in Russel's Circumplex 

<img src='img\Russells-Circumplex.webp' >


## Data Overview

Participants are exposed to 38 different stimuli. For each stimulus, they assign a score from 0 to 9 to each of 12 emotions.

- **Participants**: before the experiment, participants are screened by a psychologist based on their ability to understand and report emotions.
    - Selected participants are labeled as **green participants**.
    - Participants who do not pass the screening are labeled as **red participants**.

Number of green participants: **63**

- **Stimuli**: the experiment includes 38 visual stimuli (videos).
    - Each stimulus ID is made of letters and numbers.
    - The first letter represents expected **Arousal** (`H` = high, `L` = low).
    - The second letter represents expected **Valence** (`P` = positive, `N` = negative).
    - This notation is used to map feelings into a 2D space according to **Russell's Circumplex Model**.

```python
stimulus = [
        "HN_1", "HN_4", "HN_5", "HN_6", "HN_7", "HN_8",
        "HP_2", "HP_4", "HP_5", "HP_6", "HP_8",
        "LN_1", "LN_2", "LN_3", "LN_4", "LN_5", "LN_6", "LN_8",
        "LP_1", "LP_2", "LP_3", "LP_4", "LP_5", "LP_6", "LP_7", "LP_8",
        "HP_1_H", "HP_1_L", "HP_3_H", "HP_3_L", "HP_7_L", "HP_7_H",
        "HN_2_H", "HN_2_L", "HN_3_H", "HN_3_L", "LN_7_N", "LN_7_P"
]
```

- **Emotions**: for each stimulus, participants assign a score (0-9) to the following 12 emotions.

```python
emotions = [
        "Amused*", "Angry*", "Bored*", "Calm*", "Content*", "Disgust*",
        "Excited*", "Fearful*", "Happy*", "Negative*", "Positive*", "Sad*"
]
```

## Dataset Pipeline

1. Raw data are collected as `.txt` files in:
     - `survey_data\SurveyDataJuly24\raw_data`

2. Raw files are parsed into multiple `.xlsx` files using:

```python
convert_multiple_experiments(
        raw_data_paths,
        processed_xlsx_data_path,
        emotions_to_use,
        version=survey_version,
        export_single_csvs=True,
)
```

Output folder:
- `survey_data\SurveyDataJuly24\exp-2025-11-02-with-lopo_v2\single_converted_data`

3. Files are filtered to keep only green participants and valid stimuli. Then columns named `emotion*_normalized` are added, with values normalized in the range `[-1, 1]`, using:

```python
def select_green_participants(
        emotions_to_use,
        preprocessed_data_file="preprocessed_data.csv",
        correct_participants=CORRECT_PARTICIPANTS,
):
        ...
```

4. The filtered and normalized data are saved as a `.csv` file, which is the final database.

## Dataset Structure

| Participant | Stimulus_Name | Amused* | Angry* | Bored* | Calm* | Content* | Disgust* | Excited* | Fearful* | Happy* | Negative* | Positive* | Sad* | Amused*_normalized | Angry*_normalized | Bored*_normalized | Calm*_normalized | Content*_normalized | Disgust*_normalized | Excited*_normalized | Fearful*_normalized | Happy*_normalized | Negative*_normalized | Positive*_normalized | Sad*_normalized |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Old Methods:

### INDSCAL

- **Description**: is a multi-way model designed to find a common hidden space (like the Valence/Arousal axes) shared by all participants, while accounting for individual variations through simple importance weights.

- **Issue**: participant ground truth is computed as a weighted copy of the group mean result.
    - If a stimulus is considered positive by the majority, it becomes positive for all participants, even when an individual perceived it as negative.
    - This does not account for subjectivity and personal experience.

### PARAFAC

- **Description**: decomposition of the tensor `(participants, stimuli, emotions)` into 3 matrices `A`, `B`, `C` using the ALS algorithm.

<img src="img\parafac_decomposition.png">

```text
A = (62, 2)  -> participant weights for valence/arousal (mode-0 synthesis)
B = (38, 2)  -> valence and arousal for each stimulus (mode-1 synthesis)
C = (12, 2)  -> valence and arousal for each emotion (mode-2 synthesis)
```

- **Issue**: same limitation as INDSCAL. Ground truth is computed as:

```text
Participant_i_stimulus_j_valence = A[i][0] * B[j][0]
Participant_i_stimulus_j_arousal = A[i][1] * B[j][1]
```

## New Methods

These methods are based on tensor decomposition by mode in order to reduce the dimensionality of the data in a space that can be interpreted as Russell's Circumplex space. In this version, matrix `C`, which synthesizes the emotion mode, is kept fixed.

<img src="img\fixed_C.png">

### PARAFAC 2
- **Description**: decompose the tensor into `A_list`, `B`, and `C`. Compared with PARAFAC, `A_list` is a list of stimulus-specific matrices `A_i`, which preserve the individual weights of participant `i` for each stimulus. In addition, there is no global stimulus matrix; `B` is the scaling matrix used to compute the ground truth (`A_i @ B`).

        A_i : (n_stimuli, rank)  - individual, orthogonal
        B   : (rank, rank)       - shared scaling matrix
        C   : (n_emotions, rank) - fixed (Russell)

To obtain stimulus-specific weights for each participant, the decomposition must respect the PARAFAC2 constraint: `A_i.T @ A_i = Phi` (constant across participants).

**Functions**:
Model estimation 
```python
def parafac2_fixed_C(tensor, C_fixed, n_iter_max=500,
                     tol=1e-8, verbose=True):
    ...
    return A_list, B, C, Phi
```
Ground Truth
```python
def generate_individual_ground_truth_parafac2(
        A_list,
        B,
        participant_map,
        stimulus_map):
    ...
    return pd.DataFrame(records) 
```

**Issue**: it can be unstable, too specific, and expensive in terms of parameters to estimate. If overparameterized, it may lead to overfitting and convergence problems.

**Results**:

<img src="img\mean_space_PARAFAC2.png">




### Tucker 3
- **Description**: Tucker 3 decomposes the tensor `X` as `X ≈ G ×₁ A ×₂ B ×₃ C`, where:
- `A` (n_partecipants, R_p)contains the participant factors and is estimated from the data;
- `B` (n_stimuli, R_s) contains the stimulus factors and is estimated from the data;
- `C` (n_emotions, R_e=2) is fixed to the Russell circumplex representation of the emotions;
- `G` (R_p, R_s, R_e) is the core tensor that stores the interactions among the three modes.

<img src="img\tucker_decomposition.png">

In the implementation, Tucker 3 is estimated with HOOI. `A` and `B` are kept orthogonal by construction, while the scale is absorbed by `G`. The individual ground truth is then computed with the bilinear form:

```text
Valence_ij = a_i^T @ G[:,:,0] @ b_j
Arousal_ij = a_i^T @ G[:,:,1] @ b_j
```

The code also includes a rank selection step that searches over candidate ranks for participants and stimuli using an AIC-like proxy:

```text
AIC_proxy = n_obs * log(MSE) + 2 * n_params
n_params = I * R_p + J * R_s + R_p * R_s * R_e
```

where `R_p` is the participant rank, `R_s` is the stimulus rank, and `R_e = 2` because `C` is fixed in the Russell space.

The selected rank changes how expressive the model is:
- a smaller `R_p` forces participants to be described by fewer latent profiles, so the model behaves more like a compact set of participant prototypes;
- a larger `R_p` allows more participant prototypes and more individual variation, but also makes the model easier to overfit and harder to interpret;
- the same trade-off holds for `R_s` on the stimulus side.

In practice, each row of `A` is the latent coordinate of one participant in that reduced space. With `R_p = 2`, every participant is represented as a mixture of two prototypes; with larger values, the participant space becomes richer and the ground truth can adapt to more subject-specific patterns.


- **Functions**:

Rank selection 
```python

def select_tucker3_ranks(tensor: np.ndarray,
                         C_fixed: np.ndarray,
                         rank_p_range: tuple = (2, 4),
                         rank_s_range: tuple = (2, 4),
                         n_iter_max: int = 300,
                         verbose: bool = True) -> pd.DataFrame:


```
Model estimation 
```python
def tucker3_fixed_C(tensor: np.ndarray,
                    C_fixed: np.ndarray,
                    rank_participants: int,
                    rank_stimuli: int,
                    n_iter_max: int = 500,
                    tol: float = 1e-9,
                    verbose: bool = True) -> tuple:
    ...
    return A, B, C, G, loss_history
```
Ground Truth
```python
def generate_individual_gt_tucker3(A, B, G,
                                   participant_map, stimulus_map,
                                   normalize=True) -> pd.DataFrame:
    ...
    return pd.DataFrame(records) 
```

- **Issue**: it is more flexible than PARAFAC because it can capture interactions through the core tensor, but it is also more sensitive to rank choice and can become harder to interpret if the model is over-parameterized.

- **Results**: At the end, the selected rank for R_p and R_s are 2 and 2. 
- `A` (63, 2)
- `B` (38, 2)
- `C` (12, 2) 
- `G` (2, 2, 2)
 
log output:
``` text

Top 5:
    rank_p  rank_s       mse  n_params  aic_proxy
0       2       2  9.093352       210   63838.31
1       3       2  9.164012       277   64194.68
2       4       2  9.173266       344   64357.68
3       5       2  9.173266       411   64491.68
4       2       3  9.287573       252   64529.44
  [tucker3] iter    0 | MSE: 9.099774 | rank=(2,2,2)
  [tucker3] Convergenza iter 14 | MSE: 9.093352
```

<img src="img\mean_space_TUCKER3.png">
    
