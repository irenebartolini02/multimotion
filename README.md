# Ground Truth

## Goal

*Dimensionality reduction of participant responses about their feelings* - In 2 dimensions space that can be interpreted as Valence and Arousal in Russel's Circumplex 

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
        normalization='MinMax'
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

```text
A_i : (n_stimuli, rank)  - individual, orthogonal
B   : (rank, rank)       - shared scaling matrix
C   : (n_emotions, rank) - fixed (Russell)
```

To obtain stimulus-specific weights for each participant, the decomposition must respect the PARAFAC2 constraint: `A_i.T @ A_i = Phi` (constant across participants).

**Functions**:
Model estimation:
```python
def parafac2_fixed_C(tensor, C_fixed, n_iter_max=500,
                     tol=1e-8, verbose=True):
    ...
    return A_list, B, C, Phi
```
Ground truth:
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

- **Pros**: the ground truth is specific for each individual and each stimulus; in general, it is very precise, and each single ground truth is not influenced by the group mean response.
- **Cons**: the number of parameters to estimate is very large, and the model can be too sensitive to noise.




### Tucker 3
- **Description**: Tucker 3 decomposes the tensor `X` as `X ≈ G ×₁ A ×₂ B ×₃ C`, where:

- `A` (`n_participants`, `R_p`) contains the participant factors and is estimated from the data.
- `B` (`n_stimuli`, `R_s`) contains the stimulus factors and is estimated from the data.
- `C` (`n_emotions`, `R_e = 2`) is fixed to the Russell circumplex representation of emotions.
- `G` (`R_p`, `R_s`, `R_e`) is the core tensor that stores interactions among the three modes.

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
- A smaller `R_p` forces participants to be described by fewer latent profiles, so the model behaves more like a compact set of participant prototypes.
- A larger `R_p` allows more participant prototypes and more individual variation, but also makes the model easier to overfit and harder to interpret.
- The same trade-off holds for `R_s` on the stimulus side.

In practice, each row of `A` is the latent coordinate of one participant in that reduced space. With `R_p = 2`, every participant is represented as a mixture of two prototypes; with larger values, the participant space becomes richer and the ground truth can adapt to more subject-specific patterns.



**Functions**:

Rank selection:
```python

def select_tucker3_ranks(tensor: np.ndarray,
                         C_fixed: np.ndarray,
                         rank_p_range: tuple = (2, 4),
                         rank_s_range: tuple = (2, 4),
                         n_iter_max: int = 300,
                         verbose: bool = True) -> pd.DataFrame:


```
Model estimation:
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
Ground truth:
```python
def generate_individual_gt_tucker3(A, B, G,
                                   participant_map, stimulus_map,
                                   normalize=True) -> pd.DataFrame:
    ...
    return pd.DataFrame(records) 
```

- **Issue**: it is more flexible than PARAFAC because it can capture interactions through the core tensor, but it is also more sensitive to rank choice and can become harder to interpret if the model is over-parameterized.

- **Results**: at the end, the selected ranks for `R_p` and `R_s` are `2` and `2`.
- `A`: `(63, 2)`
- `B`: `(38, 2)`
- `C`: `(12, 2)`
- `G`: `(2, 2, 2)`
 
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
    
- **Pros**: fewer parameters to estimate compared with PARAFAC 2, and a more general/stable representation.
- **Cons**: only two prototype types are extracted, so the individual ground truth is influenced by participant clusters. In general, the ground truth results from translations and dilations of group ground truths (outliers are often ignored). In addition, parameters `R_p` and `R_s` must be set manually, so the model is not fully independent.




## PREPROCESSING DEL TENSORE

### Tensore non centrato, con voti normalizzati usando MinMax

MSE= 0.4018 (Alto)

- Quando normalizzi i dati nell'intervallo $[-1, 1]$, l'intervallo totale di variazione (il range) è pari a $2$.
- Se calcoli l'errore quadratico medio (MSE), un valore di 0.4018 significa che, in media, la deviazione standard dell'errore (la radice quadrata dell'MSE) è $\sqrt{0.4018} \approx 0.634$.
- Un errore medio di $0.634$ su un range totale di $2$ significa che il modello sta mancando il valore reale di circa il 31.7% dell'intero range a disposizione.

### Tensore con normalizzazione Simmetrica 
Centriamo i voti del singolo partecipante in modo da togliere il bias del metro di giudizio individuale (chi è più di manica larga e chi meno)
Centriamo i voti sottraendo il valor medio che il partecipante a dato in base all'emozione 

```python
elif normalization == 'Simmetric':
        # Normalizzazione column-wise per partecipante: centra e scala in [-1, +1] in modo indipendente
        for participant in df_for_tensor['Participant'].unique():
            participant_mask = df_for_tensor['Participant'] == participant
            
            for emotion in emotions_to_use:
                # Seleziona la singola colonna per il singolo partecipante
                participant_values = df_for_tensor.loc[participant_mask, emotion]
                
                # Centra sui dati specifici di QUELLA emozione
                mean_score = participant_values.mean()
                centered_values = participant_values - mean_score
                
                # Trova il massimo assoluto specifico per QUELLA emozione
                max_abs = np.abs(centered_values).max()
                
                if max_abs > 0:
                    normalized_values = centered_values / max_abs
                else:
                    normalized_values = centered_values.copy()
                    normalized_values[:] = 0.0
                    
                df_for_tensor.loc[participant_mask, f'{emotion}_normalized'] = normalized_values
```
**Mean Space GT**
<img src='img\mean_space_simmetric_normalization.png'>
Noto che rispetto al grafo ottenuto con MinMax i valori sono più schiacciati, ma centrati negli assi


MSE=  0.087448 (Accettabile)
- (radice dell'errore $\approx 0.2957$), ovvero il 15% del range:
sulla dimensione totale della scala (ampiezza = 2.00):
$$\text{RMSE} = \sqrt{0.087448} \approx 0.2957$$

$$\text{Percentuale di Errore Medio} = \frac{0.295}{2.00} \times 100 \approx 14.7\%$$

#### Errori di ricostruzione 
*MSE reconstruction per partecipant:*
<img  src="img\reconstruction_err_participants.png">
Notiamo dal grafico a barre che due partecipanti spiccano per l'errore di ricostruzione ['LR96S','SxRtt99']
Il il partecipante che è stato meglio ricostruito invece è 'V9D5x'

*MSE reconstruction per stimulus:*
<img  src="img\reconstruction_err_stimulus.png">

*MSE reconstruction per emotion:*
<img  src="img\reconstruction_err_emotions.png">



#### Scatter Plot: Valori Reali vs. Valori Ricostruiti
appiattendo i tensori, originale e ricostruito ho graficato i punti ottenuti che hanno per x il valore originale e per y quello riscostruito.
Idealmente vorremmo x=y quindi che i punti si disponessero sulla diagonale primo-terzo quadrante (evidenziata in rosso)

<img  src="img\scatter_plot_reconstruction.png">
Notiamo che il trend segue le aspettative.


#### Analisi dei Residui per Partecipante (Boxplot + Swarmplot):
Il grafico si basa sul concetto di IQR (Interquartile Range).
Dividiamo i dati in base all'errore di ricostruzione per partecipante in $ guppi:
- Q1(Primo Quartile / 25° percentile): È il valore sotto il quale si trova il 25% dei partecipanti con l'MSE più basso.
- Q2 (Secondo Quartile o Mediana / 50° percentile): È il valore centrale.
- Q3 (Terzo Quartile / 75° percentile): È il valore sotto il quale si trova il 75% dei partecipanti. Solo il 25% ha un errore più alto di questo punto. 

$$IQR = Q3 - Q1$$ 
*Rappresenta la "lunghezza" della scatola e indica dove si concentra il 50% centrale dei dati.*

In seguito selezioniamo gli outliers seguendo il **Metodo di Tukey**:
Il metodo dell'1.5 × IQR (chiamato anche criterio di Tukey) è lo standard scientifico per decretare se un dato è una legittima fluttuazione statistica o un'anomalia (outlier).
Limite Superiore:$$\text{Limite Superiore} = Q3 + (1.5 \times IQR)$$



<img  src="img\outliers.png">

Il metodo rileva due outliers: ['LR96S','SxRtt99'] che il modello fa fatica a ricostruire fedelmente, questo è dovuto al fatto che la loro percezione si discosta da quella grammatica latante del gruppo ($\Phi$)


#### Heatmaps 
*Best reconstructed partecipant*
<img  src="img\best_partecipant_reconstruction_heatmap.png">
Osservando la mapppa originale (a sinistra) e quella ricostruita (a destra), si nota che le bande verticali coincidono quasi perfettamente. Le colonne molto chiare (giallo/verdi) e quelle scure (viola) si trovano nelle stesse posizioni in entrambe le matrici.

Questo partecipante è un "soggetto ideale". Le sue risposte emotive seguono fedelmente la struttura latente globale del gruppo. Il modello riesce a catturare i suoi pattern con estrema facilità, "ripulendo" al contempo i dati dal rumore di fondo (infatti l'immagine ricostruita a destra appare come una versione più fluida e sfumata dell'originale a sinistra).
I colori ricostruiti sono meno intensi (sfumati) questo fenomeno prende il nome di smoothing (smussamento) dovuto alla riduzione di dimensionalità.

*Worse reconstructed partecipant*
<img  src="img\worse_partecipant_reconstruction_heatmap.png">

In questo caso spesso i pattern di alternanza blu/giallo non sono rispettati perchè il modello fatica a ricostruire le risposte del partecipante in quanto si discosta molto dallo stato latente del gruppo.

