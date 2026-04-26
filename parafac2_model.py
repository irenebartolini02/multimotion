import numpy as np
import pandas as pd
from tensorly.decomposition import parafac2

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac2
from scipy.linalg import orthogonal_procrustes



def parafac2_fixed_C(tensor, C_fixed, n_iter_max=500,
                     tol=1e-8, verbose=True):
    """
    PARAFAC2 con matrice C (emozioni) fissa.

    Strategia: usa l'ALS di PARAFAC2 di tensorly come base,
    poi proietta C sul valore fisso ad ogni iterazione.

    Il modello per ogni partecipante i e':
        X_i ≈ A_i @ B @ C.T
    dove:
        A_i : (n_stimuli, rank)  - individuale, ortogonale
        B   : (rank, rank)       - matrice di scaling condivisa
        C   : (n_emotions, rank) - FISSA (Russell)

    Il vincolo PARAFAC2: A_i.T @ A_i = Phi (costante tra partecipanti)
    permette a ogni partecipante di avere pesi diversi per ogni stimolo.

    Args:
        tensor    : (n_participants, n_stimuli, n_emotions)
        C_fixed   : (n_emotions, rank) - valori del circomplesso Russell
        n_iter_max: iterazioni massime
        tol       : soglia convergenza
        verbose   : stampa progresso

    Returns:
        A_list : lista di n_participants array (n_stimuli, rank) -  rappresenta come il partecipante $i$ reagisce a ogni singolo stimolo $j$
        B      : (rank, rank) matrice di scaling condivisa - Serve a legare i fattori latenti estratti dai dati alle dimensioni fisiche di $C$
        C      : C_fixed - valori di Russell
        Phi    : (rank, rank) matrice covarianza condivisa - vincolo fondamentale di PARAFAC2 è che $A_i^T A_i = \Phi$ per tutti i partecipanti. Questo garantisce che, sebbene le reazioni agli stimoli siano soggettive, la struttura sottostante dello spazio emotivo rimanga comparabile tra le persone.
    """
    I, J, K = tensor.shape # I=n_participants, J= n_stimuli, K= n_emotions
    R = C_fixed.shape[1] # rank del modello (dimensione latente=2)
    C = C_fixed.copy()  # non si aggiorna mai

    # Pre-calcola C_pinv e la proiezione una sola volta
    # perché C è fissa
    CtC_inv = np.linalg.pinv(C.T @ C)   # (R, R)
    C_proj = C @ CtC_inv @ C.T          # (K, K) proiettore su spazio C

    # Inizializzazione di B come identità e A_list casuali ma ortogonali
    rng = np.random.default_rng(42)
    B = np.eye(R)

    # Inizializza A_i con decomposizione SVD della slice di ogni partecipante
    # proiettata su C — warm start più stabile della randomizzazione
    A_list = []
    for i in range(I):
        Xi = tensor[i, :, :]       # (J, K) - dati del partecipante i
        M = Xi @ C                 # (J, R) - proiezione su spazio C
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        A_list.append(U @ Vt)      # (J, R) ortogonale

    Phi = np.eye(R)
    prev_loss = np.inf

    for iteration in range(n_iter_max):

        # ── STEP 1: Aggiorna A_i (per ogni partecipante) ────────────────
        # Minimizza ||X_i - A_i @ B @ C.T||_F^2
        # soggetto a A_i.T @ A_i = Phi
        #
        # Soluzione via SVD (Kiers et al. 1999):
        #   SVD( X_i @ C @ CtC_inv @ B.T ) = U S V.T
        #   A_i = U @ V.T
        #
        # Questo garantisce che A_i sia semi-ortogonale e rispetti
        # il vincolo PARAFAC2

        new_A_list = []
        for i in range(I):
            Xi = tensor[i, :, :]              # (J, K)
            # Target per SVD: (J, R)
            target = Xi @ C @ CtC_inv @ B.T
            try:
                U, s, Vt = np.linalg.svd(target, full_matrices=False)
                A_i = U @ Vt                  # (J, R) semi-ortogonale vincolo 
            except np.linalg.LinAlgError:
                A_i = A_list[i]               # fallback

            # Correggi i segni: la colonna r di A_i deve avere
            # correlazione positiva con la colonna r di B @ diag(s_r)
            # — euristica semplice: allinea il segno alla media
            for r in range(R):
                if np.dot(A_i[:, r], A_list[i][:, r]) < 0:
                    A_i[:, r] *= -1

            new_A_list.append(A_i)

        A_list = new_A_list

        # Aggiorna Phi come media delle A_i.T @ A_i
        Phi = np.mean(
            [A_list[i].T @ A_list[i] for i in range(I)],
            axis=0
        )

        # ── STEP 2: Aggiorna B ───────────────────────────────────────────
        # Aggrega tutte le slice:
        #   sum_i A_i.T @ X_i @ C = (sum_i A_i.T @ A_i) @ B @ C.T @ C
        # => B = pinv(sum_i A_i.T @ A_i) @ (sum_i A_i.T @ X_i @ C) @ CtC_inv

        sum_AtA = np.zeros((R, R))
        sum_AtXC = np.zeros((R, R))

        for i in range(I):
            Xi = tensor[i, :, :]
            sum_AtA  += A_list[i].T @ A_list[i]
            sum_AtXC += A_list[i].T @ Xi @ C

        try:
            B = np.linalg.pinv(sum_AtA) @ sum_AtXC @ CtC_inv
        except np.linalg.LinAlgError:
            pass  # mantieni B precedente

        # ── STEP 3: Loss ─────────────────────────────────────────────────
        # Ricostruzione: X_i ≈ A_i @ B @ C.T
        loss = 0.0
        for i in range(I):
            Xi = tensor[i, :, :]
            recon_i = A_list[i] @ B @ C.T   # (J, K)
            loss += np.mean((Xi - recon_i) ** 2)
        loss /= I

        if verbose and iteration % 50 == 0:
            print(f'  [parafac2] iter {iteration:4d} | MSE: {loss:.6f}')

        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f'  [parafac2] Convergenza iter {iteration}'
                      f' | MSE: {loss:.6f}')
            break

        prev_loss = loss

    return A_list, B, C, Phi


def generate_individual_ground_truth_parafac2(
        A_list,
        B,
        participant_map,
        stimulus_map):
    """
    Genera la Ground Truth individuale dal modello PARAFAC2.

    In PARAFAC2 ogni partecipante ha la propria matrice A_i (J x R),
    quindi la GT per (partecipante i, stimolo j) è:

        Valence_ij = sum_r  A_i[j, r] * B[r, 0_col]
        Arousal_ij = sum_r  A_i[j, r] * B[r, 1_col]

    In forma matriciale per il partecipante i:
        GT_i = A_i @ B          shape (J, R)

    dove la colonna 0 è Valence e la colonna 1 è Arousal.

    Differenza rispetto a PARAFAC classico:
        PARAFAC:   GT_ij = A[i,0] * B[j,0]  (stesso peso per tutti gli stimoli)
        PARAFAC2:  GT_ij = sum_r A_i[j,r] * B[r,c]  (peso stimolo-specifico)

    Questo permette di catturare che lo stesso partecipante può avere
    reazioni diverse da quelle attese per stimoli specifici.

    Args:
        A_list       : lista di I matrici (n_stimuli, rank)
        B            : matrice di scaling condivisa (rank, rank)
        C            : matrice emozioni (n_emotions, rank) - per riferimento
        participant_map : dict indice -> nome partecipante
        stimulus_map    : dict indice -> nome stimolo

    Returns:
        DataFrame con [Participant, Stimulus_Name, Valence, Arousal]
    """
    records = []
    n_stimuli = len(stimulus_map)

    for p_idx, A_i in enumerate(A_list):
        p_name = participant_map[p_idx]

        # GT per tutti gli stimoli di questo partecipante in un colpo
        # GT_i shape: (n_stimuli, rank)
        GT_i = A_i @ B

        for s_idx in range(n_stimuli):
            s_name = stimulus_map[s_idx]
            records.append({
                'Participant':   p_name,
                'Stimulus_Name': s_name,
                'Valence':       GT_i[s_idx, 0],
                'Arousal':       GT_i[s_idx, 1],
            })

    return pd.DataFrame(records)


# PARAFAC 2 No Fixed C
    
def parafac2_classic(tensor, rank , n_iter_max=500, tol=1e-8):
    """
    Esegue PARAFAC2 
    
    Args:
        tensor: Array 3D (Soggetti, Stimoli, Emozioni)
        rank: Rango del modello (es. 2 per Valence-Arousal)
        
    Returns:
        A_list: Lista di matrici (Stimoli x 2) soggettive
        B: Matrice di scaling comune
        C: Matrice C 
    """
    
    # 1. Esecuzione PARAFAC2 standard
    # factors[0]: Soggetti (I x R), factors[1]: Stimoli base (R x R), factors[2]: Emozioni (K x R)
    weights, factors, projection_matrices = parafac2(
        tensor, rank=rank, n_iter_max=n_iter_max, init='svd', normalize_factors=False, tol=tol, verbose=False
    )
    
    A_weights = factors[0]
    B_base= factors[1]
    C_computed = factors[2]

    # 2. Rotazione di Procuste: Allineamento allo spazio di Russell
    # Trova la rotazione R tale che C_computed @ R ≈ C_fixed
    #rotation_matrix, _ = orthogonal_procrustes(C_computed, C_fixed)

    # 3. Applicazione della rotazione a tutti i componenti
    # Ruotiamo C per farla coincidere con Russell
    #C_aligned = C_computed @ rotation_matrix
    
    # Ruotiamo la base degli stimoli
    #B_aligned = B_base @ rotation_matrix

    # 4. Ricostruzione delle matrici A_i individuali (Soggetti x Stimoli)
    # A_i = P_i @ B_aligned @ diag(A_weights[i])
    A_list = []
    for i in range(len(projection_matrices)):
        # Ai rappresenta la posizione soggettiva degli stimoli nello spazio ruotato
        Ai = projection_matrices[i] @ np.diag(A_weights[i])
        A_list.append(Ai)

    return A_list, B_base, C_computed 
