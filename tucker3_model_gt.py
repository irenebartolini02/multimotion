"""
tucker3_model.py  —  Tucker3 con matrice C fissa (circomplesso di Russell)
==========================================================================

ALGORITMO: HOOI (Higher-Order Orthogonal Iteration) con C fissa
----------------------------------------------------------------
Tucker3 decompone X (I×J×K) come:

    X ≈ G ×₁ A ×₂ B ×₃ C

    A ∈ R^{I×R_p}      fattori partecipanti  (ortogonale, stimato)
    B ∈ R^{J×R_s}      fattori stimoli       (ortogonale, stimato)
    C ∈ R^{K×R_e}      fattori emozioni      (FISSA = Russell, R_e=2)
    G ∈ R^{R_p×R_s×R_e}  core tensor        (stimato, non vincolato)

Con C fissa, HOOI alterna:
  1. A ← R_p vettori sinistri di unfold(X ×₂ B.T ×₃ C.T, 0)
  2. B ← R_s vettori sinistri di unfold(X ×₁ A.T ×₃ C.T, 1)
  3. G  = einsum('ijk,ip,jq,kr->pqr', tensor, A, B, C)

A e B rimangono ortogonali per costruzione (SVD). La scala va in G.

PERCHÉ HOOI E NON ALS DIRETTO
-------------------------------
L'ALS diretto (A = X_(0) @ Z_A @ pinv(Z_A.T @ Z_A)) diverge quando
A e B non sono ortogonali perché G cresce senza limiti mentre A e B
si normalizzano. HOOI mantiene l'ortonormalità e garantisce convergenza.

INTERPRETAZIONE PSICOLOGICA
----------------------------
A[i,p]:    Partecipante i → profilo latente p.
           R_p=2: due stili percettivi estratti dai dati.

B[j,q]:    Stimolo j → cluster emotivo q.
           R_s=3: tre pattern emotivi degli stimoli.

G[p,q,r]:  Core tensor.
           "I partecipanti del profilo p, reagendo agli stimoli del
            cluster q, quanto attivano la dimensione r (Val/Aro)?"
           G[:,:,0] = interazioni per Valence
           G[:,:,1] = interazioni per Arousal

GT individuale: Valence_ij = a_i.T @ G[:,:,0] @ b_j
                Arousal_ij = a_i.T @ G[:,:,1] @ b_j
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd as scipy_svd


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Matricizzazione del tensore sul modo specificato."""
    return np.reshape(
        np.moveaxis(tensor, mode, 0),
        (tensor.shape[mode], -1)
    )


def mode_n_product(tensor: np.ndarray,
                   matrix: np.ndarray,
                   mode: int) -> np.ndarray:
    """
    Prodotto modo-n: tensor ×_mode matrix.
    matrix shape (m, d_mode) → sostituisce d_mode con m nel tensore.
    """
    moved  = np.moveaxis(tensor, mode, 0)
    result = np.tensordot(matrix, moved, axes=[[1], [0]])
    return np.moveaxis(result, 0, mode)


def reconstruct_tucker3(G, A, B, C):
    """X_recon[i,j,k] = sum_{p,q,r} G[p,q,r]*A[i,p]*B[j,q]*C[k,r]"""
    return np.einsum('pqr,ip,jq,kr->ijk', G, A, B, C)


def tucker3_fixed_C(tensor: np.ndarray,
                    C_fixed: np.ndarray,
                    rank_participants: int,
                    rank_stimuli: int,
                    n_iter_max: int = 500,
                    tol: float = 1e-9,
                    verbose: bool = True) -> tuple:
    """
    Tucker3 con C fissa tramite HOOI.

    Args:
        tensor           : (I, J, K)
        C_fixed          : (K, R_e) — fissa, tipicamente R_e=2
        rank_participants: R_p
        rank_stimuli     : R_s
        n_iter_max       : iterazioni massime
        tol              : soglia variazione relativa MSE
        verbose          : stampa progresso

    Returns:
        A, B, C, G, loss_history
    """
    I, J, K = tensor.shape
    R_e = C_fixed.shape[1]
    R_p = rank_participants
    R_s = rank_stimuli
    C   = C_fixed.copy()

    # Inizializzazione HOSVD: vettori singolari degli unfolding
    X0 = unfold(tensor, 0)   # (I, J*K)
    X1 = unfold(tensor, 1)   # (J, I*K)

    U0, _, _ = scipy_svd(X0, full_matrices=False)
    A = U0[:, :R_p].copy()

    U1, _, _ = scipy_svd(X1, full_matrices=False)
    B = U1[:, :R_s].copy()

    loss_history = []
    prev_loss    = np.inf

    for iteration in range(n_iter_max):

        # ── Aggiorna A ───────────────────────────────────────────────────
        # Y_A = X ×₂ B.T ×₃ C.T   shape (I, R_s, R_e)
        # Proietta X sullo spazio (B,C), lascia libero il modo I.
        # R_p vettori singolari sinistri di unfold(Y_A,0) = nuova A.
        Y_A = mode_n_product(tensor, B.T, mode=1)   # (I, R_s, K)
        Y_A = mode_n_product(Y_A,    C.T, mode=2)   # (I, R_s, R_e)
        U, _, _ = scipy_svd(unfold(Y_A, 0), full_matrices=False)
        A = U[:, :R_p]

        # ── Aggiorna B ───────────────────────────────────────────────────
        # Y_B = X ×₁ A.T ×₃ C.T   shape (R_p, J, R_e)
        # Proietta X sullo spazio (A aggiornato, C), lascia libero modo J.
        Y_B = mode_n_product(tensor, A.T, mode=0)   # (R_p, J, K)
        Y_B = mode_n_product(Y_B,    C.T, mode=2)   # (R_p, J, R_e)
        U, _, _ = scipy_svd(unfold(Y_B, 1), full_matrices=False)
        B = U[:, :R_s]

        # ── Core tensor G ─────────────────────────────────────────────────
        # Soluzione analitica esatta con A, B ortogonali:
        # G[p,q,r] = sum_{i,j,k} X[i,j,k] * A[i,p] * B[j,q] * C[k,r]
        G = np.einsum('ijk,ip,jq,kr->pqr', tensor, A, B, C)

        # ── Loss ──────────────────────────────────────────────────────────
        X_r = reconstruct_tucker3(G, A, B, C)
        mse = float(np.mean((tensor - X_r) ** 2))
        loss_history.append(mse)

        if verbose and iteration % 50 == 0:
            print(f'  [tucker3] iter {iteration:4d} | MSE: {mse:.6f} '
                  f'| rank=({R_p},{R_s},{R_e})')

        rel = abs(prev_loss - mse) / (abs(prev_loss) + 1e-12)
        if rel < tol:
            if verbose:
                print(f'  [tucker3] Convergenza iter {iteration}'
                      f' | MSE: {mse:.6f}')
            break
        prev_loss = mse

    return A, B, C, G, loss_history


def select_tucker3_ranks(tensor: np.ndarray,
                         C_fixed: np.ndarray,
                         rank_p_range: tuple = (2, 4),
                         rank_s_range: tuple = (2, 4),
                         n_iter_max: int = 300,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Griglia di ricerca rank ottimale via AIC proxy.

    AIC proxy = n_obs * log(MSE) + 2 * n_params
    n_params = I*R_p + J*R_s + R_p*R_s*R_e  (C fissa non conta)

    Returns: DataFrame [rank_p, rank_s, mse, n_params, aic_proxy]
             ordinato per aic_proxy crescente.
    """
    I, J, K = tensor.shape
    R_e     = C_fixed.shape[1]
    results = []

    for R_p in range(rank_p_range[0], rank_p_range[1] + 1):
        for R_s in range(rank_s_range[0], rank_s_range[1] + 1):
            _, _, _, _, hist = tucker3_fixed_C(
                tensor, C_fixed, R_p, R_s,
                n_iter_max=n_iter_max, verbose=False
            )
            mse      = hist[-1]
            n_params = I*R_p + J*R_s + R_p*R_s*R_e
            n_obs    = I*J*K
            aic      = n_obs * np.log(max(mse, 1e-12)) + 2*n_params

            results.append({
                'rank_p': R_p, 'rank_s': R_s,
                'mse': round(mse, 6),
                'n_params': n_params,
                'aic_proxy': round(aic, 2)
            })
            if verbose:
                print(f'  rank_p={R_p} rank_s={R_s} | '
                      f'MSE={mse:.6f} | AIC={aic:.1f}')

    df = (pd.DataFrame(results)
          .sort_values('aic_proxy')
          .reset_index(drop=True))
    if verbose:
        print('\nTop 5:\n', df.head())
    return df


def generate_individual_gt_tucker3(A, B, C, G,
                                   participant_map, stimulus_map,
                                   normalize=True) -> pd.DataFrame:
    """
    Ground Truth individuale da Tucker3.

    Formula vettorizzata:
        VA_val = A @ G[:,:,0] @ B.T    shape (I, J)  — Valence
        VA_aro = A @ G[:,:,1] @ B.T    shape (I, J)  — Arousal

    Cattura soggettività stimolo-specifica: due partecipanti con
    profili a_i diversi ottengono GT diverse per lo stesso stimolo
    tramite la forma bilineare a_i.T @ G_VA @ b_j.

    Args:
        normalize: scala in [-1,1] mantenendo lo zero al centro.

    Returns:
        DataFrame [Participant, Stimulus_Name, Valence, Arousal]
    """
    VA_val = A @ G[:, :, 0] @ B.T   # (I, J)
    VA_aro = A @ G[:, :, 1] @ B.T   # (I, J)

    records = [
        {
            'Participant':   participant_map[p_idx],
            'Stimulus_Name': stimulus_map[s_idx],
            'Valence':       float(VA_val[p_idx, s_idx]),
            'Arousal':       float(VA_aro[p_idx, s_idx]),
        }
        for p_idx in range(A.shape[0])
        for s_idx in range(B.shape[0])
    ]

    gt_df = pd.DataFrame(records)

    if normalize:
        for col in ['Valence', 'Arousal']:
            m = np.abs(gt_df[col]).max()
            if m > 1e-12:
                gt_df[col] /= m

    return gt_df


def compare_models(tensor: np.ndarray, results: dict) -> pd.DataFrame:
    """
    Confronta MSE di diversi modelli sullo stesso tensore.

    results: dict {nome: {'A':..., 'B':..., 'C':..., ['G':...,'A_list':...]}}
    """
    rows = []
    for name, res in results.items():
        if 'G' in res:
            X_r = reconstruct_tucker3(res['G'], res['A'], res['B'], res['C'])
        elif 'A_list' in res:
            X_r = np.stack([
                res['A_list'][i] @ res['B'] @ res['C'].T
                for i in range(len(res['A_list']))
            ])
        else:
            X_r = np.einsum('ir,jr,kr->ijk', res['A'], res['B'], res['C'])

        rows.append({'model': name,
                     'mse': round(float(np.mean((tensor - X_r)**2)), 6)})

    return pd.DataFrame(rows).sort_values('mse').reset_index(drop=True)
