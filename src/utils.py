"""
Shared utilities for the T2-relaxometry project.

Provides NIfTI / TE loaders for the two cohorts (six adult controls and the
EPICURE preterm-born adolescent cohort), tissue helpers, demographics,
and a high-level cohort loader that discovers, merges, and QA-filters
subjects into a ready-to-analyse list.
"""

from pathlib import Path
import re, glob

import numpy as np
import nibabel as nib
import pandas as pd


# ---------------------------------------------------------------------------
# Segmentation class ordering (matches the *-seg.nii.gz files)
# ---------------------------------------------------------------------------
SEG_NAMES = {0: 'Unassigned', 1: 'CSF', 2: 'GM', 3: 'WM',
             4: 'Deep GM', 5: 'Brainstem'}

SEG_COLOURS = {
    1: '#ffc800',   # CSF
    2: '#16766b',   # GM
    3: '#945ad6',   # WM
    4: '#09413a',   # Deep GM
    5: '#c44536',   # Brainstem
}


# ---------------------------------------------------------------------------
# Basic loaders
# ---------------------------------------------------------------------------
def load_nifti(path):
    """Load a NIfTI file and return its data as a float32 numpy array."""
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def load_tes(path):
    """Load echo times from a text file, robust to any whitespace layout."""
    with open(path) as f:
        text = f.read().replace(',', ' ')
    te = np.fromstring(text, sep=' ', dtype=float)
    if te.size == 0:
        raise ValueError(f'No numeric echo times could be read from {path}')
    return te


# ---------------------------------------------------------------------------
# Single-subject loaders
# ---------------------------------------------------------------------------
def load_adult_case(case_num, data_root='../data'):
    """
    Load one adult-control subject.

    Returns dict with keys: id, cohort, TE, reg, mask, seg, par, par_lobe.
    """
    case_dir = Path(data_root) / 'data_adult_control' / f'case_{case_num}'
    stem = f'case{case_num:02d}'

    reg = load_nifti(case_dir / f'{stem}-qt2_reg.nii.gz')
    n_echoes = reg.shape[-1]

    TE_raw = load_tes(case_dir / f'{stem}-TEs.txt')

    # Make TE schedule match the number of volumes
    if len(TE_raw) == n_echoes:
        TE = TE_raw
    elif len(np.unique(TE_raw)) == n_echoes:
        _, idx = np.unique(TE_raw, return_index=True)
        TE = TE_raw[np.sort(idx)]
    else:
        raise ValueError(
            f'Case {case_num}: TE/data mismatch. '
            f'{len(TE_raw)} TEs, {n_echoes} volumes.')

    mask = load_nifti(case_dir / f'{stem}-mask.nii.gz').astype(bool)
    seg = load_nifti(case_dir / f'{stem}-seg.nii.gz')
    par = load_nifti(case_dir / f'{stem}-par.nii.gz')
    par_lobe = load_nifti(case_dir / f'{stem}-par_lobe.nii.gz')

    return dict(id=case_num, cohort='adult', TE=TE, reg=reg, mask=mask,
                seg=seg, par=par, par_lobe=par_lobe)


def load_adolescent(epicure_id, data_root='../data'):
    """
    Load one EPICURE preterm-born adolescent subject.

    All adolescent subjects share a single TE schedule (data_root/TEs.txt).
    """
    data_root = Path(data_root)
    adol_dir = data_root / 'data_adolescent'
    stem = f'Epicure{epicure_id}'

    TE = load_tes(data_root / 'TEs.txt')
    reg = load_nifti(adol_dir / f'{stem}-qt2_reg.nii.gz')
    mask = load_nifti(adol_dir / f'{stem}-mask1.nii.gz').astype(bool)
    seg = load_nifti(adol_dir / f'{stem}-qt2_seg1.nii.gz')
    par = load_nifti(adol_dir / f'{stem}-qt2_par1.nii.gz')
    par_lobe = load_nifti(adol_dir / f'{stem}-qt2_par2.nii.gz')

    return dict(id=epicure_id, cohort='adolescent', TE=TE, reg=reg,
                mask=mask, seg=seg, par=par, par_lobe=par_lobe)


# ---------------------------------------------------------------------------
# Tissue helpers
# ---------------------------------------------------------------------------
def hard_seg(ds):
    """Argmax tissue label map with indices matching SEG_NAMES."""
    return np.argmax(ds['seg'], axis=-1)


def mean_decay_curve(ds, seg_idx):
    """Probability-weighted mean T2 decay curve for one tissue class."""
    weights = np.where(ds['mask'], ds['seg'][..., seg_idx], 0.0)
    total = float(weights.sum())
    if total < 1e-6:
        return np.zeros(ds['reg'].shape[-1])
    return (ds['reg'].astype(np.float64)
            * weights[..., None]).sum(axis=(0, 1, 2)) / total


def region_signals(ds, seg_idx):
    """Individual voxel decay curves for a tissue class. (n_voxels, n_echoes)."""
    roi = (hard_seg(ds) == seg_idx) & ds['mask']
    return ds['reg'][roi, :].astype(np.float64)


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
def load_demographics(data_root='../data'):
    """
    Load the EPICURE subject-level cohort table from
    Epicure_psychological.xlsx.  Returns a clean DataFrame with EP flag,
    gestational age, sex, birth weight, neuropsych scores, and imaging flags.
    """
    xlsx_path = Path(data_root) / 'Epicure_psychological.xlsx'
    raw = pd.read_excel(xlsx_path, sheet_name=0)

    df = pd.DataFrame()
    df['id'] = raw['ID'].astype(str)
    df['EP'] = pd.to_numeric(raw['epi19y'], errors='coerce') == 1

    for col in ['ga', 'gawks', 'male', 'bw', 'bwsds', 'eth_white',
                'y19ses_4',
                'y19ps_fsiq_comp', 'y19ps_vci_comp', 'y19ps_pri_comp',
                'y19ps_vmi_std', 'y19ps_vis_std', 'y19ps_mot_std']:
        if col in raw.columns:
            df[col] = pd.to_numeric(raw[col], errors='coerce')

    for src, dst in [('y19_T2Relaxometry', 'has_T2'),
                     ('y19_T1wVolume', 'has_T1'),
                     ('y19_MultishellDWI', 'has_DWI')]:
        df[dst] = (pd.to_numeric(raw[src], errors='coerce') == 1) \
                  if src in raw.columns else False

    return df


# ---------------------------------------------------------------------------
# Cohort loader (wraps discovery + merge + QA exclusion)
# ---------------------------------------------------------------------------
def load_preterm_cohort(data_root='../data', exclude_file=None):
    """
    Discover, load, and QA-filter the adolescent analysis cohort.

    1. Scans data_adolescent/ for NIfTI files.
    2. Inner-joins with the demographics table for EP/FT flag + gawks.
    3. Checks TE / volume alignment and drops mismatches.
    4. Optionally applies QA exclusions from a CSV file.

    Returns
    -------
    preterm : list of dict
        Each entry is a subject dict with extra keys 'EP' and 'gawks'.
    cohort : DataFrame
        Subject-level table with id, EP, gawks, male, y19ps_fsiq_comp.
    """
    adol_dir = Path(data_root) / 'data_adolescent'
    if not adol_dir.is_dir():
        raise FileNotFoundError(f'{adol_dir} not found.')

    # Discover subject IDs on disk
    reg_files = sorted(glob.glob(str(adol_dir / 'Epicure*-qt2_reg.nii.gz')))
    subject_ids = sorted({
        re.search(r'Epicure(\d+)', f).group(1)
        for f in reg_files if re.search(r'Epicure(\d+)', f)
    })

    # Merge with demographics
    demo = load_demographics(data_root)
    cohort = pd.DataFrame({'id': subject_ids}).merge(
        demo[['id', 'EP', 'gawks', 'male', 'y19ps_fsiq_comp']],
        on='id', how='inner')

    # Apply QA exclusions if provided
    if exclude_file is not None:
        exc_path = Path(exclude_file)
        if exc_path.exists():
            exc_ids = set(pd.read_csv(exc_path)['id'].astype(str).str.strip())
            cohort = cohort[~cohort['id'].isin(exc_ids)]

    # Load and validate each subject
    preterm = []
    for _, row in cohort.iterrows():
        try:
            ds = load_adolescent(int(row['id']), data_root=data_root)
        except Exception:
            continue

        # TE / volume alignment check
        if len(ds['TE']) != ds['reg'].shape[-1]:
            continue

        ds['EP'] = bool(row['EP'])
        ds['gawks'] = float(row['gawks']) if pd.notna(row['gawks']) else float('nan')
        preterm.append(ds)

    # Filter cohort table to match loaded subjects
    loaded_ids = {str(ds['id']) for ds in preterm}
    cohort = cohort[cohort['id'].isin(loaded_ids)].reset_index(drop=True)

    return preterm, cohort
