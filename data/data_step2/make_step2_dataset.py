"""
make_step2_dataset.py

Build the Step-2 (clinical-coding) source artifacts from the Lehman et al.
re-identified MIMIC-III release:

    pesudo_mimic3_processed.csv   # SUBJECT_ID, TEXT (notes joined per patient), code_name
    train_data.pickle             # (X_train, y_train), X = [[note_text, SUBJECT_ID], ...]
    test_data.pickle              # (X_test,  y_test)

The CSV inputs below all come straight from the PhysioNet release, directory
`setup_outputs/`. They are NOT generated here — you already have them after
running the PhysioNet setup. This script only reproduces the patient-level
note aggregation and the train/test split.


Usage:
    python make_step2_dataset.py --setup_dir /path/to/clinical-bert-mimic-notes/1.0.0/setup_outputs
"""

import argparse
import ast
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


def build_pesudo_csv(setup_dir, out_csv):
    """SUBJECT_ID_to_NOTES_1b.csv + MedCAT labels -> pesudo_mimic3_processed.csv."""
    medcat_descript = pd.read_csv(os.path.join(setup_dir, "MedCAT_Descriptions.csv"))      # CODE, DESCRIPTION
    subject_id_to_medcat = pd.read_csv(os.path.join(setup_dir, "SUBJECT_ID_to_MedCAT.csv"))  # SUBJECT_ID, CODE
    notes_1b = pd.read_csv(os.path.join(setup_dir, "SUBJECT_ID_to_NOTES_1b.csv"))            # SUBJECT_ID, TEXT

    # All notes of one patient -> a single space-joined document.
    candi = notes_1b.groupby("SUBJECT_ID")["TEXT"].agg(" ".join).reset_index()

    # Per-patient list of MedCAT concept descriptions (the multilabel target).
    medcat_name_subject = pd.merge(subject_id_to_medcat, medcat_descript, on="CODE", how="inner")
    subject_medcat_names = (
        medcat_name_subject.groupby(["SUBJECT_ID"])["DESCRIPTION"]
        .apply(list)
        .reset_index(name="code_name")
    )

    df_data = pd.merge(candi, subject_medcat_names, on="SUBJECT_ID", how="inner")
    df_data.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}  ({len(df_data)} patients)")
    return df_data


def make_split(df_data, out_dir, test_size=0.1, seed=0):
    """Iterative (multilabel-stratified) train/test split -> train/test_data.pickle.

    X rows are [TEXT, SUBJECT_ID]; this is exactly what step2.py:preprocessing_data
    consumes via data[:][0]. The binarized label matrix is only used to stratify
    the split (downstream code re-derives ICD codes from SUBJECT_ID_to_ICD9.csv).
    """
    X = df_data[["TEXT", "SUBJECT_ID"]]
    Y = df_data["code_name"]

    # code_name may already be a list, or a stringified list when re-loaded from CSV.
    Y_list = [v if isinstance(v, list) else ast.literal_eval(v) for v in Y]
    y = MultiLabelBinarizer().fit_transform(Y_list)
    print(f"[info] label matrix y.shape = {y.shape}")

    X_resize = X.values.reshape((len(y), -1))
    np.random.seed(seed)
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X_resize, y, test_size=test_size
    )
    print(f"[info] train={len(X_train)}  test={len(X_test)}")

    train_path = os.path.join(out_dir, "train_data.pickle")
    test_path = os.path.join(out_dir, "test_data.pickle")
    with open(train_path, "wb") as f:
        pickle.dump((X_train, y_train), f)
    with open(test_path, "wb") as f:
        pickle.dump((X_test, y_test), f)
    print(f"[ok] wrote {train_path} / {test_path}")


def main():
    ap = argparse.ArgumentParser(description="Build Step-2 dataset artifacts")
    ap.add_argument(
        "--setup_dir",
        required=True,
        help="PhysioNet clinical-bert-mimic-notes .../1.0.0/setup_outputs directory",
    )
    ap.add_argument("--out_dir", default=".", help="where to write the artifacts")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "pesudo_mimic3_processed.csv")

    df_data = build_pesudo_csv(args.setup_dir, out_csv)
    make_split(df_data, args.out_dir, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
