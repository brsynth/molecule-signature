########################################################################################################################
# This file provide utilities
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Jan 2023
########################################################################################################################


import csv

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


########################################################################################################################
# Sanitize and inspect molecular structures
########################################################################################################################


def mol_from_smiles(smiles: str, keep_stereo: bool = False) -> Chem.Mol:
    """Sanitize a molecule

    Parameters
    ----------
    smiles : str
        Smiles string to sanitize.
    max_mw : int, optional
        Maximum molecular weight, by default 500.
    keep_stereo : bool, optional
        Whether to keep stereochemistry, by default False.

    Returns
    -------
    Chem.Mol
        Sanitized molecule. If smiles is not valid, returns None.
    """
    try:
        if smiles == "nan" or smiles == "" or pd.isna(smiles):
            return
        if "." in smiles:  # Reject molecules
            return
        if "*" in smiles:   # Reject generic molecules
            return
        if not keep_stereo:
            # Wild but effective stereo removal
            smiles = smiles.replace("@", "").replace("/", "").replace("\\", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        return mol

    except Exception as err:
        raise err


def mol_filter(mol: Chem.Mol, max_mw: int = 500, no_radical: bool = False) -> Chem.Mol:
    """Filter a molecule

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to filter.
    max_mw : int, optional
        Maximum molecular weight, by default 500.
    keep_radical : bool, optional
        Whether to keep radicals, by default False.

    Returns
    -------
    Chem.Mol
        The molecule if it passes the filter, None otherwise.
    """
    if Chem.Descriptors.MolWt(mol) > max_mw:
        return
    if no_radical:
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return
    return mol


########################################################################################################################
# Read and write txt file where Data is a list
########################################################################################################################


def read_txt(filename):
    """
    Read lines from a text file and return them as a list.

    Parameters
    ----------
    filename : str
        The name of the text file to read.

    Returns
    -------
    list of str
        A list containing the lines read from the text file.
    """

    with open(filename, "r") as fp:
        Lines = fp.readlines()
        Data, i = {}, 0
        for line in Lines:
            Data[i] = line.strip()
            i += 1

        return list(Data.values())


def write_txt(filename, Data):
    """
    Write data to a text file, with each item in the list written as a separate line.

    Parameters
    ----------
    filename : str
        The name of the text file to write to.
    Data : list
        A list containing the data to write to the file.
    """

    with open(filename, "w") as fp:
        for i in range(len(Data)):
            fp.write("%s\n" % Data[i])


########################################################################################################################
# read write csv file with panda where Data is a np array
########################################################################################################################


def read_csv(filename):
    """
    Read data from a CSV file using pandas.

    Parameters
    ----------
    filename : str
        The name of the CSV file to read from.

    Returns
    -------
    HEADER : list
        List of column names extracted from the CSV file.
    DATA: numpy.ndarray
        Array containing the data read from the CSV file.
    """

    filename += ".csv"
    dataframe = pd.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])

    return HEADER, DATA


def write_csv(filename, H, D):
    """
    Write data to a CSV file.

    Parameters
    ----------
    filename : str
        The name of the CSV file to write to.
    H : list or None
        Header row to be written to the CSV file.
    D : numpy.ndarray
        Data to be written to the CSV file.
    """

    # H = Header, D = Data
    filename += ".csv"
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the header
        if H is not None:
            writer.writerow(H)
        # write the data
        for i in range(D.shape[0]):
            writer.writerow(D[i])


def read_tsv(filename):
    """
    Read data from a TSV file using pandas.

    Parameters
    ----------
    filename : str
        The name of the TSV file to read from.

    Returns
    -------
    HEADER : list
        The header row read from the TSV file.
    DATA : numpy.ndarray
        The data read from the TSV file.
    """

    filename += ".tsv"
    dataframe = pd.read_csv(filename, header=0, sep="\t")
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])

    return HEADER, DATA


def write_tsv(filename, H, D):
    """
    Write data to a TSV file using the provided header and data.

    Parameters
    ----------
    filename : str
        The name of the TSV file to write to.
    H : list
        The header to write to the TSV file.
    D : numpy.ndarray
        The data to write to the TSV file.
    """

    filename += ".tsv"
    with open(filename, "w", encoding="UTF8") as f:
        writer = csv.writer(f, delimiter="\t")
        # write the header
        if H is not None:
            writer.writerow(H)
        # write the data
        for i in range(D.shape[0]):
            writer.writerow(D[i])


########################################################################################################################
# other utilities
########################################################################################################################


def vector_to_dic(V):
    """
    Convert a 1D array into a dictionary where each element is a key and its corresponding index is the value.

    Parameters
    ----------
    V : numpy.ndarray
        The 1D array to be converted into a dictionary.

    Returns
    -------
    D : dict
        A dictionary where each element of the array is a key, and its corresponding index is the value.
    """

    D = {}
    for i in range(len(V)):
        D[V[i]] = i

    return D


def dic_to_vector(D):
    """
    Convert a dictionary into a set containing its keys.

    Parameters
    ----------
    D : dict
        The dictionary to be converted into a set.

    Returns
    -------
    set
        A set containing the keys of the input dictionary.
    """

    return {key for key in D.keys()}


def print_matrix(A):
    """
    Print the non-zero elements of a matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix.
    """

    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j]:
                print(f"A ({i}, {j})")
