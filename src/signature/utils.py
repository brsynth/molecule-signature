# =================================================================================================
# Utilities to sanitize and filter molecular structures, as well as read and write data.
#
# Authors:
#   - Jean-loup Faulon <jfaulon@gmail.com>
#   - Thomas Duigou <thomas.duigou@inrae.fr>
#   - Philippe Meyer <philippe.meyer@inrae.fr>
# =================================================================================================


import csv

import numpy as np
import pandas as pd
from rdkit import Chem


# =================================================================================================
# Sanitize and inspect molecular structures
# =================================================================================================


def mol_from_smiles(
    smiles: str,
    clear_stereo: bool = False,
    clear_aam: bool = True,
    clear_isotope: bool = True,
    clear_hs: bool = True,
) -> Chem.Mol:
    """Sanitize a molecule

    Parameters
    ----------
    smiles : str
        Smiles string to sanitize.
    max_mw : int, optional
        Maximum molecular weight, by default 500.
    clear_stereo : bool, optional
        Clear stereochemistry information, by default True.
    clear_aam : bool, optional
        Clear atom atom mapping, by default True.
    clear_isotope : bool, optional
        Clear isotope information, by default True.
    clear_hs : bool, optional
        Clear hydrogen atoms, by default True.

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
        if "*" in smiles:  # Reject generic molecules
            return

        if clear_stereo:  # Wild but effective
            smiles = smiles.replace("@", "").replace("/", "").replace("\\", "")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        if clear_aam:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

        if clear_isotope:
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

        if clear_stereo or clear_isotope or clear_hs:
            # Removing stereochemistry and isotope information might leave
            # the molecule with explicit hydrogens that does not carry any
            # useful information. We remove them.
            mol = Chem.RemoveHs(mol)

        return mol

    except Exception as err:
        raise err


def mol_filter(
    mol: Chem.Mol, max_mw: int = 500, exclude_radical: bool = True, exclude_dative: bool = True
) -> Chem.Mol:
    """Filter a molecule

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to filter.
    max_mw : int, optional
        Maximum molecular weight, by default 500.
    exclude_radical : bool, optional
        Exclude molecules having radicals, by default True.
    exclude_dative : bool, optional
        Exclude molecules having dative bonds, by default True.

    Returns
    -------
    Chem.Mol
        The molecule if it passes the filter, None otherwise.
    """
    if Chem.Descriptors.MolWt(mol) > max_mw:
        return
    if exclude_radical:
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return
    if exclude_dative:
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() in [17, 18, 19, 20]:
                return
    return mol


# =================================================================================================
# Read and write txt file where Data is a list
# =================================================================================================


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


# =================================================================================================
# read write csv file with panda where Data is a np array
# =================================================================================================


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
