import argparse
import logging
import os
import sys
import time

import pandas as pd
from molsig.enumerate_signature import enumerate_molecule_from_morgan
from molsig.Signature import MoleculeSignature
from molsig.SignatureAlphabet import load_alphabet, SignatureAlphabet
from rdkit import Chem
from rdkit.Chem import AllChem


PARSER = argparse.ArgumentParser(description="")
subparsers = PARSER.add_subparsers(help="Sub-commnands (use with -h for more info)")


# Enumerate ---------------------------------------------------------------------------------------

def _cmd_enumerate(args):
    logging.info("Start - enumerate")

    # Load Alphabet
    logging.info("Load alphabet")
    Alphabet = load_alphabet(args.alphabet, verbose=True)

    # Create ECFP
    logging.info("Create ECFP")
    smiles = args.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"SMILES is not validated by rdkit: {smiles}")
        PARSER.exit(1)
    fpgen = AllChem.GetMorganGenerator(
        radius=Alphabet.radius,
        fpSize=Alphabet.nBits,
        includeChirality=Alphabet.use_stereo,
    )
    morgan = fpgen.GetCountFingerprint(mol).ToList()

    logging.info("Enumerate molecules - start")
    start = time.time()
    (Ssig, Smol, Nsig, thresholds_reached, computational_times) = enumerate_molecule_from_morgan(
        morgan, Alphabet
    )
    end = round(time.time() - start, 2)
    logging.info("Enumerate molecules - end")
    logging.info(f"Time computed: {end}")

    sthresholds_reached = ", ".join([str(x) for x in thresholds_reached])
    scomputational_times = ",".join([str(x) for x in computational_times])

    logging.info(f"Thresholds_reached: {sthresholds_reached}")
    logging.info(f"Computational times: {scomputational_times}")

    df = pd.DataFrame(list(Smol), columns=["SMILES"])
    df.to_csv(args.output, sep="\t", index=False)

    logging.info("End - enumerate")
    return 0


parser_enumerate = subparsers.add_parser("enumerate", help=_cmd_enumerate.__doc__)
parser_enumerate.add_argument("--smiles", metavar="STR", required=True, help="Input SMILES string")
parser_enumerate.add_argument("--alphabet", metavar="FILE", required=True, help="Alphabet file (.npz)")  # noqa
parser_enumerate.add_argument("--output", metavar="FILE", required=True, help="Output file (.tsv)")
parser_enumerate.set_defaults(func=_cmd_enumerate)


# Signature ---------------------------------------------------------------------------------------

def _cmd_signature(args):
    # Check arguments.
    logging.info("Start - signature")

    # Build signature
    logging.info("Load SMILES into rdkit")
    smiles = args.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"SMILES is not validated by rdkit: {smiles}")
        PARSER.exit(1)

    logging.info("Build signature")
    mol_sig = MoleculeSignature(mol)

    # Output
    data = dict(SMILES=args.smiles, signature=mol_sig.to_string())
    df = pd.DataFrame([data])
    logging.info(f"Save signature to: {args.output}")
    df.to_csv(args.output, sep="\t", index=False)
    logging.info("End - signature")
    return 0


parser_signature = subparsers.add_parser("signature", help=_cmd_signature.__doc__)
parser_signature.add_argument("--smiles", metavar="STR", required=True, help="Input SMILES string")
parser_signature.add_argument("--output", metavar="FILE", required=True, help="Output file (.tsv)")
parser_signature.set_defaults(func=_cmd_signature)


# Alphabet ----------------------------------------------------------------------------------------

def _cmd_alphabet(args):
    logging.info("Start - alphabet")
    if not os.path.isfile(args.smiles):
        logging.error(f"Input file does not exist: {args.smiles}")
        PARSER.exit(1)
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        logging.error(f"Directory of the alphabet does not exist: {args.output}")
        PARSER.exit(1)

    logging.info(f"Radius: {args.radius}")
    logging.info(f"NBits: {args.nbits}")
    logging.info(f"Use stereo: {args.stereo}")

    logging.info("Parse SMILES")
    smiles_list = []
    with open(args.smiles) as fd:
        smiles_list = fd.read().splitlines()

    if len(smiles_list) < 1:
        logging.info("Input file is empty")
        PARSER.exit(1)

    logging.info("Build Alphabet")
    Alphabet = SignatureAlphabet(radius=args.radius, nBits=args.nbits, use_stereo=args.stereo)
    Alphabet.fill(smiles_list, verbose=True)

    logging.info("Save Alphabet")
    Alphabet.save(args.output)

    logging.info("End - alphabet")
    return 0


parser_alphabet = subparsers.add_parser("alphabet", help=_cmd_alphabet.__doc__)
parser_alphabet.add_argument(
    "--smiles",
    metavar="FILE",
    required=True,
    help="Files containing SMILES, one entry per line (.txt, .smi)",
)
parser_alphabet.add_argument(
    "--radius", metavar="INT", type=int, default=2, help="Radius. Default: %(default)s"
)
parser_alphabet.add_argument(
    "--nbits", metavar="INT", type=int, default=2048, help="Number of bits. Default: %(default)s"
)
parser_alphabet.add_argument(
    "--stereo",
    metavar="BOOL",
    type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
    default=True,
    help="Weither to use stereochemistry in fingerprints. Default: %(default)s",
)
parser_alphabet.add_argument(
    "--output", metavar="FILE", required=True, help="Output Alphabet (.npz)"
)
parser_alphabet.set_defaults(func=_cmd_alphabet)


# Help --------------------------------------------------------------------------------------------

def print_help():
    """Display this program"s help"""
    print(subparsers.help)
    PARSER.exit()


# Main --------------------------------------------------------------------------------------------

def parse_args(args=None):
    """Parse the command line"""
    return PARSER.parse_args(args=args)


def main():
    """Entrypoint to commandline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M",
    )
    args = PARSER.parse_args()
    # No arguments or subcommands were given.
    if len(args.__dict__) < 1:
        print_help()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
