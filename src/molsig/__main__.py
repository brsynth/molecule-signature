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


AP = argparse.ArgumentParser(description="")
AP_subparsers = AP.add_subparsers(help="Sub-commnands (use with -h for more info)")


def _cmd_enumerate(args):
    logging.info("Start - enumerate")

    # Load Alphabet
    logging.info("Load alphabet")
    Alphabet = load_alphabet(args.input_alphabet_npz, verbose=True)

    # Create ECFP
    logging.info("Create ECFP")
    smiles = args.input_smiles_str
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"SMILES is not validated by rdkit: {smiles}")
        AP.exit(1)
    fpgen = AllChem.GetMorganGenerator(
        radius=Alphabet.radius,
        fpSize=Alphabet.nBits,
        includeChirality=Alphabet.use_stereo,
    )
    morgan = fpgen.GetCountFingerprint(mol).ToList()

    logging.info("Enumerate molecules - start")
    start = time.time()
    Ssig, Smol, Nsig, thresholds_reached, computational_times = enumerate_molecule_from_morgan(
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
    df.to_csv(args.output_data_tsv, sep="\t", index=False)

    logging.info("End - enumerate")
    return 0


P_enumerate = AP_subparsers.add_parser("enumerate", help=_cmd_enumerate.__doc__)
P_enumerate.add_argument("--input-smiles-str", required=True, help="SMILES string")
P_enumerate.add_argument("--input-alphabet-npz", required=True, help="Alphabet file")
P_enumerate.add_argument("--output-data-tsv", required=True, help="Output file")
P_enumerate.set_defaults(func=_cmd_enumerate)


def _cmd_signature(args):
    # Check arguments.
    logging.info("Start - signature")

    # Build signature
    logging.info("Load SMILES into rdkit")
    smiles = args.input_smiles_str
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"SMILES is not validated by rdkit: {smiles}")
        AP.exit(1)

    logging.info("Build signature")
    mol_sig = MoleculeSignature(mol)

    data = dict(SMILES=args.input_smiles_str, signature=mol_sig.to_list())
    df = pd.DataFrame([data])
    logging.info(f"Save signature to: {args.output_data_tsv}")
    df.to_csv(args.output_data_tsv, sep="\t", index=False)

    logging.info("End - signature")
    return 0


P_signature = AP_subparsers.add_parser("signature", help=_cmd_signature.__doc__)
P_signature.add_argument("--input-smiles-str", required=True, help="SMILES string")
P_signature.add_argument("--output-data-tsv", required=True, help="Output file")
P_signature.set_defaults(func=_cmd_signature)


def _cmd_alphabet(args):
    logging.info("Start - alphabet")
    if not os.path.isfile(args.input_smiles_txt):
        logging.error(f"Input file does not exist: {args.input_smiles_txt}")
        AP.exit(1)
    if not os.path.isdir(os.path.dirname(args.output_alphabet_npz)):
        logging.error(f"Directory of the alphabet does not exsit: {args.output_alphabet_npz}")
        AP.exit(1)

    use_stereo = True
    if args.parameter_no_stereo_bool:
        use_stereo = False

    logging.info(f"Radius: {args.parameter_radius_int}")
    logging.info(f"NBits: {args.parameter_nbits_int}")
    logging.info(f"Use stereo: {use_stereo}")

    logging.info("Parse SMILES")
    smiles = []
    with open(args.input_smiles_txt) as fd:
        smiles = fd.read().splitlines()

    if len(smiles) < 1:
        logging.info("Input file is empty")
        AP.exit(1)

    logging.info("Build Alphabet")
    Alphabet = SignatureAlphabet(
        radius=args.parameter_radius_int, nBits=args.parameter_nbits_int, use_stereo=use_stereo
    )
    Alphabet.fill(smiles, verbose=True)

    logging.info("Save Alphabet")
    Alphabet.save(args.output_alphabet_npz)

    logging.info("End - alphabet")
    return 0


P_alphabet = AP_subparsers.add_parser("alphabet", help=_cmd_alphabet.__doc__)
P_alphabet.add_argument(
    "--input-smiles-txt", required=True, help="Files containing SMILES, one entry per line"
)
P_alphabet.add_argument("--parameter-radius-int", type=int, default=2, help="Radius")
P_alphabet.add_argument("--parameter-nbits-int", type=int, default=2048, help="Number of bits")
P_alphabet.add_argument(
    "--parameter-no-stereo-bool",
    action="store_false",
    help="If set, no use stereochemistry information",
)
P_alphabet.add_argument("--output-alphabet-npz", required=True, help="Output Alphabet")
P_alphabet.set_defaults(func=_cmd_alphabet)


# Help.
def print_help():
    """Display this program"s help"""
    print(AP_subparsers.help)
    AP.exit()


# Main.
def parse_args(args=None):
    """Parse the command line"""
    return AP.parse_args(args=args)


def main():
    """Entrypoint to commandline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M",
    )
    args = AP.parse_args()
    # No arguments or subcommands were given.
    if len(args.__dict__) < 1:
        print_help()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
