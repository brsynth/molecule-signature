import os
import subprocess
import sys
import tempfile

import pytest


@pytest.fixture(scope="session")
def use_shell():
    return True if sys.platform.startswith("win") else False


class TestCmd:

    def test_signature(self, use_shell):

        temp_path = ""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            args = ["molsig", "signature"]
            args += ["--smiles", "CCO"]
            args += ["--output", temp_path]
            ret = subprocess.run(args, capture_output=True, shell=use_shell)
            assert ret.returncode < 1, f"stdout: {ret.stdout}\nstderr: {ret.stderr}"

        with open(temp_path) as fd:
            lines = fd.read().splitlines()
        assert lines[0] == "SMILES\tsignature"
        assert lines[1].startswith("CCO\t")
        # clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_alphabet_enumerate(self, use_shell):
        # Build alphabet
        smiles = [
            "O=C(O)[C@@H]1O[C@H](Oc2cccc3c(=O)oc(/C=C/CCO)cc23)[C@H](O)[C@H](O)[C@@H]1O",
            "Cc1ccc(Cn2nc(C)cc2C(=O)Nc2ccc(Cl)cc2)cc1",
            "COc1ccc([C@@H]2NC(=O)c3ccccc3O2)c(OC)c1OC",
            "COc1ccc([C@@H]2CCC[C@H](CCc3ccc(O)cc3)O2)cc1",
            "C[C@]12CC[C@H]3[C@@](O)(CCC4=CC(=O)CC[C@@]43C)[C@@H]1CC[C@@H]2C(=O)CO",
            "C=C1/C(=C\\C=C2/CCC[C@]3(C)[C@@H]([C@@H](C)[C@@H](C#CC(O)(CC)CC)OCC)CC[C@@H]23)C[C@@H](O)C[C@@H]1O",
            "CC[C@H](C)[C@H](N)C(=O)N[C@H](C(=O)N[C@@H](CO)C(=O)O)[C@@H](C)O",
            "CSCCN=C=S",
            "O=C1C[C@H](O)[C@](O)([C@@H]2C(=O)[C@]3(Cl)[C@H](Cl)C[C@]2(Cl)C3(Cl)Cl)[C@@H]1O",
            "CN(C)[C@@H]1C(=O)[C@H](C(N)=O)[C@H](O)[C@]2(O)C(=O)[C@@H]3C(=O)c4c(O)ccc(Cl)c4[C@](C)(O)[C@@H]3C[C@H]12",
            "O[C@@]12[C@@H]3C[C@@](O)(C(Cl)=C3Cl)[C@H]1[C@@H]1C[C@]2(O)[C@@H]2O[C@@H]12",
            "CC(=O)NCC/C(=C\\N)C(=O)OC(=O)C(=O)C(=O)[O-]",
            "CSCC[C@H](NC(=O)[C@@H](N)CO)C(=O)N1CCC[C@@H]1C(=O)O",
            "C[C@@H](O)[C@H](NC(=O)CNC(=O)[C@@H](N)CC(=O)O)C(=O)O",
            "CCc1ccccc1OC[C@@H](O)CN[C@H]1CCCc2ccccc21",
            "CO[C@@H]1CN(C)C(=O)c2ccc(NC(C)=O)cc2OC[C@@H](C)N(Cc2cc(F)ccc2F)C[C@@H]1C",
            "NC(=O)CCCCCC[C@H](O)/C=C/CCCCCCCCO",
            "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCSC)C(=O)O",
            "CCC[C@H](O)C(=O)NCC(=O)[O-]",
            "O=C[C@@H]1[C@H](O)[C@](O)(C(=O)[O-])[C@@H]2[C@]3(Cl)C(Cl)=C(Cl)[C@](Cl)([C@@H]3Cl)[C@]12O",
        ]

        path_smiles, path_alphabet, path_enumerate = "", "", ""

        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as temp_smiles, tempfile.NamedTemporaryFile(
            delete=False
        ) as temp_alphabet, tempfile.NamedTemporaryFile(
            delete=False
        ) as temp_enumerate:
            path_smiles = temp_smiles.name
            path_alphabet = temp_alphabet.name
            path_enumerate = temp_enumerate.name

            for smi in smiles:
                value = f"{smi}\n"
                temp_smiles.write(value)
            temp_smiles.flush()

            args = ["molsig", "alphabet"]
            args += ["--smiles", path_smiles]
            args += ["--output", path_alphabet]
            ret = subprocess.run(args, capture_output=True, shell=use_shell)
            assert ret.returncode < 1, f"stdout: {ret.stdout}\nstderr: {ret.stderr}"

            # Enumerate
            args = ["molsig", "enumerate"]
            args += ["--smiles", "CCO"]
            args += ["--alphabet", path_alphabet]
            args += ["--output", path_enumerate]
            ret = subprocess.run(args, capture_output=True, shell=use_shell)
            assert ret.returncode < 1, f"stdout: {ret.stdout}\nstderr: {ret.stderr}"

            with open(path_enumerate) as fd:
                lines = fd.read().splitlines()
            assert lines == ["SMILES"]

        # Clean up
        for filename in [path_smiles, path_alphabet, path_enumerate]:
            if os.path.exists(filename):
                os.remove(filename)
