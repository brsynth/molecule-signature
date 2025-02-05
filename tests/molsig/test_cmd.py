import os
import subprocess
import tempfile


class TestCmd:
    def test_signature(self):

        output_data = tempfile.mkstemp(
            suffix=".tsv"
        )  # no usage of TemporaryFile to prevent "Permission denied" windows error

        args = ["molsig", "signature"]
        args += ["--input-smiles-str", "CCO"]
        args += ["--output-data-tsv", output_data[1]]
        ret = subprocess.run(args)
        assert ret.returncode < 1

        with open(output_data[1]) as fd:
            lines = fd.read().splitlines()
        assert lines == [
            "SMILES\tsignature",
            "CCO\t['80-1410 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4:1]-[O;H1;h1;D1;X2]', '807-222 ## [C;H3;h3;D1;X4]-[C;H2;h2;D2;X4]-[O;H1;h1;D1;X2:1]', '1057-294 ## [O;H1;h1;D1;X2]-[C;H2;h2;D2;X4]-[C;H3;h3;D1;X4:1]']",
        ]
        # clean up
        os.remove(output_data[1])

    def test_alphabet_enumerate(self):
        input_fd, input_path = tempfile.mkstemp(suffix=".txt")
        alphabet_fd, alphabet_path = tempfile.mkstemp(suffix=".npz")
        output_fd, output_path = tempfile.mkstemp(suffix=".tsv")

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

        with os.fdopen(input_fd, "w") as fd:
            for smi in smiles:
                value = f"{smi}\n"
                fd.write(value)

        args = ["molsig", "alphabet"]
        args += ["--input-smiles-txt", input_path]
        args += ["--output-alphabet-npz", alphabet_path]
        ret = subprocess.run(args, capture_output=True, text=True)
        assert ret.returncode < 1, f"stdout: {ret.stdout}\nstderr: {ret.stderr}"

        # Enumerate
        args = ["molsig", "enumerate"]
        args += ["--input-smiles-str", "CCO"]
        args += ["--input-alphabet-npz", alphabet_path]
        args += ["--output-data-tsv", output_path]
        ret = subprocess.run(args)
        assert ret.returncode < 1

        with open(output_path) as fd:
            lines = fd.read().splitlines()
        assert lines == ["SMILES"]

        # Clean up
        for filename in [input_path, alphabet_path, output_path]:
            os.remove(filename)
