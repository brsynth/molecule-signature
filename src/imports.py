###############################################################################
# All imports
# Authors: Jean-loup Faulon jfaulon@gmail.com 
# Jan 2023
###############################################################################

from __future__ import print_function

import copy
import csv
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas
import pickle
import random
import sys
import time

from collections import defaultdict
from IPython.display import SVG
from itertools import chain, combinations
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
IPythonConsole.ipython_useSVG=False
RDLogger.DisableLog('rdApp.*')  

# to use the diophantine package:
# from sympy import Matrix
# from diophantine import solve