from scripts.utils import construct_data
from scripts.utils.file_operation import sdf_write_from_mol
import sys
from rdkit import Chem
import torch
from scripts.model import RGA_direct_with_charge_interac_acfct_encode_whole_no1234
import subprocess
import numpy as np
import os

dir = os.path.split(os.path.realpath(__file__))[0]
input = sys.argv[1]
file_type = sys.argv[2]
output_path = sys.argv[3]
os.system(f"cp -r {input} {output_path}")
os.system(f"cp -r {input} {output_path}_sym")
print(dir)
print(input, file_type, output_path)