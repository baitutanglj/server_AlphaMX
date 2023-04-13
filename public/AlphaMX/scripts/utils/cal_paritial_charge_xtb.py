from xtb.interface import Calculator, Param
from xtb.libxtb import VERBOSITY_MINIMAL
import pandas as pd
from collections import defaultdict
from rdkit.Chem import AllChem
import numpy as np

def atom_position(rdkit_mol):
    pos = rdkit_mol.GetConformer().GetPositions()
    return pos

def atomic_number(rdkit_mol):
    atomic_numbers = [atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()]
    return np.array(atomic_numbers)
    
def get_charge(rdkit_mol, optimize_geometry=True):
    mol = rdkit_mol
    if optimize_geometry:
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        
    calc = Calculator(Param.GFN2xTB, atomic_number(mol), atom_position(mol))
    calc.set_verbosity(VERBOSITY_MINIMAL)
    res = calc.singlepoint()
    charges = res.get_charges()
    
    for i,atom in enumerate(rdkit_mol.GetAtoms()):
        atom.SetProp('charges', str(charges[i]))
    return charges,rdkit_mol

def get_ACFCT_table(rdkit_mol, heavy_atom_only=True, optimize_geometry=True):
    
    charges, mol = get_charge(rdkit_mol, optimize_geometry)
    
    if heavy_atom_only:
        mol = AllChem.RemoveHs(mol)
        
    ACFCT_table = pd.DataFrame(columns=['idx','atom','degree','charges','neighbor_charges'])
    charges = defaultdict(list)
    for i,atom in enumerate(mol.GetAtoms()):
        for neighbor in atom.GetNeighbors():
            charges[i].append(neighbor.GetProp('charges'))
    degrees = defaultdict(list)
    for i,atom in enumerate(mol.GetAtoms()):
        degrees[i].append(atom.GetDegree())
    for i,atom in enumerate(mol.GetAtoms()):
        ACFCT_table.loc[i,'idx'] = i    
        ACFCT_table.loc[i,'atom'] = atom.GetSymbol()
        ACFCT_table.loc[i,'degree'] = degrees[i]
        ACFCT_table.loc[i,'charges'] = atom.GetProp('charges')
        ACFCT_table.loc[i,'neighbor_charges'] = np.sort(np.array(charges[i]))
    return ACFCT_table

def get_ACFCT(charge,rdkit_mol):
    mol = rdkit_mol
    degrees = [[atom.GetDegree() for atom in mol.GetAtoms()]] 
    for i,atom in enumerate(mol.GetAtoms()):
        degrees.append(atom.GetDegree())