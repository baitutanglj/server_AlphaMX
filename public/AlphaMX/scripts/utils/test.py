from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def test_rdkit(smiles, reff_mol, sample_num, MMFF=False, rmsd='RMSD'):
    '''
    generate conformation of molecule, and return the mean rmsd of the refference molecule
    '''
    mol = Chem.MolFromSmiles(smiles)
    rmsd_list = []
    for count in range(sample_num):
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        if MMFF:
            AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        if rmsd == 'RMSD':
            rmsd_list.append(AllChem.GetBestRMS(mol, reff_mol))
        if rmsd == 'rRMSD':
            rmsd_list.append(get_rRMSD(mol, reff_mol))
    return np.array(rmsd_list).mean()

        
def get_rRMSD(ref, probe):
    rRMSD = []
    for i in range(len(AllChem.GetSymmSSSR(ref))):
        neigbors = []
        for atom in list(AllChem.GetSymmSSSR(ref)[i]):
            atoms = ref.GetAtoms()
            neigbors += [ center.GetIdx()   for center in list(atoms[atom].GetNeighbors()) ]
        neigbors = list(set(neigbors))
        whole = list(set(neigbors + list(AllChem.GetSymmSSSR(ref)[i])))

        rRMSD.append(AllChem.GetBestRMS(ref, probe, map = [list(zip(whole,whole))]))
    return np.array(rRMSD).mean()
