import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem
import networkx as nx
import numpy as np
from torch_geometric import utils
from collections import defaultdict
import random


def get_atomic_number(rdkit_mol):          
    # transform atoms into its atomic number
    mol = rdkit_mol
    atomic_number_dict = {'C':6,'O':8,'N':7,'P':15,'S':16,'F':9,'Cl':17,'Br':35}
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atomic_numbers = [atomic_number_dict[atom] for atom in atoms]
    return torch.tensor(atomic_numbers).to(torch.float32).reshape(-1,1)

def get_atomic_ele_arrangement(rdkit_mol):
    # get atomic ele_arrangement outside the nucleus
    mol = rdkit_mol 
    atomic_ele_arrangement_dict = {'C':(2,4,0,0),
                                  'N':(2,5,0,0),
                                  'O':(2,6,0,0),
                                  'F':(2,7,0,0),
                                  'P':(2,8,5,0),
                                  'S':(2,8,6,0),
                                  'Cl':(2,8,7,0),
                                  'Br':(2,8,18,7)}

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atomic_ele_arrangement_list = [atomic_ele_arrangement_dict[atom] for atom in atoms]
    return torch.tensor(atomic_ele_arrangement_list).to(dtype=torch.float32).reshape(-1,4)

def get_gasteiger_charge(rdkit_mol):
    mol = rdkit_mol
    gasteiger_charge = []
    AllChem.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        gasteiger_charge.append(atom.GetProp('_GasteigerCharge'))
    gasteiger_charge = [float(charge) for charge in gasteiger_charge]
    return torch.tensor(gasteiger_charge, dtype=torch.float32).reshape(-1,1)

def get_resp_charge(chg_file,num_atoms):
    with open(chg_file,'r') as chg:
        lines = chg.readlines()
    lines = [line.split()[4] for line in lines]
    resp_charges = np.array(lines[0:num_atoms],dtype=np.float32).reshape(-1,1)
    return torch.tensor(resp_charges, dtype=torch.float32).reshape(-1,1)

def get_atomic_val_electron(rdkit_mol):
    mol = rdkit_mol
    # get atomic valence electron
    atomic_val_electron_dict = {'C':2.5, 'N':3.0, 'O':3.5, 'F':4.0, 'P':2.1, 'S':2.5, 'Cl':3.0, 'Br':2.8}
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atomic_val_electron_list = [atomic_val_electron_dict[atom] for atom in atoms]
    return torch.tensor(atomic_val_electron_list).to(dtype=torch.float32).reshape(-1,1)

def get_atomic_degree(bonds_index, num_atoms):
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(list(range(num_atoms)))
    mol_graph.add_edges_from(bonds_index.T.tolist())
    atomic_degree_list = []
    for n in range(num_atoms):
        atomic_degree_list.append(mol_graph.degree(n))
    return torch.tensor(atomic_degree_list).to(dtype=torch.float32).reshape(-1,1)

def judge_atom_if_aromatic(rdkit_mol):
    mol = rdkit_mol
    # judge the atom is atomatic or not
    atom_if_aromatic = []      
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            atom_if_aromatic.append(1)
        else:
            atom_if_aromatic.append(0)
    return torch.tensor(atom_if_aromatic).to(dtype=torch.float32).reshape(-1,1)

def judge_atom_if_in_ring(rdkit_mol,num_atoms):
    mol = rdkit_mol
    ring_info = mol.GetRingInfo()
    atoms_in_ring = ()
    for ring in ring_info.AtomRings():
        atoms_in_ring += ring
    atoms_in_ring = set(atoms_in_ring)
    atom_if_in_ring = []
    for n in range(num_atoms):
        if n in atoms_in_ring:
            atom_if_in_ring.append(1)
        else:
            atom_if_in_ring.append(0)
    atom_if_in_ring = torch.tensor(atom_if_in_ring).to(dtype=torch.float32).reshape(-1,1)
    return atom_if_in_ring

def judge_atom_if_in_ring_ringnum_smallestringsize(rdkit_mol,num_atoms):
    mol = rdkit_mol
    ring_info = mol.GetRingInfo()
    atoms_in_ring = ()
    for ring in ring_info.AtomRings():
        atoms_in_ring += ring
    atoms_in_ring = set(atoms_in_ring)
    atom_if_in_ring = []
    atom_in_ringnum = []
    atom_in_smallestringsize = []
    for n in range(num_atoms):
        if n in atoms_in_ring:
            atom_if_in_ring.append(1)
            atom_in_ringnum.append(ring_info.NumAtomRings(n))
            all_ring_size = []
            for ring in ring_info.AtomRings():
                if n in ring:
                    all_ring_size.append(len(ring))
            atom_in_smallestringsize.append(min(all_ring_size))
        else:
            atom_if_in_ring.append(0)
            atom_in_ringnum.append(ring_info.NumAtomRings(n))
            atom_in_smallestringsize.append(0)
    atom_if_in_ring = torch.tensor(atom_if_in_ring).to(dtype=torch.float32).reshape(-1,1)
    atom_in_ringnum = torch.tensor(atom_in_ringnum).to(dtype=torch.float32).reshape(-1,1)
    atom_in_smallestringsize = torch.tensor(atom_in_smallestringsize).to(dtype=torch.float32).reshape(-1,1)        
    return atom_if_in_ring, atom_in_ringnum, atom_in_smallestringsize
    
def judge_atom_if_in_aromatic_ring(rdkit_mol, num_atoms):
    mol = rdkit_mol
    # Get all the aromatic atoms
    atoms_aromatic_list = []
    for i,atom in enumerate(mol.GetAtoms()):
        if atom.GetIsAromatic():
            atoms_aromatic_list.append(i)
    
    # Get all atoms in aromatic ring
    ring_info = mol.GetRingInfo()
    
    atoms_in_aromatic_ring = ()
    for ring in ring_info.AtomRings():
        for n,atom in enumerate(ring):
            if atom not in atoms_aromatic_list:
                break
            if n == len(ring)-1:
                atoms_in_aromatic_ring += ring 
    atoms_in_aromatic_ring = set(atoms_in_aromatic_ring)
    
    atom_if_in_aromatic_ring = []
    for n in range(num_atoms):
        if n in atoms_in_aromatic_ring:
            atom_if_in_aromatic_ring.append(1)
        else:
            atom_if_in_aromatic_ring.append(0)
    atom_if_in_aromatic_ring = torch.tensor(atom_if_in_aromatic_ring).to(dtype=torch.float32).reshape(-1,1)
    return atom_if_in_aromatic_ring

def judge_atom_if_conjugate(rdkit_mol, num_atoms):
    mol = rdkit_mol    
    conjugate_atom = []
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            begin_atom, end_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            conjugate_atom.append(begin_atom)
            conjugate_atom.append(end_atom)
    conjugate_atom = list(set(conjugate_atom))
    atom_if_conjugate = []
    for n in range(num_atoms):
        if n in conjugate_atom:
            atom_if_conjugate.append(1)
        else:
            atom_if_conjugate.append(0)
    atom_if_conjugate = torch.tensor(atom_if_conjugate).to(dtype=torch.float32).reshape(-1,1)
    return atom_if_conjugate

def judge_atom_if_in_conjugate_ring(rdkit_mol, num_atoms):
    mol = rdkit_mol    
    # Get all the conjugate atoms
    conjugate_atom = []
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            begin_atom, end_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            conjugate_atom.append(begin_atom)
            conjugate_atom.append(end_atom)
    conjugate_atom = list(set(conjugate_atom))
        # Get all atoms in aromatic ring
    ring_info = mol.GetRingInfo()
    
    # Get all atoms in aromatic ring
    ring_info = mol.GetRingInfo()
    
    atoms_in_conjugate_ring = ()
    for ring in ring_info.AtomRings():
        for n,atom in enumerate(ring):
            if atom not in conjugate_atom:
                break
            if n == len(ring)-1:
                atoms_in_conjugate_ring += ring 
    atoms_in_conjugate_ring = set(atoms_in_conjugate_ring)
    
    atom_if_in_conjugate_ring = []
    for n in range(num_atoms):
        if n in atoms_in_conjugate_ring:
            atom_if_in_conjugate_ring.append(1)
        else:
            atom_if_in_conjugate_ring.append(0)
    atom_if_in_conjugate_ring = torch.tensor(atom_if_in_conjugate_ring).to(dtype=torch.float32).reshape(-1,1)
    return atom_if_in_conjugate_ring

def get_bond_index_type(rdkit_mol): 
    mol = rdkit_mol
    bond_type_dict = {'SINGLE':1,'DOUBLE':2,'TRIPLE':3,'AROMATIC':4}
    bond_order_dict = {'SINGLE':(1,0,0,0),'DOUBLE':(0,1,0,0),'TRIPLE':(0,0,1,0),'AROMATIC':(0,0,0,1)}
    bond_index_list = []
    bond_type_list = []
    bond_order_list = []
    bond_if_conjugate = []
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_type_dict[str(bond.GetBondType())]
        bond_order = bond_order_dict[str(bond.GetBondType())]
        bond_index_list.append((begin_atom, end_atom))
        bond_type_list.append(bond_type)
        bond_order_list.append(bond_order)
        if bond.GetIsConjugated():
            bond_if_conjugate.append(1)
        else:
            bond_if_conjugate.append(0)

    bond_index_list = torch.tensor(bond_index_list).T.reshape(2,-1).to(torch.long)
    bond_type_list = torch.tensor(bond_type_list).reshape(-1,1).to(torch.float32)
    bond_order_list = torch.tensor(bond_order_list).reshape(-1,4).to(torch.float32)
    bond_if_conjugate = torch.tensor(bond_if_conjugate).reshape(-1,1).to(torch.float32)
    bond_attr = torch.cat([bond_order_list,bond_if_conjugate],dim=1)
    return bond_index_list, bond_type_list, bond_attr
            
def get_angle_index(bonds_index, num_atoms):
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(list(range(num_atoms)))
    mol_graph.add_edges_from(bonds_index.T.tolist())
    angle_index_list = []
    for begin_atom in range(num_atoms):
        for key in list(nx.single_source_shortest_path_length(mol_graph, begin_atom).keys()):
            if nx.single_source_shortest_path_length(mol_graph, begin_atom)[key] == 2 and key > begin_atom:
                angle_index_list.append((begin_atom,key))
    return torch.tensor(angle_index_list).T.reshape(2,-1).to(torch.long)

def get_dihedral_index(bonds_index, num_atoms):
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(list(range(num_atoms)))
    mol_graph.add_edges_from(bonds_index.T.tolist())
    dihedral_index_list = []
    for begin_atom in range(num_atoms):
        for key in list(nx.single_source_shortest_path_length(mol_graph, begin_atom).keys()):
            if nx.single_source_shortest_path_length(mol_graph, begin_atom)[key] == 3 and key > begin_atom:
                dihedral_index_list.append((begin_atom,key))
    return torch.tensor(dihedral_index_list).T.reshape(2,-1).to(torch.long)

def get_hop4_index(bonds_index, num_atoms):
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(list(range(num_atoms)))
    mol_graph.add_edges_from(bonds_index.T.tolist())
    hop4_index_list = []
    for begin_atom in range(num_atoms):
        for key in list(nx.single_source_shortest_path_length(mol_graph, begin_atom).keys()):
            if nx.single_source_shortest_path_length(mol_graph, begin_atom)[key] == 4 and key > begin_atom:
                hop4_index_list.append((begin_atom,key))
    return torch.tensor(hop4_index_list).T.reshape(2,-1).to(torch.long)

def get_all_path_pair(data, pair_index):
    all_path_pair = []
    if len(pair_index[0]) != 0:
        nx_g = utils.to_networkx(data)
        for n in range(len(pair_index[0])):
            for pair in nx.all_shortest_paths(nx_g, pair_index[0][n].item(), pair_index[1][n].item()):
                all_path_pair.append(pair)
    return torch.tensor(all_path_pair).to(torch.long)

def partition_atom_pair(atom_pair_index, atom_if_in_ring):
    ring_ring_pair = []
    acyclic_acyclic_pair = []
    acyclic_ring_pair = []
    for index in atom_pair_index.T:
        if atom_if_in_ring[index[0].item()] == 1 and atom_if_in_ring[index[1].item()] == 1:
            ring_ring_pair.append((index[0].item(),index[1].item()))
        elif atom_if_in_ring[index[0].item()] == 0 and atom_if_in_ring[index[1].item()] == 0:
            acyclic_acyclic_pair.append((index[0].item(),index[1].item()))
        else:
            acyclic_ring_pair.append((index[0].item(),index[1].item()))
    return torch.tensor(ring_ring_pair).T.reshape(2,-1).to(torch.long),torch.tensor(acyclic_acyclic_pair).T.reshape(2,-1).to(torch.long),torch.tensor(acyclic_ring_pair).T.reshape(2,-1).to(torch.long)

def get_atom_distance_matrix(rdkit_mol):
    atom_distance_matrix = AllChem.Get3DDistanceMatrix(rdkit_mol)
    return torch.tensor(atom_distance_matrix).to(torch.float32)

def get_atom_pair_distance(pair_index, atom_distance_matrix, num_atoms):
    # distance of bonds
    index = pair_index[0]*num_atoms+pair_index[1]
    pair_distance_list = atom_distance_matrix[index]
    return pair_distance_list

def get_all_path_pair_dis(all_path_pair, atom_distance_matrix, num_atoms):
    if len(all_path_pair) != 0:
        index = all_path_pair[:,0]*num_atoms+all_path_pair[:,-1]
        all_path_pair_distance_list = atom_distance_matrix[index]
    else:
        all_path_pair_distance_list = torch.tensor([]).reshape(-1,1)
    return all_path_pair_distance_list

def split_data_to_1hop_fragment(data):
    nx_G = utils.to_networkx(data)
    frag_list = {}
    atom_bond_dict = {}
    feature_list = defaultdict(list)
    acfc_keys = defaultdict(list)
    n = 0
    for atom in  range(len(data.x)):
        frag_list[atom] = list(nx_G.neighbors(atom))
        feature_list[n].append(data.x[atom])
        acfc_keys[n].append( ('').join( [str(int(f)) for f in data.x[atom].tolist()] ))
        atom_bond_dict[atom] = data.edge_attr[torch.nonzero(data.edge_index[0]==atom).squeeze().tolist()].reshape(-1,5)
        for i,neighbor in enumerate(list(nx_G.neighbors(atom))):
            feature_list[n].append(torch.cat([data.x[neighbor],atom_bond_dict[atom][i]],dim=-1))
            acfc_keys[n].append(('').join( [str(int(f)) for f in torch.cat([data.x[neighbor],atom_bond_dict[atom][i]],dim=-1).tolist()] ))
        n += 1
    return feature_list,acfc_keys,frag_list



def cal_acfc(data,center_neighbor_whole_mean_resp):
    feature_list,acfc_keys,frag_list = split_data_to_1hop_fragment(data)
    center_neighbor_key = []
    for n in range(len(data.x)):
        center_neighbor_key.append((acfc_keys[n][0], ('').join(sorted(acfc_keys[n][1:]))))
    acfc_list = defaultdict(list)
    for n in range(len(data.x)):
        acfc_charge = center_neighbor_whole_mean_resp[center_neighbor_key[n][0]][center_neighbor_key[n][1]]
        center_charge = acfc_charge[0][1]
        acfc_list[n].append(center_charge)

        for m in range(len(frag_list[n])):
            if len(acfc_charge[1][acfc_keys[n][m+1]].tolist()) > 1:
                neighbor_charge = random.sample(acfc_charge[1][acfc_keys[n][m+1]].tolist(),1)[0]
            else:
                neighbor_charge = acfc_charge[1][acfc_keys[n][m+1]].tolist()[0]
            acfc_list[frag_list[n][m]].append(neighbor_charge)
    acfc = [ np.array(acfc_list[n]).mean() for n in range(len(data.x)) ]
    return acfc

def cal_acfct(data,center_neighbor_whole_mean_resp):
    feature_list,acfc_keys,frag_list = split_data_to_1hop_fragment(data)
    center_neighbor_key = []
    for n in range(len(data.x)):
        center_neighbor_key.append((acfc_keys[n][0], ('').join(sorted(acfc_keys[n][1:]))))
    acfct_list = []
    for n in range(len(data.x)):
        acfc_charge = center_neighbor_whole_mean_resp[center_neighbor_key[n][0]][center_neighbor_key[n][1]]
        center_charge = torch.tensor(acfc_charge[0][1]).to(dtype=torch.float32).reshape(1,-1)
        neighbor_charges = []
        for key in list(acfc_charge[1].keys()):
            neighbor_charges += acfc_charge[1][key].tolist()
        neighbor_charges = torch.tensor(sorted(neighbor_charges)).to(dtype=torch.float32).reshape(1,-1)
        acfct_ = torch.cat([center_charge,neighbor_charges],dim=-1)
        acfct = torch.zeros((1,7))
        length = acfct_.shape[1]
        acfct[:,0:length] = acfct_ 
        acfct_list.append(acfct)
    acfct_list = torch.cat(acfct_list,dim=0)
    return acfct_list

def cal_acfct_gas(data, gasteiger_charge):
    nx_g = utils.to_networkx(data)
    
    acfct_list = []
    for n in range(len(data.x)):
        center_charge = gasteiger_charge[n].reshape(-1,1)
        neighbor_charges = []
        neighbors = nx_g.neighbors(n)
        for neighbor in neighbors:
            neighbor_charges.append(gasteiger_charge[neighbor])
        neighbor_charges = torch.tensor(sorted(neighbor_charges)).to(dtype=torch.float32).reshape(1,-1)
        acfct_ = torch.cat([center_charge,neighbor_charges],dim=-1)
        acfct = torch.zeros((1,7))
        length = acfct_.shape[1]
        acfct[:,0:length] = acfct_ 
        acfct_list.append(acfct)
    acfct_list = torch.cat(acfct_list,dim=0)
    return acfct_list


def create_moldata(rdkit_mol, mode, center_neighbor_whole_mean_resp, chg_path=None):
    atomic_numbers = get_atomic_number(rdkit_mol)
    atomic_ele_arrangement = get_atomic_ele_arrangement(rdkit_mol)
    atomic_val_electron = get_atomic_val_electron(rdkit_mol)
    gasteiger_charge = get_gasteiger_charge(rdkit_mol)

    bonds_index, bonds_type, bonds_attr = get_bond_index_type(rdkit_mol)       
    atomic_degree = get_atomic_degree(bonds_index, len(atomic_numbers))
    atom_if_aromatic = judge_atom_if_aromatic(rdkit_mol)
    atom_if_conjugate = judge_atom_if_conjugate(rdkit_mol,len(atomic_numbers))
    atom_if_in_ring, atom_in_ringnum, atom_in_smallestringsize = judge_atom_if_in_ring_ringnum_smallestringsize(rdkit_mol, len(atomic_numbers))
    atom_if_in_ring = judge_atom_if_in_ring(rdkit_mol, len(atomic_numbers))
    atom_if_in_aromatic_ring = judge_atom_if_in_aromatic_ring(rdkit_mol, len(atomic_numbers))
    atom_if_in_conjugate_ring = judge_atom_if_in_conjugate_ring(rdkit_mol, len(atomic_numbers))
    angles_index = get_angle_index(bonds_index, len(atomic_numbers))
    dihedrals_index = get_dihedral_index(bonds_index, len(atomic_numbers))
    hop4s_index = get_hop4_index(bonds_index, len(atomic_numbers))
   
    x = torch.cat([atomic_numbers,
               atomic_ele_arrangement,
               atomic_val_electron,
               atomic_degree,
               atom_if_aromatic,
               atom_if_conjugate,
               atom_if_in_ring,
               atom_if_in_aromatic_ring,
               atom_if_in_conjugate_ring,
               atom_in_ringnum,
               atom_in_smallestringsize], dim=-1)
    edge_index = torch.cat([bonds_index, torch.cat([bonds_index[1],bonds_index[0]],dim=0).reshape(2,-1)],dim=-1)
    edge_attr = torch.cat([bonds_attr,bonds_attr], dim=0)
    
    temp_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    all_angle_pair = get_all_path_pair(temp_data, angles_index)
    all_dihedral_pair = get_all_path_pair(temp_data, dihedrals_index)
    all_hop4_pair = get_all_path_pair(temp_data, hop4s_index)
    
    try:
        acfc = torch.tensor(cal_acfc(temp_data,center_neighbor_whole_mean_resp)).reshape(-1,1).to(torch.float32)
        acfct = cal_acfct(temp_data,center_neighbor_whole_mean_resp).reshape(-1,7).to(torch.float32)
    except:
        acfc = gasteiger_charge
        acfct = cal_acfct_gas(temp_data,gasteiger_charge)
    
    if mode =='train':
        atom_distance_matrix = get_atom_distance_matrix(rdkit_mol).reshape(-1,1)
        bonds_distance = get_atom_pair_distance(bonds_index, atom_distance_matrix, len(atomic_numbers))
        angles_distance = get_atom_pair_distance(angles_index, atom_distance_matrix, len(atomic_numbers))
        dihedrals_distance = get_atom_pair_distance(dihedrals_index, atom_distance_matrix, len(atomic_numbers))
        hop4s_distance = get_atom_pair_distance(hop4s_index, atom_distance_matrix, len(atomic_numbers))
        all_pair_angles_distance = get_all_path_pair_dis(all_angle_pair, atom_distance_matrix, len(atomic_numbers))
        all_pair_dihedrals_distance = get_all_path_pair_dis(all_dihedral_pair, atom_distance_matrix, len(atomic_numbers))
        all_pair_hop4s_distance = get_all_path_pair_dis(all_hop4_pair, atom_distance_matrix, len(atomic_numbers))
        if chg_path is not None:
            resp_charge = get_resp_charge(chg_path,len(atomic_numbers))
        else:
            resp_charge = None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    atom_distance_matrix=atom_distance_matrix,
                    bonds_index = bonds_index, angles_index = angles_index, dihedrals_index = dihedrals_index, hop4s_index = hop4s_index,
                    all_angle_pair = all_angle_pair,
                    all_dihedral_pair = all_dihedral_pair,
                    all_hop4_pair = all_hop4_pair,
                    bonds_distance = bonds_distance, angles_distance = angles_distance, dihedrals_distance = dihedrals_distance, hop4s_distance = hop4s_distance,
                    all_pair_angles_distance = all_pair_angles_distance, all_pair_dihedrals_distance = all_pair_dihedrals_distance, all_pair_hop4s_distance = all_pair_hop4s_distance,
                    gasteiger_charge=gasteiger_charge,
                    acfc = acfc,
                    acfct = acfct,
                    resp_charge = resp_charge)

    if mode == 'test':
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,          
                    bonds_index = bonds_index, angles_index = angles_index, dihedrals_index = dihedrals_index, hop4s_index = hop4s_index,
                    all_angle_pair = all_angle_pair,
                    all_dihedral_pair = all_dihedral_pair,
                    all_hop4_pair = all_hop4_pair,
                    acfc = acfc,
                    acfct = acfct,
                    gasteiger_charge=gasteiger_charge)

        