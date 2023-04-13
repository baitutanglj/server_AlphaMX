from collections import defaultdict
import networkx as nx
import torch
from ..utils.mols_dataset import  mols_dataset
from torch_geometric import utils
import copy
import numpy as np

dataset = mols_dataset('./data/whole_data/')


def split_dataset_to_1hop_fragment(dataset):
    feature_list = defaultdict(list)
    resp_list = defaultdict(list)
    n = 0
    for data in dataset:
        nx_G = utils.to_networkx(data)
        frag_list = {}
        atom_bond_dict = {}
        for atom in  range(len(data.x)):
            frag_list[atom] = list(nx_G.neighbors(atom))
            feature_list[n].append(data.x[atom])
            resp_list[n].append(data.resp_charge[atom].item())
            atom_bond_dict[atom] = data.edge_attr[torch.nonzero(data.edge_index[0]==atom).squeeze().tolist()].reshape(-1,5)
            for i,neighbor in enumerate(list(nx_G.neighbors(atom))):
                feature_list[n].append(torch.cat([data.x[neighbor],atom_bond_dict[atom][i]],dim=-1))
                resp_list[n].append(data.resp_charge[neighbor].item())
            n += 1
    return feature_list,resp_list

feature_list,resp_list = split_dataset_to_1hop_fragment(dataset)

feature_dict2 = defaultdict(list)
for n in range(len(feature_list)):
    feature_dict2[n].append( {('').join( [str(int(f)) for f in feature_list[n][0].tolist()]): resp_list[n][0]})
    resp_dict = []
    for i in range(1,len(feature_list[n])):
        resp_dict.append( (('').join( [str(int(f)) for f in feature_list[n][i].tolist()]),resp_list[n][i]) )
    feature_dict2[n].append(resp_dict)

center_dict = defaultdict(list)
for n in range(len(feature_dict2)):
    key = list(feature_dict2[n][0].keys())[0]
    center_dict[key].append(feature_dict2[n])

center_dict_keys = list(center_dict.keys())
center_dict_neighbor_dict = {}
for key in center_dict_keys:
    temp = defaultdict(list)
    for n in range(len(center_dict[key])):
        temp_key = []
        for m in range(len(center_dict[key][n][1])):
            temp_key.append(center_dict[key][n][1][m][0])
        temp[ ('').join(sorted(temp_key)) ].append(center_dict[key][n])    
    center_dict_neighbor_dict[key] = temp

center_dict_neighbor_dict_resp = copy.deepcopy(center_dict_neighbor_dict)
for center_key in center_dict_keys:
    neighbor_keys = list(center_dict_neighbor_dict[center_key].keys())
    for neighbor_key in neighbor_keys:
        for n in range(len(center_dict_neighbor_dict[center_key][neighbor_key])):
            temp = defaultdict(list)
            for m in range(len(center_dict_neighbor_dict[center_key][neighbor_key][n][1])):
                temp[center_dict_neighbor_dict[center_key][neighbor_key][n][1][m][0]].append(center_dict_neighbor_dict[center_key][neighbor_key][n][1][m][1])
            temp[center_dict_neighbor_dict[center_key][neighbor_key][n][1][m][0]].sort()
            center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1] = temp

center_neighbor_whole_resp = {}
for center_key in center_dict_keys:
    whole_charge = {}
    for neighbor_key in list(center_dict_neighbor_dict_resp[center_key].keys()):
        center_charge = []
        neighbor_charge = defaultdict(list)
        for n in range(len(center_dict_neighbor_dict_resp[center_key][neighbor_key])):
            center_charge.append(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][0][center_key])
            for solo_neighbor_key in list(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1].keys()):
                neighbor_charge[solo_neighbor_key].append(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1][solo_neighbor_key])
        whole_charge[neighbor_key] = [(center_key,center_charge),neighbor_charge]
    center_neighbor_whole_resp[center_key] = whole_charge

center_neighbor_whole_mean_resp = {}
for center_key in center_dict_keys:
    whole_charge = {}
    for neighbor_key in list(center_dict_neighbor_dict_resp[center_key].keys()):
        center_charge = []
        neighbor_charge = defaultdict(list)
        for n in range(len(center_dict_neighbor_dict_resp[center_key][neighbor_key])):
            center_charge.append(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][0][center_key])
            for solo_neighbor_key in list(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1].keys()):
                neighbor_charge[solo_neighbor_key].append(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1][solo_neighbor_key])
        for solo_neighbor_key in list(center_dict_neighbor_dict_resp[center_key][neighbor_key][n][1].keys()): 
            neighbor_charge[solo_neighbor_key] = np.array(neighbor_charge[solo_neighbor_key]).mean(0)
        whole_charge[neighbor_key] = [(center_key,np.array(center_charge).mean()),neighbor_charge]
    center_neighbor_whole_mean_resp[center_key] = whole_charge

torch.save(center_neighbor_whole_mean_resp,'center_neighbor_whole_mean_resp')