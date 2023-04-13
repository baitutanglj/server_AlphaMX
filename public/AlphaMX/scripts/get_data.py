import os
from utils import construct_data
from utils.mols_dataset import mols_dataset
from rdkit import Chem
import torch
import pandas as pd

## resp
# chg_files = os.listdir('/dssg/home/sw_xj/aihp/gaussian_result/mols_correct_single/chg/')
# unable_list = []
# for chg_file in chg_files:
#     entry_id = chg_file.split('.')[0]
#     try:
#         mol = Chem.SDMolSupplier('/dssg/home/sw_xj/aihp/workspace/structure_predict/shared_data/mols_correct/'+entry_id+'.sdf')[0]
#         data = construct_data.create_moldata(mol, 'train', '/dssg/home/sw_xj/aihp/gaussian_result/mols_no_I_single/chg/'+chg_file)
#         torch.save(data, '../data/resp_data/'+entry_id+'.resp_data')
#     except:
#         unable_list.append(entry_id)

# pd.DataFrame(unable_list).to_csv('./unable_list.csv')

## acfct
# acfc_para = torch.load('./scripts/acfc/center_neighbor_whole_mean_resp')


# whole_data = mols_dataset('../data/whole_data/')
# for i,data in enumerate(whole_data):
#     acfct = torch.tensor(construct_data.cal_acfct(data,acfc_para))
#     data.acfct = acfct
#     torch.save(data,'../data/whole_data/'+whole_data.name(i).split('.')[0]+'.acfc_data')


# train_data = mols_dataset('../data/train_data/')
# for i,data in enumerate(train_data):
#     acfct = torch.tensor(construct_data.cal_acfct(data,acfc_para))
#     data.acfct = acfct
#     torch.save(data,'../data/train_data/'+train_data.name(i).split('.')[0]+'.acfc_data')

# test_data = mols_dataset('../data/test_data/')
# for i,data in enumerate(test_data):
#     acfct = torch.tensor(construct_data.cal_acfct(data,acfc_para))
#     data.acfct = acfct
#     torch.save(data,'../data/test_data/'+test_data.name(i).split('.')[0]+'.acfc_data')

# valid_data = mols_dataset('../data/valid_data/')
# for i,data in enumerate(valid_data):
#     acfct = torch.tensor(construct_data.cal_acfct(data,acfc_para))
#     data.acfct = acfct
#     torch.save(data,'../data/valid_data/'+valid_data.name(i).split('.')[0]+'.acfc_data')


## all pair path 

whole_data = mols_dataset('../data/whole_data/')
for i,data in enumerate(whole_data):
    data.all_pair_angles_distance = construct_data.get_all_path_pair_dis(data.all_angle_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_dihedrals_distance = construct_data.get_all_path_pair_dis(data.all_dihedral_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_hop4s_distance = construct_data.get_all_path_pair_dis(data.all_hop4_pair, data.atom_distance_matrix, len(data.x))
    
    torch.save(data,'../data/whole_data/'+whole_data.name(i).split('.')[0]+'.acfc_data')

train_data = mols_dataset('../data/train_data/')
for i,data in enumerate(train_data):
    data.all_pair_angles_distance = construct_data.get_all_path_pair_dis(data.all_angle_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_dihedrals_distance = construct_data.get_all_path_pair_dis(data.all_dihedral_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_hop4s_distance = construct_data.get_all_path_pair_dis(data.all_hop4_pair, data.atom_distance_matrix, len(data.x))
    torch.save(data,'../data/train_data/'+train_data.name(i).split('.')[0]+'.acfc_data')

test_data = mols_dataset('../data/test_data/')
for i,data in enumerate(test_data):
    data.all_pair_angles_distance = construct_data.get_all_path_pair_dis(data.all_angle_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_dihedrals_distance = construct_data.get_all_path_pair_dis(data.all_dihedral_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_hop4s_distance = construct_data.get_all_path_pair_dis(data.all_hop4_pair, data.atom_distance_matrix, len(data.x))
    torch.save(data,'../data/test_data/'+test_data.name(i).split('.')[0]+'.acfc_data')

valid_data = mols_dataset('../data/valid_data/')
for i,data in enumerate(valid_data):
    data.all_pair_angles_distance = construct_data.get_all_path_pair_dis(data.all_angle_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_dihedrals_distance = construct_data.get_all_path_pair_dis(data.all_dihedral_pair, data.atom_distance_matrix, len(data.x))
    data.all_pair_hop4s_distance = construct_data.get_all_path_pair_dis(data.all_hop4_pair, data.atom_distance_matrix, len(data.x))
    torch.save(data,'../data/valid_data/'+valid_data.name(i).split('.')[0]+'.acfc_data')