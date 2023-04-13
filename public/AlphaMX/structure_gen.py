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
os.chdir(dir)

device = torch.device('cpu')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
acfc_para = torch.load('./scripts/acfc/center_neighbor_whole_mean_resp')
# acfct
model = RGA_direct_with_charge_interac_acfct_encode_whole_no1234(14,5).to(device)
model.load_state_dict(torch.load('./model/CDGNN.pkl',map_location=device))
model = model.eval()

def structure_gen(mol, data, path):
    
    with torch.no_grad():
            
        output = model(data,data.acfc)
        latent, pred_bond_dis, pred_angle_dis, pred_dihedral_dis, pred_hop4_dis = output


    with open('test.data', 'w') as f:
        for index,dis in zip(data.bonds_index.T,pred_bond_dis):
            f.write('%8d %10d %24.16f %24.16f\n' % (index[0].item()+1,index[1].item()+1,dis.item(),dis.item()))
        for index,dis in zip(data.angles_index.T,pred_angle_dis):
            f.write('%8d %10d %24.16f %24.16f\n' % (index[0].item()+1,index[1].item()+1,dis.item(),dis.item()))
        for index,dis in zip(data.dihedrals_index.T, pred_dihedral_dis):
            f.write('%8d %10d %24.16f %24.16f\n' % (index[0].item()+1,index[1].item()+1,dis.item(),dis.item()))
        for index,dis in zip(data.hop4s_index.T, pred_hop4_dis):
            f.write('%8d %10d %24.16f %24.16f\n' % (index[0].item()+1,index[1].item()+1,dis.item(),dis.item()))

    subprocess.call(['./scripts/dgsol','-s10','test.data'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with open('dg.sol', 'r') as dg:
        lines = dg.readlines()
    lines = [line.split() for line in lines]
    
    with open('dg.sum', 'r') as dgsum:
        sum_info = dgsum.readlines()
    sum_info = [line.split() for line in sum_info][5:]
    n = np.array(sum_info,dtype=float)[:,2:].mean(1).argmin()

    start_line = (len(data.x)+3)*n
    end_line = (len(data.x)+3)*n + len(data.x)

    coords_lines = np.array(lines[start_line:end_line],dtype=float)
    coords_liness_sym = -coords_lines

    sdf_write_from_mol(mol, coords_lines, path)
    sdf_write_from_mol(mol, coords_liness_sym, path+'_sym')


    print('the sdf file has been saved in '+ path)
    os.remove('./test.data')
    os.remove('./dg.sol')
    os.remove('./dg.sum')



if __name__ == '__main__':
    """
    C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl
    smiles
    /mnt/home/linjie/projects/AlphaMX/a.sdf
    """
    input = sys.argv[1]
    file_type = sys.argv[2]
    output_path = sys.argv[3]
    if file_type == 'sdf':
        mol = Chem.SDMolSupplier(input)[0]
    elif file_type == 'smiles':
        mol = Chem.MolFromSmiles(input)
    data = construct_data.create_moldata(mol, 'test', acfc_para)
    data = data.to(device)
    structure_gen(mol, data, output_path)