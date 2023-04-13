import os
from openbabel import pybel
# import pybel
import copy
from rdkit import Chem 

'''merge many sdf files into single sdf '''
def merge_sdf_files(sdf_files_path, target_sdffile_savepath):
    files = os.listdir(sdf_files_path)
    with open(target_sdffile_savepath,'a') as whole:
        for filename in files:
            with open(sdf_files_path+filename, 'r') as reader:
                lines = reader.readlines()
            whole.writelines(lines)

def sdf_write(lines,data,name,save_path):
    atom = {6:'C',8:'O',7:'N',15:'P',16:'S',9:'F',17:'Cl',35:'Br',53:'I'}
    bond_order = {0:1,1:2,2:3,3:4}
    sdf_filename = name + '.sdf'
    with open(save_path+sdf_filename, 'w') as f:
        f.write(f'''{name}
--rcdd 334

''')
        f.write("%3.0f%3.0f\n" % (len(data.x),len(data.bonds_index[0])))
        for n in range(len(data.x)):
            f.write("%10.4f%10.4f%10.4f %-3s\n" % (lines[n,0],lines[n,1],lines[n,2],atom[int(data.x[n,0].item())]))
        for n in range(len(data.bonds_index[0])):
            f.write("%3.0f%3.0f%3.0f\n" % (data.bonds_index[0,n].item()+1,data.bonds_index[1,n].item()+1,bond_order[data.edge_attr[n][0:4].nonzero().item()]))
        f.write(f'''M  END
$$$$\n''')

def fchk2sdf(fchk_filepath, output_filepath):
    mols = pybel.readfile('fchk', fchk_filepath)
    output_file = pybel.Outputfile('sdf',output_filepath, overwrite=True)
    for m in mols:
        output_file.write(m)


def sdf_write_from_mol(mol,coords_lines,save_path):
    temp_mol = copy.copy(mol)
    conf = Chem.Conformer()
    for n in range(len(coords_lines)):
        conf.SetAtomPosition(n,coords_lines[n])
    temp_mol.RemoveAllConformers()
    temp_mol.AddConformer(conf)
    sdf_writer = Chem.SDWriter(save_path)
    sdf_writer.write(temp_mol)
    sdf_writer.close() 
    return 0
