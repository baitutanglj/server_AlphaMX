import numpy as np
import networkx as nx
from scipy.linalg import sqrtm
import torch

def construct_complete_graph(data):
    complete_graph = nx.Graph()
    complete_graph.add_nodes_from(list(range(len(data.x))))
    complete_graph.add_edges_from(data.bonds_index.T.tolist())
    complete_graph.add_edges_from(data.angles_index.T.tolist())
    complete_graph.add_edges_from(data.dihedrals_index.T.tolist())
    complete_graph.add_edges_from(data.hop4s_index.T.tolist())
    return complete_graph

def construct_mol_graph(data):
    mol_graph = nx.Graph()
    mol_graph.add_nodes_from(list(range(len(data.x))))
    mol_graph.add_edges_from(data.bonds_index.T.tolist())
    return mol_graph

def metric_matr_ij(i,j,distance_matrix):
    # 根据距离矩阵计算metric_matr的值
    d2io_1 = (distance_matrix[:,j]**2).sum()/np.array(distance_matrix.shape[0],dtype='float64')
    d2io_2 = 0
    for m in range(1,distance_matrix.shape[0]):
        for k in range(0,m):
            d2io_2 += distance_matrix[m,k]**2
    d2io_2 /= (np.array(distance_matrix.shape[0],dtype='float64'))**2
    d2io = d2io_1 - d2io_2

    d2jo_1 = (distance_matrix[:,i]**2).sum()/np.array(distance_matrix.shape[0],dtype='float64')
    d2jo_2 = 0
    for m in range(1,distance_matrix.shape[0]):
        for k in range(0,m):
            d2jo_2 += distance_matrix[m,k]**2
    d2jo_2 /= (np.array(distance_matrix.shape[0],dtype='float64'))**2
    d2jo = d2jo_1 - d2jo_2
    return (d2io + d2jo - distance_matrix[i,j]**2) /2.0

def gen_coords(distance_matrix):
    # 生成metric_matrix
    metric_matr = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[0]):
            metric_matr[i,j] = metric_matr_ij(i,j,distance_matrix)

    # 获得metric_matr的特征值和特征向量
    eigen_values = np.linalg.eig(metric_matr)[0]
    eigen_values = np.diag(eigen_values)
    eigen_vec = np.linalg.eig(metric_matr)[1]

    # 坐标等于 eigen_vec · eigen_values
    coords = np.real(eigen_vec.dot(sqrtm(eigen_values))).round(6)
    return coords

def build_up_mol(dismatrix, clique, coords):
    whole = [i for i in range(len(dismatrix[0]))]
    while len(rest) != 0:
        rest = [x for x in whole if x not in clique]
        for i in rest:
            index = dismatrix[clique,i].nonzero().numpy()
            if len(index) >= 4:
                B = ((dismatrix[np.array(clique)[index],i])**2-(dismatrix[np.array(clique)[index[0]],i])**2).numpy().reshape(-1,1) + ((np.linalg.norm(coords[index[0]]))**2 - (np.linalg.norm(coords[index.reshape(-1)],axis=1))**2).reshape(-1,1)
                A = (coords[index[0]] - coords[index.reshape(-1)])*2
                fth_coords = np.linalg.pinv(A[1:len(A)]).dot(B[1:len(B)]).T
                coords = np.concatenate((coords,fth_coords),axis=0)
                clique.append(i)
            else:
                pass
    return coords, clique

def multi_dim_scalling(dismatrix):
    n = len(dismatrix[0])
    dismatrix_quare = dismatrix**2
    J = np.identity(n) - 1/n
    B = -0.5 * (J.dot(dismatrix_quare)).dot(J)
    B_eigen_values = np.linalg.eig(B)[0]
    B_eigen_vec = np.linalg.eig(B)[1]
    index_largest3_value = np.argsort(B_eigen_values)[::-1][0:3]
    coords = B_eigen_vec[:,index_largest3_value].dot(np.diag(B_eigen_values[index_largest3_value])**0.5)
    return coords

def triangle_rule(atom_pair, sparse_distance):
    num = len(atom_pair[0])
    if num == 3:
        upper = sparse_distance[atom_pair[:,0],atom_pair[:,1]] + sparse_distance[atom_pair[:,1],atom_pair[:,2]] 
        lower = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,1]] - sparse_distance[atom_pair[:,1],atom_pair[:,2]])
    if num == 4:
        upper1 = sparse_distance[atom_pair[:,0],atom_pair[:,1]] + sparse_distance[atom_pair[:,1],atom_pair[:,3]]
        upper2 = sparse_distance[atom_pair[:,0],atom_pair[:,2]] + sparse_distance[atom_pair[:,2],atom_pair[:,3]]
        upper = torch.min( torch.cat([upper1,upper2],dim=-1),1)[0].reshape(-1,1)
        lower1 = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,1]] - sparse_distance[atom_pair[:,1],atom_pair[:,3]])
        lower2 = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,2]] - sparse_distance[atom_pair[:,2],atom_pair[:,3]]) 
        lower = torch.max( torch.cat([lower1,lower2],dim=-1),1)[0].reshape(-1,1) 
    if num == 5:
        upper1 = sparse_distance[atom_pair[:,0],atom_pair[:,1]] + sparse_distance[atom_pair[:,1],atom_pair[:,4]]
        upper2 = sparse_distance[atom_pair[:,0],atom_pair[:,2]] + sparse_distance[atom_pair[:,2],atom_pair[:,4]]
        upper3 = sparse_distance[atom_pair[:,0],atom_pair[:,3]] + sparse_distance[atom_pair[:,3],atom_pair[:,4]]
        upper = torch.min( torch.cat([upper1,upper2,upper3],dim=-1),1)[0].reshape(-1,1)     
        lower1 = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,1]] - sparse_distance[atom_pair[:,1],atom_pair[:,4]])
        lower2 = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,2]] - sparse_distance[atom_pair[:,2],atom_pair[:,4]])
        lower3 = torch.abs(sparse_distance[atom_pair[:,0],atom_pair[:,3]] - sparse_distance[atom_pair[:,3],atom_pair[:,4]])
        lower = torch.max( torch.cat([lower1,lower2,lower3],dim=-1),1)[0].reshape(-1,1) 
    dis = sparse_distance[atom_pair[:,0],atom_pair[:,num-1]]
    if torch.all(lower < dis) and torch.all(dis < upper):
        return True
    else:
        return False