import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.nn import functional as F
from scripts.utils.mols_dataset import mols_dataset
from scripts.model import RGA_direct_with_charge_acfct_encode_whole
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

def train(num_workers):
    
    train_dataset = mols_dataset('../data/train_data/') #load train dataset
    valid_dataset = mols_dataset('../data/valid_data/') #load train dataset

    model = RGA_direct_with_charge_acfct_encode_whole(len(train_dataset[0].x[0]),5).to(device)

    # define loss function, optimizer, batch_size, epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.001)
    criterion = nn.MSELoss(reduction='mean') 
    batch_size = 256
    epochs = 500

    batch_loss_list = []
    valid_epoch_loss_list = []

    for epoch in range(epochs):

        model.train()    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for i, batch in enumerate(train_loader):
            # learn per batch
            batch = batch.to(device)
            latent, bond_dis, angle_dis, dihedral_dis, hop4_dis = model(batch, batch.acfc)

            loss =  criterion(bond_dis, batch.bonds_distance) + criterion(angle_dis, batch.angles_distance) + criterion(dihedral_dis, batch.dihedrals_distance) + criterion(hop4_dis, batch.hop4s_distance)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss_list.append(loss.item())

            if i % 100 == 0:                
                # save model loss
                batch_loss = pd.DataFrame(batch_loss_list)
                batch_loss.to_csv('./rga_direct_with_acfct_encode_whole_batch_loss_all.csv')
       

        model.eval()
        valid_loss_list = []
        for i, batch in enumerate(valid_loader):
            batch = batch.to(device)
            latent, bond_dis, angle_dis, dihedral_dis, hop4_dis = model(batch, batch.acfc)

            valid_loss =  F.mse_loss(bond_dis, batch.bonds_distance) + F.mse_loss(angle_dis, batch.angles_distance) + F.mse_loss(dihedral_dis, batch.dihedrals_distance) + F.mse_loss(hop4_dis, batch.hop4s_distance)
            
            valid_loss_list.append(valid_loss.item())
        valid_epoch_loss_list.append(np.array(valid_loss_list).mean())
        valid_epoch_loss = pd.DataFrame(valid_epoch_loss_list)
        valid_epoch_loss.to_csv('./rga_direct_with_acfct_encode_whole_valid_epoch_loss_all.csv')

        if (epoch+1) % 5 == 0 and (epoch+1) != 0:
            torch.save(model.state_dict(),'./rga_direct_with_acfct_encode_whole_model_epoch'+str(epoch+1)+'_all.pkl', _use_new_zipfile_serialization=False)

    # save model parameters 
    torch.save(model.state_dict(),'./rga_direct_with_acfct_encode_whole_model_epoch'+str(epoch+1)+'_all.pkl', _use_new_zipfile_serialization=False)

    # save model loss
    batch_loss = pd.DataFrame(batch_loss_list)
    batch_loss.to_csv('./rga_direct_with_acfct_encode_whole_batch_loss_all.csv')


if __name__ == '__main__':
    num_workers = sys.argv[1]
    train(int(num_workers))