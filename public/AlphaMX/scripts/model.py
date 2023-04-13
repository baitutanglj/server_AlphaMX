import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from .GAT import GAT

class ResLinear(nn.Module):
    def __init__(self, channel):
        super(ResLinear, self).__init__()
        self.fc1 = nn.Linear(channel,channel)
        self.bn1 = nn.BatchNorm1d(channel)


    def forward(self, x):
        row_input = x
        x = self.bn1(self.fc1(x))
        output = F.relu(x + row_input)
        return output

class resGATv2Conv(nn.Module):
    def __init__(self, channels,edge_dim):
        super(resGATv2Conv, self).__init__()
        self.gat1 = GATv2Conv(channels, channels, heads=4, edge_dim=edge_dim, concat=False)
        self.bn1 = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index, edge_attr):
        row_input = x
        x = self.bn1(self.gat1(x, edge_index, edge_attr))
        output = F.relu(x + row_input)
        return output

class resGAT(nn.Module):
    def __init__(self, channels,edge_dim):
        super().__init__()
        self.gat1 = GAT(channels, channels, heads=4, edge_dim=edge_dim, concat=False)
        self.bn1 = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index, edge_attr):
        row_input = x
        x = self.bn1(self.gat1(x, edge_index, edge_attr))
        output = F.relu(x + row_input)
        return output

class resGAT_concat(nn.Module):
    def __init__(self, channels,edge_dim):
        super().__init__()
        self.gat1 = GAT(channels, channels, heads=4, edge_dim=edge_dim, concat=True)
        self.fc1 = nn.Linear(channels*4,channels)
        self.bn1 = nn.BatchNorm1d(channels)


    def forward(self, x, edge_index, edge_attr):
        row_input = x
        x = self.bn1(self.fc1(self.gat1(x, edge_index, edge_attr)))
        output = F.relu(x + row_input)
        return output

class ResPair_Distance(nn.Module):
    def __init__(self,dim):
        super(ResPair_Distance, self).__init__()
        self.fc1 = nn.Linear(dim,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = ResLinear(256)
        self.fc3 = ResLinear(256)
        self.fc4 = ResLinear(256)
        self.fc5 = nn.Linear(256,32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32,1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.fc6(x))
        return x

class ResPair_Distance2(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc1 = nn.Linear(dim,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = ResLinear(256)
        self.fc3 = ResLinear(256)
        self.fc4 = ResLinear(256)
        self.fc5 = ResLinear(256)
        self.fc6 = ResLinear(256)       
        self.fc7 = nn.Linear(256,32)
        self.bn7 = nn.BatchNorm1d(32)
        self.fc8 = nn.Linear(32,1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.fc8(x))
        return x


class ResPair_Distance_Triangle(nn.Module):
    def __init__(self,dim):
        super(ResPair_Distance_Triangle, self).__init__()
        self.fc1 = nn.Linear(dim,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = ResLinear(256)
        self.fc3 = ResLinear(256)
        self.fc4 = ResLinear(256)
        self.fc5 = nn.Linear(256,32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32,1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = F.relu(self.bn5(self.fc5(x)))
        x = torch.sigmoid(self.fc6(x))
        return x

# separately
class RGA_direct_with_charge_encode(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_encode, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(32)

        self.dis_predict = ResPair_Distance(64)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        return charge

    def forward(self,data, charge, target_index):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x, data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)

        latent = torch.cat([top_latent, charge_latent],dim=1)
 
        pair = latent[target_index[0]]+latent[target_index[1]]
        dis = self.dis_predict(pair)
        return latent, dis

# whole_model
class RGA_direct_encode_whole(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_encode_whole, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(65)
        self.angle_predict = ResPair_Distance(65)
        self.dihedral_predict = ResPair_Distance(65)
        self.hop4_predict = ResPair_Distance(65)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def forward(self,data):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)

        latent = top_latent

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis

class RGA_direct_encode_whole_no1234(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(64)
        self.angle_predict = ResPair_Distance(64)
        self.dihedral_predict = ResPair_Distance(64)
        self.hop4_predict = ResPair_Distance(64)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def forward(self,data):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)

        latent = top_latent

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)

        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)

        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)

        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis

class RGA_direct_with_charge_encode_whole(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_encode_whole, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(129)
        self.angle_predict = ResPair_Distance(129)
        self.dihedral_predict = ResPair_Distance(129)
        self.hop4_predict = ResPair_Distance(129)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        return charge

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        latent = torch.cat([top_latent, charge_latent],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis




class RGA_direct_with_charge_acfct_encode_whole(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_acfct_encode_whole, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis


class RGA_direct_with_charge_interac_acfct_encode_whole(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_fc1 = nn.Linear(1,128)
        self.charge_bn1 = nn.BatchNorm1d(128)
        self.charge_gat2 = GATv2Conv(128,128,heads=4, concat=False)
        self.charge_bn2 = nn.BatchNorm1d(128)
        self.charge_fc3 = nn.Linear(128,32)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_conv_fc1 = nn.Linear(7,128)
        self.acfct_conv_bn1 = nn.BatchNorm1d(128)
        self.acfct_conv_gat2= GATv2Conv(128,128,heads=4, concat=False)
        self.acfct_conv_bn2 = nn.BatchNorm1d(128)
        self.acfct_conv_fc3 = nn.Linear(128,32)
        self.acfct_conv_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(257)
        self.angle_predict = ResPair_Distance(257)
        self.dihedral_predict = ResPair_Distance(257)
        self.hop4_predict = ResPair_Distance(257)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def chg_interac_graph(self,data):
        chg_interac_index = torch.cat([ data.bonds_index, torch.cat([data.bonds_index[1],data.bonds_index[0]],dim=0).reshape(2,-1),
                                        data.angles_index, torch.cat([data.angles_index[1],data.angles_index[0]],dim=0).reshape(2,-1),
                                        data.dihedrals_index, torch.cat([data.dihedrals_index[1],data.dihedrals_index[0]],dim=0).reshape(2,-1),
                                        data.hop4s_index, torch.cat([data.hop4s_index[1],data.hop4s_index[0]],dim=0).reshape(2,-1) 
                                        ],dim=-1)
        return chg_interac_index

    def charge_encoder(self,charge,edge_index):
        charge = F.relu(self.charge_bn1(self.charge_fc1(charge)))
        charge = F.relu(self.charge_bn2(self.charge_gat2(charge, edge_index)))
        charge = F.relu(self.charge_bn3(self.charge_fc3(charge)))
        return charge

    def acfct_conv_encoder(self,acfct,edge_index):
        acfct_conv = F.relu(self.acfct_conv_bn1(self.acfct_conv_fc1(acfct)))
        acfct_conv = F.relu(self.acfct_conv_bn2(self.acfct_conv_gat2(acfct_conv, edge_index)))
        acfct_conv = F.relu(self.acfct_conv_bn3(self.acfct_conv_fc3(acfct_conv)))
        return acfct_conv

    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct


    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)

        chg_interac_index = self.chg_interac_graph(data)

        charge_latent = self.charge_encoder(charge, chg_interac_index)
        acfct_conv_latent = self.acfct_conv_encoder(data.acfct, chg_interac_index)

        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent, acfct_conv_latent, acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis


class RGA_direct_with_charge_interac_acfct_encode_whole_no1234(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_fc1 = nn.Linear(1,128)
        self.charge_bn1 = nn.BatchNorm1d(128)
        self.charge_gat2 = GATv2Conv(128,128,heads=4, concat=False)
        self.charge_bn2 = nn.BatchNorm1d(128)
        self.charge_fc3 = nn.Linear(128,32)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_conv_fc1 = nn.Linear(7,128)
        self.acfct_conv_bn1 = nn.BatchNorm1d(128)
        self.acfct_conv_gat2= GATv2Conv(128,128,heads=4, concat=False)
        self.acfct_conv_bn2 = nn.BatchNorm1d(128)
        self.acfct_conv_fc3 = nn.Linear(128,32)
        self.acfct_conv_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(256)
        self.angle_predict = ResPair_Distance(256)
        self.dihedral_predict = ResPair_Distance(256)
        self.hop4_predict = ResPair_Distance(256)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def chg_interac_graph(self,data):
        chg_interac_index = torch.cat([ data.bonds_index, torch.cat([data.bonds_index[1],data.bonds_index[0]],dim=0).reshape(2,-1),
                                        data.angles_index, torch.cat([data.angles_index[1],data.angles_index[0]],dim=0).reshape(2,-1),
                                        data.dihedrals_index, torch.cat([data.dihedrals_index[1],data.dihedrals_index[0]],dim=0).reshape(2,-1),
                                        data.hop4s_index, torch.cat([data.hop4s_index[1],data.hop4s_index[0]],dim=0).reshape(2,-1) 
                                        ],dim=-1)
        return chg_interac_index

    def charge_encoder(self,charge,edge_index):
        charge = F.relu(self.charge_bn1(self.charge_fc1(charge)))
        charge = F.relu(self.charge_bn2(self.charge_gat2(charge, edge_index)))
        charge = F.relu(self.charge_bn3(self.charge_fc3(charge)))
        return charge

    def acfct_conv_encoder(self,acfct,edge_index):
        acfct_conv = F.relu(self.acfct_conv_bn1(self.acfct_conv_fc1(acfct)))
        acfct_conv = F.relu(self.acfct_conv_bn2(self.acfct_conv_gat2(acfct_conv, edge_index)))
        acfct_conv = F.relu(self.acfct_conv_bn3(self.acfct_conv_fc3(acfct_conv)))
        return acfct_conv

    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct


    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)

        chg_interac_index = self.chg_interac_graph(data)

        charge_latent = self.charge_encoder(charge, chg_interac_index)
        acfct_conv_latent = self.acfct_conv_encoder(data.acfct, chg_interac_index)

        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent, acfct_conv_latent, acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)

        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)

        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)

        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)


        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis


class RGA_direct_with_charge_interac_acfct_encode_whole_no1234_resdis2(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_fc1 = nn.Linear(1,128)
        self.charge_bn1 = nn.BatchNorm1d(128)
        self.charge_gat2 = GATv2Conv(128,128,heads=4, concat=False)
        self.charge_bn2 = nn.BatchNorm1d(128)
        self.charge_fc3 = nn.Linear(128,32)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_conv_fc1 = nn.Linear(7,128)
        self.acfct_conv_bn1 = nn.BatchNorm1d(128)
        self.acfct_conv_gat2= GATv2Conv(128,128,heads=4, concat=False)
        self.acfct_conv_bn2 = nn.BatchNorm1d(128)
        self.acfct_conv_fc3 = nn.Linear(128,32)
        self.acfct_conv_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance2(256)
        self.angle_predict = ResPair_Distance2(256)
        self.dihedral_predict = ResPair_Distance2(256)
        self.hop4_predict = ResPair_Distance2(256)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def chg_interac_graph(self,data):
        chg_interac_index = torch.cat([ data.bonds_index, torch.cat([data.bonds_index[1],data.bonds_index[0]],dim=0).reshape(2,-1),
                                        data.angles_index, torch.cat([data.angles_index[1],data.angles_index[0]],dim=0).reshape(2,-1),
                                        data.dihedrals_index, torch.cat([data.dihedrals_index[1],data.dihedrals_index[0]],dim=0).reshape(2,-1),
                                        data.hop4s_index, torch.cat([data.hop4s_index[1],data.hop4s_index[0]],dim=0).reshape(2,-1) 
                                        ],dim=-1)
        return chg_interac_index

    def charge_encoder(self,charge,edge_index):
        charge = F.relu(self.charge_bn1(self.charge_fc1(charge)))
        charge = F.relu(self.charge_bn2(self.charge_gat2(charge, edge_index)))
        charge = F.relu(self.charge_bn3(self.charge_fc3(charge)))
        return charge

    def acfct_conv_encoder(self,acfct,edge_index):
        acfct_conv = F.relu(self.acfct_conv_bn1(self.acfct_conv_fc1(acfct)))
        acfct_conv = F.relu(self.acfct_conv_bn2(self.acfct_conv_gat2(acfct_conv, edge_index)))
        acfct_conv = F.relu(self.acfct_conv_bn3(self.acfct_conv_fc3(acfct_conv)))
        return acfct_conv

    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct


    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)

        chg_interac_index = self.chg_interac_graph(data)

        charge_latent = self.charge_encoder(charge, chg_interac_index)
        acfct_conv_latent = self.acfct_conv_encoder(data.acfct, chg_interac_index)

        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent, acfct_conv_latent, acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)

        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)

        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)

        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)


        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis

class RGA_direct_with_charge_acfct_encode_whole_kaiming(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GAT(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GAT(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GAT(1,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis
        


class RGA_direct_with_charge_infea_acfct_encode_whole(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features + 1
        
        self.top_init_norm = nn.BatchNorm1d(self.num_features)
        self.en_gat1 = GATv2Conv(self.num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)


        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(129)
        self.angle_predict = ResPair_Distance(129)
        self.dihedral_predict = ResPair_Distance(129)
        self.hop4_predict = ResPair_Distance(129)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    

    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device
        x = torch.cat([data.x,charge],dim=1)
        
        top_latent = self.topo_encoder(x,data.edge_index, data.edge_attr)

        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis
        

class RGA_direct_with_charge_acfct_encode_whole_coords(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_acfct_encode_whole_coords, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,32)
        self.acfct_fn2 = ResLinear(32)
        self.acfct_bn1 = nn.BatchNorm1d(32)
        self.acfct_bn2 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

        self.coords_gat1 = GATv2Conv(96,256,heads=4, edge_dim=1, concat=False)
        self.coords_bn1 = nn.BatchNorm1d(256)
        self.coords_gat2 = resGATv2Conv(channels=256,edge_dim=1)
        self.coords_fc1 = ResLinear(256)
        self.coords_fcbn2 = nn.BatchNorm1d(256)
        self.coords_fc2 = nn.Linear(256,3)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        return acfct

    def coords_pre(self,x,edge_index,edge_attr):
        x = F.relu(self.coords_bn1(self.coords_gat1(x, edge_index, edge_attr)))
        x = self.coords_gat2(x, edge_index, edge_attr)
        x = F.relu(self.coords_fcbn2(self.coords_fc1(x)))
        x = F.relu(self.coords_fc2(x))
        return x

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)

        x = latent
        dis_index = torch.cat([data.bonds_index,data.bonds_index,
                                data.angles_index,data.angles_index,
                                data.dihedrals_index,data.dihedrals_index,
                                data.hop4s_index,data.hop4s_index], dim=1)
        dis = torch.cat([bond_dis,bond_dis,
                            angle_dis,angle_dis,
                            dihedral_dis,dihedral_dis,
                            hop4_dis,hop4_dis], dim=0)

        coords = self.coords_pre(x, dis_index, dis)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis, coords

class RGA_direct_with_charge_encode_whole2(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_encode_whole2, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,256,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(256)
        self.charge_encode2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.charge_encode3 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.bond_predict = ResPair_Distance(129)
        self.angle_predict = ResPair_Distance(129)
        self.dihedral_predict = ResPair_Distance(129)
        self.hop4_predict = ResPair_Distance(129)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        charge = self.charge_encode2(charge, edge_index, edge_attr)
        charge = F.relu(self.charge_bn3(self.charge_encode3(charge, edge_index, edge_attr)))
        return charge

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        latent = torch.cat([top_latent, charge_latent],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis

class RGA_direct_with_charge_acfct_encode_whole2(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_acfct_encode_whole2, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,256,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(256)
        self.charge_encode2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.charge_encode3 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,256)
        self.acfct_bn1 = nn.BatchNorm1d(256)
        self.acfct_fn2 = ResLinear(256)
        self.acfct_bn2 = nn.BatchNorm1d(256)
        self.acfct_fn3 = nn.Linear(256,32)
        self.acfct_bn3 = nn.BatchNorm1d(32)       

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        charge = self.charge_encode2(charge, edge_index, edge_attr)
        charge = F.relu(self.charge_bn3(self.charge_encode3(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        acfct = F.relu(self.acfct_bn3(self.acfct_fn3(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis

class RGA_direct_with_charge_acfct_encode_whole2_coords(nn.Module):
    def __init__(self,num_features,edge_dim):
        super(RGA_direct_with_charge_acfct_encode_whole2_coords, self).__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GATv2Conv(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GATv2Conv(1,256,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(256)
        self.charge_encode2 = resGATv2Conv(channels=256,edge_dim=edge_dim)
        self.charge_encode3 = GATv2Conv(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,256)
        self.acfct_bn1 = nn.BatchNorm1d(256)
        self.acfct_fn2 = ResLinear(256)
        self.acfct_bn2 = nn.BatchNorm1d(256)
        self.acfct_fn3 = nn.Linear(256,32)
        self.acfct_bn3 = nn.BatchNorm1d(32)   

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

        self.coords_gat1 = GATv2Conv(96,256,heads=4, edge_dim=1, concat=False)
        self.coords_bn1 = nn.BatchNorm1d(256)
        self.coords_gat2 = resGATv2Conv(channels=256,edge_dim=1)
        self.coords_fc1 = ResLinear(256)
        self.coords_fcbn2 = nn.BatchNorm1d(256)
        self.coords_fc2 = nn.Linear(256,3)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        charge = self.charge_encode2(charge, edge_index, edge_attr)
        charge = F.relu(self.charge_bn3(self.charge_encode3(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        acfct = F.relu(self.acfct_bn3(self.acfct_fn3(acfct)))
        return acfct

    def coords_pre(self,x,edge_index,edge_attr):
        x = F.relu(self.coords_bn1(self.coords_gat1(x, edge_index, edge_attr)))
        x = self.coords_gat2(x, edge_index, edge_attr)
        x = F.relu(self.coords_fcbn2(self.coords_fc1(x)))
        x = F.relu(self.coords_fc2(x))
        return x

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)

        x = latent
        dis_index = torch.cat([data.bonds_index,data.bonds_index,
                                data.angles_index,data.angles_index,
                                data.dihedrals_index,data.dihedrals_index,
                                data.hop4s_index,data.hop4s_index], dim=1)
        dis = torch.cat([bond_dis,bond_dis,
                            angle_dis,angle_dis,
                            dihedral_dis,dihedral_dis,
                            hop4_dis,hop4_dis], dim=0)

        coords = self.coords_pre(x, dis_index, dis)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis, coords

class RGA_direct_with_charge_acfct_encode_whole2_kaiming(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GAT(num_features,256,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn1 = nn.BatchNorm1d(256)
        self.en_gat2 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGAT(channels=256,edge_dim=edge_dim)
        self.en_gat5 = GAT(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GAT(1,256,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn1 = nn.BatchNorm1d(256)
        self.charge_encode2 = resGAT(channels=256,edge_dim=edge_dim)
        self.charge_encode3 = GAT(256,32,heads=4, edge_dim=edge_dim, concat=False)
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,256)
        self.acfct_bn1 = nn.BatchNorm1d(256)
        self.acfct_fn2 = ResLinear(256)
        self.acfct_bn2 = nn.BatchNorm1d(256)
        self.acfct_fn3 = nn.Linear(256,32)
        self.acfct_bn3 = nn.BatchNorm1d(32)       

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_gat1(x, edge_index, edge_attr)))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_gat5(x, edge_index, edge_attr)))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_encode1(charge, edge_index, edge_attr)))
        charge = self.charge_encode2(charge, edge_index, edge_attr)
        charge = F.relu(self.charge_bn3(self.charge_encode3(charge, edge_index, edge_attr)))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        acfct = F.relu(self.acfct_bn3(self.acfct_fn3(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis


class RGA_direct_with_charge_acfct_encode_whole2_kaiming_cancat(nn.Module):
    def __init__(self,num_features,edge_dim):
        super().__init__()
        self.num_features = num_features
        
        self.top_init_norm = nn.BatchNorm1d(num_features)
        self.en_gat1 = GAT(num_features,256,heads=4, edge_dim=edge_dim, concat=True)
        self.en_fc1 = nn.Linear(256*4,256)
        self.en_bn1 = nn.BatchNorm1d(256)
    
        self.en_gat2 = resGAT_concat(channels=256,edge_dim=edge_dim)
        self.en_gat3 = resGAT_concat(channels=256,edge_dim=edge_dim)
        self.en_gat4 = resGAT_concat(channels=256,edge_dim=edge_dim)

        self.en_gat5 = GAT(256,32,heads=4, edge_dim=edge_dim, concat=True)
        self.en_fc5 = nn.Linear(32*4,32)
        self.en_bn5 = nn.BatchNorm1d(32)

        self.charge_init_norm = nn.BatchNorm1d(1)         
        self.charge_encode1 = GAT(1,256,heads=4, edge_dim=edge_dim, concat=True)
        self.charge_fc1 = nn.Linear(256*4,256)
        self.charge_bn1 = nn.BatchNorm1d(256)

        self.charge_encode2 = resGAT_concat(channels=256,edge_dim=edge_dim)

        self.charge_encode3 = GAT(256,32,heads=4, edge_dim=edge_dim, concat=True)
        self.charge_fc3 = nn.Linear(32*4,32)        
        self.charge_bn3 = nn.BatchNorm1d(32)

        self.acfct_fn1 = nn.Linear(7,256)
        self.acfct_bn1 = nn.BatchNorm1d(256)
        self.acfct_fn2 = ResLinear(256)
        self.acfct_bn2 = nn.BatchNorm1d(256)
        self.acfct_fn3 = nn.Linear(256,32)
        self.acfct_bn3 = nn.BatchNorm1d(32)       

        self.bond_predict = ResPair_Distance(193)
        self.angle_predict = ResPair_Distance(193)
        self.dihedral_predict = ResPair_Distance(193)
        self.hop4_predict = ResPair_Distance(193)

    def topo_encoder(self,x,edge_index, edge_attr):
        x = self.top_init_norm(x)
        x = F.relu(self.en_bn1(self.en_fc1(self.en_gat1(x, edge_index, edge_attr))))
        x = self.en_gat2(x, edge_index, edge_attr)
        x = self.en_gat3(x, edge_index, edge_attr)
        x = self.en_gat4(x, edge_index, edge_attr)
        x = F.relu(self.en_bn5(self.en_fc5(self.en_gat5(x, edge_index, edge_attr))))
        return x
    
    def charge_encoder(self,charge,edge_index, edge_attr):
        charge = self.charge_init_norm(charge)
        charge = F.relu(self.charge_bn1(self.charge_fc1(self.charge_encode1(charge, edge_index, edge_attr))))
        charge = self.charge_encode2(charge, edge_index, edge_attr)
        charge = F.relu(self.charge_bn3(self.charge_fc3(self.charge_encode3(charge, edge_index, edge_attr))))
        return charge
    
    def acfct_encoder(self, acfct):
        acfct = F.relu(self.acfct_bn1(self.acfct_fn1(acfct)))
        acfct = F.relu(self.acfct_bn2(self.acfct_fn2(acfct)))
        acfct = F.relu(self.acfct_bn3(self.acfct_fn3(acfct)))
        return acfct

    def forward(self,data, charge):
        
        device = data.x.device

        top_latent = self.topo_encoder(data.x,data.edge_index, data.edge_attr)
        charge_latent = self.charge_encoder(charge, data.edge_index, data.edge_attr)
        acfct = self.acfct_encoder(data.acfct)

        latent = torch.cat([top_latent, charge_latent,acfct],dim=1)

        bond_pair = torch.cat([latent[data.bonds_index[0]],latent[data.bonds_index[1]]],dim=1)
        bond_pair = torch.cat([bond_pair, torch.ones(len(bond_pair),1).to(device)], dim=1)
        angle_pair = torch.cat([latent[data.angles_index[0]],latent[data.angles_index[1]]],dim=1)
        angle_pair = torch.cat([angle_pair, torch.full((len(angle_pair),1),fill_value=2.0).to(device)], dim=1)
        dihedral_pair = torch.cat([latent[data.dihedrals_index[0]],latent[data.dihedrals_index[1]]],dim=1)
        dihedral_pair = torch.cat([dihedral_pair, torch.full((len(dihedral_pair),1),fill_value=3.0).to(device)], dim=1)
        hop4_pair = torch.cat([latent[data.hop4s_index[0]],latent[data.hop4s_index[1]]],dim=1)
        hop4_pair = torch.cat([hop4_pair, torch.full((len(hop4_pair),1),fill_value=4.0).to(device)], dim=1)

        bond_dis = self.bond_predict(bond_pair)
        angle_dis = self.angle_predict(angle_pair)
        dihedral_dis = self.dihedral_predict(dihedral_pair)
        hop4_dis = self.hop4_predict(hop4_pair)
        
        return latent, bond_dis, angle_dis, dihedral_dis, hop4_dis