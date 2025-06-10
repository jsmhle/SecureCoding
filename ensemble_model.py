import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data as GeoData

class GCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2):
        super(GCNClassifier, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.eff = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.eff.classifier = nn.Identity()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()
        self.gcn = GCNClassifier(input_dim=2304)  # 1536(Eff) + 768(ViT)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        b = x.size(0)
        eff_feat = self.eff(x)
        vit_feat = self.vit(x)
        all_feat = torch.cat([eff_feat, vit_feat], dim=1)

        edge_index = pyg_utils.dense_to_sparse(torch.ones(b, b))[0].to(x.device)
        g = GeoData(x=all_feat, edge_index=edge_index)
        gcn_out = self.gcn(g.x, g.edge_index)
        out = self.fc(gcn_out)
        return out
