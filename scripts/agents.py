import pandas as pd
import numpy as np
import yaml
import time
from typing import Any, Set, Tuple, List, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import download_url, extract_zip, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.transforms as T
from torch_geometric import EdgeIndex
from torch_geometric.utils import add_self_loops, spmm, is_sparse
from torch_geometric.typing import Adj, OptPairTensor, SparseTensor

latent1 = 64
latent2 = 16
latent3 = 8
encoder_node1_channels = [111, 132, latent1] # thief
encoder_node2_channels = [12, 32, latent2] # slot
encoder_node3_channels = [4, 16, latent3] # heist

message_hidden_node1_size = 128
message_hidden_node2_size = 32
message_hidden_node3_size = 16

decoder_node1_channels = [latent1, 32, 8]
decoder_node2_channels = [latent2, 12, 8]
# decoder_node3_channels = [latent3, 6, 4]

class GNN(nn.Module):
    def __init__(self, encoder_node1_channels: list = encoder_node1_channels,
                encoder_node2_channels: list = encoder_node2_channels,
                encoder_node3_channels: list = encoder_node3_channels,
                message_hidden_node1_size: int = message_hidden_node1_size,
                message_hidden_node2_size: int = message_hidden_node2_size,
                message_hidden_node3_size: int = message_hidden_node3_size,
                decoder_node1_channels: List = decoder_node1_channels,
                decoder_node2_channels: List = decoder_node2_channels,
                out_channels = 1,
                n_passes = 3):
        super(GNN, self).__init__()
        
        # Define encoders
        _, _, enc1_out = encoder_node1_channels
        _, _, enc2_out = encoder_node2_channels
        _, _, enc3_out = encoder_node3_channels
        
        self.encoder1 = MLP(encoder_node1_channels)
        self.encoder2 = MLP(encoder_node2_channels)
        self.encoder3 = MLP(encoder_node3_channels)

        # Define message passing: 1 for each index
        # Take in latent encoding out size
        self.n_passes = n_passes
        self.message_a = CustomMessagePassing(enc1_out, enc2_out, message_hidden_node2_size)
        self.message_b = CustomMessagePassing(enc1_out, enc2_out, message_hidden_node2_size)
        self.message_c = CustomMessagePassing(enc2_out, enc3_out, message_hidden_node3_size)
        self.message_d = CustomMessagePassing(enc3_out, enc3_out, message_hidden_node3_size)

        # Reverse message passing layers
        self.message_rev_a = CustomMessagePassing(enc2_out, enc1_out, message_hidden_node1_size)
        self.message_rev_c = CustomMessagePassing(enc3_out, enc2_out, message_hidden_node2_size)

        # Define decoders
        _, _, dec1_out = decoder_node1_channels
        _, _, dec2_out = decoder_node2_channels
        
        self.decoder1 = MLP(decoder_node1_channels)
        self.decoder2 = MLP(decoder_node2_channels)

        # Define final transformation layer
        self.linear = nn.Linear(dec1_out + dec2_out, out_channels)

    def forward_helper(self, data):
        # Encode node features
        node1_x = self.encoder1(data['thief'].x)
        node2_x = self.encoder2(data['slot'].x)
        node3_x = self.encoder3(data['heist'].x)

        # Preserve original embeddings
        node1_x_original = node1_x
        node2_x_original = node2_x
        # node3_x_original = node3_x

        # print("Encoded: ", node1_x.shape, node2_x.shape, node3_x.shape)

        # Message passing
        for _ in range(1):
            node3_x = self.message_d(data['heist1','conflicts','heist2'].edge_index, (node3_x, node3_x))
            node3_x = self.message_rev_c(data['slot','on','heist'].edge_index, (node2_x, node3_x))
            node2_x = self.message_c(data['heist','rev_on','slot'].edge_index, (node3_x, node2_x))
            if data['slot','rev_assigned','thief'].edge_index.shape[1] != 0:
                node1_x = self.message_b(data['slot','rev_assigned','thief'].edge_index, (node2_x, node1_x))
            node1_x = self.message_a(data['slot','rev_possible','thief'].edge_index, (node2_x, node1_x))
            node2_x = self.message_rev_a(data['thief','possible','slot'].edge_index, (node1_x, node2_x))

        # Update original embeddings
        node1_x += node1_x_original
        node2_x += node2_x_original
        
        # Decode
        node1_x = self.decoder1(node1_x)
        node2_x = self.decoder2(node2_x)

        # print("Decoded: ", node1_x.shape, node2_x.shape, node3_x.shape)

        return node1_x, node2_x
    def forward(self, data):
        node1_x, node2_x = self.forward_helper(data)

        # Final transformation
        src, dst = data['thief','possible','slot'].edge_index
        out = (node1_x[src] * node2_x[dst]).sum(dim=-1)
        out = nn.Softmax()(out)
        # out = self.linear(torch.cat([node1_x[src], node2_x[dst]], dim=1))

        return out

class MLP(nn.Module):
    def __init__(self, channels: List) -> None: # [in, hidden, out]
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(channels[0], channels[1]),
            nn.ReLU(),
            nn.Linear(channels[1], channels[2])
        )
    def forward(self, x):
        return self.linear(x)

class CustomMessagePassing(MessagePassing):
    def __init__(self,
                 central_size: int,
                 neighbor_size: int,
                 message_hidden_size: int):
        super(CustomMessagePassing, self).__init__(aggr='mean')
        
        # Define linear transformations
        self.neighbor_linear = nn.Linear(neighbor_size, message_hidden_size) # compress neighbors
        self.update_linear = nn.Linear(message_hidden_size, central_size)    # transform aggregated neighbors into central node size

    def forward(self, edge_index, x): # x = (neighbor, central)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return self.neighbor_linear(x_j)
    
    def update(self, aggr_out): # NOTE: was prev aggregate, so kept returning message of size [164,8]
        return self.update_linear(aggr_out)
    

critic_decoder_node1_channels = [latent1, 32, 1]
critic_decoder_node2_channels = [latent2, 12, 1]
# critic_decoder_node3_channels = [latent3, 6, 1]

class Critic(GNN):
    def __init__(self, encoder_node1_channels: list = encoder_node1_channels,
                encoder_node2_channels: list = encoder_node2_channels,
                encoder_node3_channels: list = encoder_node3_channels,
                message_hidden_node1_size: int = message_hidden_node1_size,
                message_hidden_node2_size: int = message_hidden_node2_size,
                message_hidden_node3_size: int = message_hidden_node3_size,
                decoder_node1_channels: List = critic_decoder_node1_channels,
                decoder_node2_channels: List = critic_decoder_node2_channels,
                out_channels = 1,
                n_passes = 3):
        super().__init__(encoder_node1_channels, encoder_node2_channels, encoder_node3_channels,
                    message_hidden_node1_size, message_hidden_node2_size, message_hidden_node3_size,
                    decoder_node1_channels, decoder_node2_channels,
                    out_channels, n_passes)

    def forward(self, data):
        node1_x, node2_x = self.forward_helper(data)
        out = node1_x.mean() + node2_x.mean()

        # print("Decoded: ", node1_x.shape, node2_x.shape, node3_x.shape)

        return out