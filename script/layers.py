import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData

import numpy as np
from math import pi as PI


class IAGMN_Layer(nn.Module): # Invariant Attention Graph Matching Network

    def __init__(
            self,
            args
            ):
        super().__init__()
        
        self.args = args
        self.hidden_feature = args.num_hidden_feature
        if args.conditional:
            self.cond_feature = args.num_cond_feature
            self.cond_mlp = nn.Linear(
                    self.hidden_feature + self.cond_feature,
                    self.hidden_feature, 
                    bias=False
            )
        else:
            self.cond_feature = 0
        
        self.inter_mlp = \
                nn.Sequential(
                        nn.Linear(self.args.dist_one_hot_param1[-1] + \
                                self.hidden_feature * 2, self.hidden_feature),
                        nn.SiLU(),
                        nn.Linear(self.hidden_feature, self.hidden_feature),
                )

        self.intra_mlp = \
                nn.Sequential(
                        nn.Linear(self.args.dist_one_hot_param1[-1] + \
                                self.hidden_feature * 2, self.hidden_feature),
                        nn.SiLU(),
                        nn.Linear(self.hidden_feature, self.hidden_feature),
                )
        
        self.inter_intra_gate = \
                nn.Sequential(
                        nn.Linear(2 * self.hidden_feature, 1),
                        nn.Sigmoid()
                )
        
        self.l_node_update = \
                nn.GRUCell(
                        self.hidden_feature,
                        self.hidden_feature
                )
        self.p_node_update = \
                nn.GRUCell(
                        self.hidden_feature,
                        self.hidden_feature
                )

        self.radial_cutoff = 10. # TODO
        #self.radial_cutoff = args.dist_one_hot_param1[1]


    def compute_cross_attn(
            self,
            query,
            key,
            value,
            batch_index
            ):
        attn = torch.mm(query, torch.transpose(key, 1, 0))
        if batch_index is None:
            attn_softmax = F.softmax(attn, -1)
        else:
            attn_softmax = scatter_softmax(attn, batch_index, dim=0)
        cross_attn = torch.mm(attn_softmax, value)
        return cross_attn

    def filter_function(
            self,
            edge_weight
            ):
        return 0.5 * (torch.cos(edge_weight * PI / self.radial_cutoff) + 1.)

    def forward(
            self,
            data,
            cond=None
            ):

        if self.args.conditional:
            data["pocket"].h_cond = torch.cat([data["pocket"].h, cond], -1)
            data["pocket"].h = self.cond_mlp(data["pocket"].h_cond)
        
        e_ll_src, e_ll_tar = data["l2l"].edge_index
        e_pp_src, e_pp_tar = data["p2p"].edge_index
        e_pl_src, e_pl_tar = data["p2l"].edge_index
        
        h_ll_i, h_ll_j = data["ligand"].h[e_ll_src], data["ligand"].h[e_ll_tar]
        h_pp_i, h_pp_j = data["pocket"].h[e_pp_src], data["pocket"].h[e_pp_tar]
        h_pl_i, h_pl_j = data["pocket"].h[e_pl_src], data["ligand"].h[e_pl_tar]
        h_lp_i, h_lp_j = data["ligand"].h[e_pl_tar], data["pocket"].h[e_pl_src]

        ll_in = torch.cat([h_ll_i, data["l2l"].edge_attr, h_ll_j], -1)
        pp_in = torch.cat([h_pp_i, data["p2p"].edge_attr, h_pp_j], -1)
        pl_in = torch.cat([h_pl_i, data["p2l"].edge_attr, h_pl_j], -1)
        lp_in = torch.cat([h_lp_i, data["p2l"].edge_attr, h_lp_j], -1)

        # Providing distance-based attention
        filter_ll = self.filter_function(data["l2l"].edge_weight).unsqueeze(-1)
        filter_pp = self.filter_function(data["p2p"].edge_weight).unsqueeze(-1)
        filter_pl = self.filter_function(data["p2l"].edge_weight).unsqueeze(-1)

        msg_ll = self.inter_mlp(ll_in) * filter_ll # [B, num_ll_edge, num_hidden]
        msg_pp = self.inter_mlp(pp_in) * filter_pp
        msg_pl = self.intra_mlp(pl_in) * filter_pl
        msg_lp = self.intra_mlp(lp_in) * filter_pl

        aggr_msg_ll = scatter_mean(msg_ll, e_ll_tar, dim=0, \
                        out=data["ligand"].h.new_zeros(data["ligand"].h.shape))
        aggr_msg_pp = scatter_mean(msg_pp, e_pp_tar, dim=0, \
                        out=data["pocket"].h.new_zeros(data["pocket"].h.shape))
        aggr_msg_pl = scatter_mean(msg_pl, e_pl_tar, dim=0, \
                        out=data["ligand"].h.new_zeros(data["ligand"].h.shape))
        aggr_msg_lp = scatter_mean(msg_lp, e_pl_src, dim=0, \
                        out=data["pocket"].h.new_zeros(data["pocket"].h.shape))

        l_msg_cat = torch.cat([aggr_msg_ll, aggr_msg_pl], -1)
        p_msg_cat = torch.cat([aggr_msg_pp, aggr_msg_lp], -1)

        l_gate = self.inter_intra_gate(l_msg_cat)
        p_gate = self.inter_intra_gate(p_msg_cat)

        l_msg = l_gate * aggr_msg_ll + (1 - l_gate) * aggr_msg_pl
        p_msg = p_gate * aggr_msg_pp + (1 - p_gate) * aggr_msg_lp

        new_h_l = self.l_node_update(data["ligand"].h, l_msg)
        new_h_p = self.p_node_update(data["pocket"].h, p_msg)

        data["ligand"].h = new_h_l
        data["pocket"].h = new_h_p
        return data


class EGCL(nn.Module):

    def __init__(
            self,
            args
            ):
        super().__init__()
        
        self.args = args
        
        self.distance_expand = SoftOneHot(*args.dist_one_hot_param1, \
                gamma=args.gamma1)
        self.num_edge_feature = args.dist_one_hot_param1[-1]
        self.num_node_feature = args.num_pocket_atom_feature
        self.num_hidden_feature = args.num_hidden_feature

        self.edge_mlp = nn.Sequential(
                nn.Linear(self.num_node_feature * 2 + self.num_edge_feature, \
                        self.num_hidden_feature),
                nn.SiLU(),
                nn.Linear(self.num_hidden_feature, self.num_hidden_feature)
        )
        self.node_mlp = nn.Sequential(
                nn.Linear(self.num_hidden_feature, self.num_hidden_feature),
                nn.SiLU(),
                nn.Linear(self.num_hidden_feature, self.num_node_feature)
        )
        self.coord_mlp = nn.Sequential(
                nn.Linear(self.num_hidden_feature, self.num_hidden_feature),
                nn.SiLU(),
                nn.Linear(self.num_hidden_feature, 1)
        )

    def forward(
            self,
            h,
            x,
            edge_index
            ):
        src, tar = edge_index
        h_i, h_j = h[src], h[tar]
        x_i, x_j = x[src], x[tar]
        d_ij = (x_i - x_j).norm(dim=-1)
        e_ij = self.distance_expand(d_ij)[0]

        edge_msg = self.edge_mlp(torch.cat([h_i, h_j, e_ij], -1))
        coord_msg = self.coord_mlp(edge_msg)
        div = d_ij.unsqueeze(-1) + 1

        x_prime = x + scatter_add((x_i - x_j) * coord_msg / div, tar, dim=0)
        h_prime = h + scatter_mean(self.node_mlp(edge_msg), tar, \
                dim=0, out=h.new_zeros(h.shape))
        return h_prime, x_prime


class ShiftedSoftplus(nn.Module):
    r"""
    Shited-softplus activated function
    """

    def __init__(
            self,
            ):
        super().__init__()

    def forward(
            self,
            input
            ):
        return F.softplus(input) - np.log(2.0)


class SoftOneHot(nn.Module):
    r"""
    Gaussian expansion of given distance 

    Args:
        x_min (float)
        x_max (float)
        steps (int)
        gamma (float)
        normalize (bool)
    """

    def __init__(
            self,
            x_min,
            x_max,
            steps,
            gamma=10.0,
            normalize=False
            ):
        super().__init__()

        assert x_min < x_max, "x_min is larger than x_max"
        self.x_min = x_min
        self.x_max = x_max
        self.steps = int(steps)
        self.center = torch.Tensor([x_min * (steps - i - 1) / (steps - 1) + \
                           x_max * i / (steps - 1) for i in range(steps)])

        self.gamma = gamma
        self.normalize = normalize

    def forward(
            self,
            x
            ):
        r"""
        Args:
            x (torch.Tensor): distance matrix with dimension [A B]

        Returns:
            x_embed (torch.Tensor): Gaussian expanded distance matrix with \
                    dimension [A B steps]
        """
        x_repeat = x.unsqueeze(-1).repeat(1, 1, self.steps)
        c_repeat = self.center.unsqueeze(0).unsqueeze(0)
        c_repeat = c_repeat.to(x_repeat.device)
        x_embed = torch.exp(-self.gamma * torch.pow(x_repeat - c_repeat, 2))
        if self.normalize:
            x_embed /= torch.sum(x_embed, -1, keepdim=True)
        return x_embed


class HardOneHot(nn.Module):

    def __init__(
            self,
            x_min,
            x_max,
            steps,
            ):
        super().__init__()
        assert x_min < x_max, "x_min is larger than x_max"
        self.x_min = x_min
        self.x_max = x_max
        self.steps = int(steps)
        self.eye = torch.eye(self.steps)

    def forward(
            self,
            x
            ):
        x = x - self.x_min
        x = x * (self.steps - 1) / (self.x_max - self.x_min)
        x = x.clamp(0, self.steps - 1).long()
        c = self.eye[x] 
        return c


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.num_hidden_feature = 10
    args.conditional = False
    args.dist_one_hot_param1 = [0, 10, 20]

    layer = IAGMN_Layer(args)

