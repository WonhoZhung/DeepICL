import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from layers import EGCL, ConstrainedCrossAttention, E3II_Layer


class DeepICL(nn.Module):
    r""" """

    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args

        self.embedding = Embedding(args)
        self.vae = VariationalEncoder(args)

        if args.conditional:
            self.conditional = True
            self.num_cond_feature = args.num_cond_feature
        else:
            self.conditional = False
            self.num_cond_feature = 0

        if args.ssl:
            self.ssl_model = SSLModel(args)

        self.latent_mlp = nn.Linear(
            args.num_hidden_feature + args.num_latent_feature + self.num_cond_feature,
            args.num_hidden_feature,
        )

        self.next_type_ll = NextType(args, self.embedding.l_node_emb)
        self.next_type_lp = NextType(args, self.embedding.l_node_emb)
        self.next_dist_ll = NextDist(args, self.embedding.l_node_emb)
        self.next_dist_lp = NextDist(args, self.embedding.l_node_emb)

        self.loss_fn = nn.KLDivLoss(reduction="none")
        if args.ssl:
            self.ssl_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        data_dict,
    ):
        r""" """

        whole, traj, _ = data_dict.values()

        if self.args.conditional:
            whole_cond = whole.pocket_prop
            traj_cond = traj.pocket_prop
        else:
            whole_cond = None
            traj_cond = None

        # Embed(propagate) whole graph
        self.embedding(whole, cond=whole_cond)

        # Sample latent vector and calculate vae loss
        latent, vae_loss = self.vae(whole)

        # Embed(propagate) unfinished graph
        self.embedding(traj, cond=traj_cond)

        # Concat latent vector with atom features
        if self.conditional:
            traj["pocket"].h = self.latent_mlp(
                torch.cat(
                    [
                        traj["pocket"].h,
                        traj_cond,
                        latent.repeat(traj["pocket"].h.shape[0], 1),
                    ],
                    -1,
                )
            )
        else:
            traj["pocket"].h = self.latent_mlp(
                torch.cat(
                    [traj["pocket"].h, latent.repeat(traj["pocket"].h.shape[0], 1)], -1
                )
            )

        # Predict p(Type|L) & p(Type|P)
        type_ll_pred = self.next_type_ll(traj, "ligand")
        type_lp_pred = self.next_type_lp(traj, "pocket")
        type_ll_loss = self.loss_fn(type_ll_pred, traj.type_output)
        type_lp_loss = self.loss_fn(type_lp_pred, traj.type_output)

        # Predict p(Position|L) & p(Position|P)
        dist_ll_pred = self.next_dist_ll(traj, traj.type_output, "ligand")
        dist_lp_pred = self.next_dist_lp(traj, traj.type_output, "pocket")
        ll_mask = (
            traj.mask[traj["ligand"].batch]
            .unsqueeze(-1)
            .repeat(1, self.args.dist_one_hot_param2[-1])
        )
        lp_mask = (
            traj.mask[traj["pocket"].batch]
            .unsqueeze(-1)
            .repeat(1, self.args.dist_one_hot_param2[-1])
        )
        dist_ll_loss = self.loss_fn(dist_ll_pred, traj.dist_ll_output) * ll_mask
        dist_lp_loss = self.loss_fn(dist_lp_pred, traj.dist_lp_output) * lp_mask

        dist_ll_loss = scatter_mean(dist_ll_loss, traj["ligand"].batch, 0)
        dist_lp_loss = scatter_mean(dist_lp_loss, traj["pocket"].batch, 0)

        vae_loss = self.args.vae_coeff * vae_loss.sum()  # KLDivLoss annealing
        type_ll_loss = type_ll_loss.mean(0).sum(0)  # Averaging in batch dimension
        type_lp_loss = type_lp_loss.mean(0).sum(0)  # Averaging in batch dimension
        dist_ll_loss = dist_ll_loss.sum() / traj.mask.sum()
        dist_lp_loss = dist_lp_loss.sum() / traj.mask.sum()

        type_loss = type_ll_loss + type_lp_loss
        dist_loss = dist_ll_loss + dist_lp_loss

        total_loss = vae_loss + type_loss + dist_loss

        if self.args.ssl:
            cond_pred = self.ssl_model(whole)
            ssl_loss = self.ssl_loss_fn(cond_pred, whole_cond.argmax(-1)).mean()
            total_loss += ssl_loss
            return total_loss, vae_loss, type_loss, dist_loss, ssl_loss

        return total_loss, vae_loss, type_loss, dist_loss, None


class Embedding(nn.Module):
    r"""
    Node embedding and propagation
    """

    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args

        self.l_node_emb = nn.Sequential(
            nn.Linear(args.num_ligand_atom_feature, args.num_hidden_feature),
        )
        self.p_node_emb = nn.Sequential(
            nn.Linear(args.num_pocket_atom_feature, args.num_hidden_feature),
        )
        self.emb_dict = {"ligand": self.l_node_emb, "pocket": self.p_node_emb}

        self.layers = nn.ModuleList([E3II_Layer(args) for _ in range(args.num_layers)])

    def forward(self, data, cond=None):
        data["ligand"].h = self.l_node_emb(data["ligand"].x)
        data["pocket"].h = self.p_node_emb(data["pocket"].x)

        for lay in self.layers:
            lay(data, cond=cond)

        return data


class NextType(nn.Module):
    r"""
    Predict p(Type)
    """

    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_ligand_atom_feature, args.num_hidden_feature, bias=False
            )

        self.act = nn.SiLU()
        self.last_act = None

        layers = []
        n_dims = list(
            np.linspace(args.num_hidden_feature, 1, args.num_dense_layers + 1).astype(
                int
            )
        )
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)

        self.dense = nn.Sequential(*layers)

        self.atom_type = torch.eye(args.num_ligand_atom_feature)
        self.atom_type = nn.Parameter(self.atom_type)
        self.atom_type.requires_grad = False

    def forward(self, data, key="ligand"):
        r"""
        Args:
            data (torch_geometric.data.HeteroDataBatch)

        Returns:
            next_type (torch.Tensor): [num_type]
        """

        embed_type = self.embedding(self.atom_type)  # [num_type, num_hidden]
        embed_type = embed_type.unsqueeze(0)  # [1, num_type, num_hidden]
        repr_type = data[key].h.unsqueeze(1)  # [N, 1, num_hidden]

        mul_type = embed_type * repr_type  # [N, num_type, num_hidden]
        dense_type = self.dense(mul_type).squeeze(-1)  # [N, num_type]

        batch = data[key].batch
        next_type = F.log_softmax(dense_type, dim=-1)  # [N, num_type]
        next_type_agg = scatter_add(next_type, batch, dim=0)  # [B, num_type]
        next_type_agg = next_type_agg - torch.logsumexp(
            next_type_agg, dim=-1, keepdim=True
        )

        return next_type_agg


class NextDist(nn.Module):
    r"""
    Predict p(Distance)
    """

    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_ligand_atom_feature, args.num_hidden_feature, bias=False
            )

        self.act = nn.SiLU()
        self.last_act = None

        layers = []
        n_dims = list(
            np.linspace(
                args.num_hidden_feature,
                args.dist_one_hot_param2[-1],
                args.num_dense_layers + 1,
            ).astype(int)
        )
        for n_in, n_out in zip(n_dims[:-1], n_dims[1:]):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(self.act)
        layers = layers[:-1]
        if self.last_act is not None:
            layers.append(self.last_act)

        self.dense = nn.Sequential(*layers)

        self.use_attention = False  # TODO
        if self.use_attention:
            # Constrained cross attention from E3Bind
            self.attn = ConstrainedCrossAttention(args)

    def forward(self, data, next_type, key):  # "ligand" or "pocket"
        batch = data[key].batch
        type_embed = self.embedding(next_type)[batch]  # [N, num_hidden]
        dist_embed = data[key].h * type_embed
        if self.use_attention:
            dist_embed, attn = self.attn(dist_embed, data[key].pos, batch)
        next_dist = F.log_softmax(self.dense(dist_embed), dim=-1)
        return next_dist


class VariationalEncoder(nn.Module):
    r""" """

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.mean = nn.Linear(args.num_hidden_feature, args.num_latent_feature)
        self.logvar = nn.Linear(args.num_hidden_feature, args.num_latent_feature)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device)
        return eps * std + mean

    def vae_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)

    def forward(
        self,
        data,
    ):
        h_cat = torch.cat([data["ligand"].h, data["pocket"].h], 0)  # [L+P F]
        readout = h_cat.mean(dim=0, keepdim=True)  # [1 F]
        mean = self.mean(readout)
        logvar = self.logvar(readout)
        latent = self.reparameterize(mean, logvar)  # [1 F']
        # l_latent, p_latent = latent[:data["ligand"].h.shape[0]], \
        #        latent[data["ligand"].h.shape[0]:]
        vae_loss = self.vae_loss(mean, logvar)
        return latent, vae_loss


class SSLModel(nn.Module):
    def __init__(self, args, embedding=None):
        super().__init__()

        self.args = args

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Linear(
                args.num_pocket_atom_feature, args.num_hidden_feature, bias=False
            )

        self.num_layers = args.num_dense_layers
        self.layers = nn.ModuleList([EGCL(args) for _ in range(self.num_layers)])
        self.fc_layer = nn.Linear(args.num_hidden_feature, args.num_cond_feature)

    def forward(self, data):
        edge_index_ = data["p2p"].edge_index.clone()
        h_ = data["pocket"].x.clone()
        x_ = data["pocket"].pos.clone()

        for layer in self.layers:
            h_, x_ = layer(h_, x_, edge_index_)

        y = self.fc_layer(h_)
        return y
