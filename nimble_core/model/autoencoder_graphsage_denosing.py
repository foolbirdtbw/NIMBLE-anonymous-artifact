from .gat import GAT
from .gin import GIN
from .graphsage import GraphSAGE
from .gcnii import GCNII
from nimble_core.utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random


def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    noise_rate = args.mask_rate  # 重用mask_rate参数作为noise_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim
    lambda_weight = getattr(args, 'lambda_weight', 1.0)  # Default to 1.0 if not specified
    noise_std = getattr(args, 'noise_std', 0.1)
    bounded_noise = getattr(args, 'bounded_noise', False)
    renorm_noise = getattr(args, 'renorm_noise', False)
    aggregator = getattr(args, 'aggregator', 'mean')

    model = GMAEModel(
        n_dim=n_dim,
        e_dim=e_dim,
        hidden_dim=num_hidden,
        n_layers=num_layers,
        n_heads=4,
        activation="prelu",
        feat_drop=0.1,
        negative_slope=negative_slope,
        residual=True,
        noise_rate=noise_rate,
        noise_std=noise_std,
        bounded_noise=bounded_noise,
        renorm_noise=renorm_noise,
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l,
        lambda_weight=lambda_weight,
        aggregator=aggregator,
    )
    return model


class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, noise_rate=0.5,
                 noise_std=0.1, bounded_noise=False, renorm_noise=False,
                 loss_fn="sce", alpha_l=2, lambda_weight=1.0,
                 aggregator='mean'):
        super(GMAEModel, self).__init__()
        self._noise_rate = noise_rate
        self._noise_std = noise_std
        self._bounded_noise = bounded_noise
        self._renorm_noise = renorm_noise
        self._output_hidden_size = hidden_dim
        self._lambda_weight = lambda_weight
        self._aggregator = aggregator
        self.recon_loss = nn.BCELoss(reduction='mean')

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant_(m.bias, 0)

        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        dec_in_dim = hidden_dim
        dec_num_hidden = hidden_dim

        # build encoder
        self.encoder = GraphSAGE(
            n_dim=n_dim,
            e_dim=e_dim,
            hidden_dim=enc_num_hidden,
            out_dim=enc_num_hidden,
            n_layers=n_layers,
            n_heads=enc_nhead,
            n_heads_out=enc_nhead,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=True,
            aggregator=aggregator,
        )

        # build decoder for attribute prediction
        self.decoder = GraphSAGE(
            n_dim=dec_in_dim,
            e_dim=e_dim,
            hidden_dim=dec_num_hidden,
            out_dim=n_dim,
            n_layers=1,
            n_heads=n_heads,
            n_heads_out=1,
            concat_out=True,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=0.0,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=False,
            aggregator=aggregator,
        )


        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def apply_gaussian_corruption(self, x, noise_std):
        noisy_x = x + torch.randn_like(x) * noise_std
        if self._bounded_noise:
            noisy_x = noisy_x.clamp(0.0, 1.0)
        if self._renorm_noise:
            row_sum = noisy_x.sum(dim=-1, keepdim=True)
            noisy_x = torch.where(row_sum > 1e-6, noisy_x / row_sum.clamp_min(1e-6), x)
        return noisy_x

    def encoding_denoising(self, g, noise_rate=0.3, noise_std=0.1):
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # random denoising - add gaussian noise to selected nodes
        num_noise_nodes = int(noise_rate * num_nodes)
        noise_nodes = perm[: num_noise_nodes]
        keep_nodes = perm[num_noise_nodes:]

        new_g.ndata["attr"][noise_nodes] = self.apply_gaussian_corruption(
            new_g.ndata["attr"][noise_nodes],
            noise_std,
        )

        return new_g, (noise_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        # Feature Reconstruction with Denoising
        pre_use_g, (noise_nodes, keep_nodes) = self.encoding_denoising(g, self._noise_rate, self._noise_std)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][noise_nodes]  # Original clean features
        x_rec = recon[noise_nodes]  # Reconstructed features
        loss = self.criterion(x_rec, x_init)

        # Structural Reconstruction  
        # Reduced threshold to prevent OOM on large graphs
        threshold = min(5000, g.num_nodes())  # Reduced from 10000 to 5000

        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), min(threshold, g.number_of_edges()))
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        loss += self._lambda_weight * self.recon_loss(y_pred, y)
        return loss

    def embed(self, g):
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
