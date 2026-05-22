"""
支持消融实验的模型变体

包含以下变体：
1. Full: GraphSAGE + Denoising
2. No Denoising: GraphSAGE only
3. No GraphSAGE: GCN + Denoising
4. Baseline: GCN only
"""

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
    """根据args构建相应的模型变体"""
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    noise_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim
    
    # 从args中获取消融实验参数
    use_graphsage = getattr(args, 'use_graphsage', True)
    use_denoising = getattr(args, 'use_denoising', True)
    use_edge_recon = getattr(args, 'use_edge_recon', True)  # 新增：边重建开关
    graphsage_rep_mode = getattr(args, 'graphsage_rep_mode', 'concat')  # 'concat' 或 'last'
    aggregator = getattr(args, 'aggregator', 'mean')  # 新增：聚合器类型

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
        norm='BatchNorm',
        loss_fn='sce',
        alpha_l=alpha_l,
        use_graphsage=use_graphsage,
        use_denoising=use_denoising,
        use_edge_recon=use_edge_recon,
        graphsage_rep_mode=graphsage_rep_mode,
        aggregator=aggregator,
    )
    return model


class SimpleGCNEncoder(nn.Module):
    """简单的GCN编码器，用于消融实验"""
    
    def __init__(self, n_dim, e_dim, hidden_dim, out_dim, n_layers, activation="prelu", 
                 feat_drop=0.1, negative_slope=0.1, norm=None):
        super(SimpleGCNEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # 输入层
        self.fc_in = nn.Linear(n_dim, hidden_dim)
        
        # 隐藏层 - 每层都输出 hidden_dim
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 激活函数和正则化
        if activation == "prelu":
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.norm = norm
        
    def forward(self, g, x, return_hidden=False):
        """
        前向传播
        
        Args:
            g: DGL图
            x: 节点特征
            return_hidden: 是否返回所有隐藏层输出
            
        Returns:
            输出表示，如果return_hidden=True则返回(输出, 隐藏层列表)
        """
        all_hidden = []
        
        # 输入层
        h = self.fc_in(x)
        h = self.activation(h)
        h = self.feat_drop(h)
        all_hidden.append(h)
        
        # 隐藏层
        for layer in self.layers:
            h = layer(h)
            h = self.activation(h)
            h = self.feat_drop(h)
            all_hidden.append(h)
        
        # 最后一层输出也是 hidden_dim，不需要额外的 fc_out
        if return_hidden:
            return h, all_hidden
        return h


class GMAEModel(nn.Module):
    """支持消融实验的GMAE模型"""
    
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, noise_rate=0.5, 
                 noise_std=0.1, loss_fn="sce", alpha_l=2, 
                 use_graphsage=True, use_denoising=True, use_edge_recon=True,
                 graphsage_rep_mode: str = 'concat', aggregator: str = 'mean'): 
        super(GMAEModel, self).__init__()
        
        self._noise_rate = noise_rate if use_denoising else 0.0
        self._noise_std = noise_std if use_denoising else 0.0
        self._output_hidden_size = hidden_dim
        self._use_graphsage = use_graphsage
        self._use_denoising = use_denoising
        self._use_edge_recon = use_edge_recon  # 新增：边重建开关
        self._graphsage_rep_mode = graphsage_rep_mode
        self._aggregator = aggregator  # 新增：聚合器类型
        
        self.recon_loss = nn.BCELoss(reduction='mean')

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        # edge_recon_fc 的输入维度是两个节点表示的拼接
        # GraphSAGE: 根据表示模式选择 last 或 concat
        #  - last: hidden_dim * 2
        #  - concat: (hidden_dim * n_layers) * 2
        # SimpleGCN: (hidden_dim * n_layers) * 2
        if use_graphsage:
            if graphsage_rep_mode == 'concat':
                edge_fc_in_dim = hidden_dim * n_layers * 2
            else:
                edge_fc_in_dim = hidden_dim * 2
        else:
            edge_fc_in_dim = hidden_dim * n_layers * 2
        
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(edge_fc_in_dim, hidden_dim),
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

        # 构建编码器
        if use_graphsage:
            # 使用GraphSAGE编码器
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
                aggregator=aggregator,  # 传递聚合器参数
            )
            # GraphSAGE 输出维度取决于表示模式
            encoder_out_dim = hidden_dim * n_layers if graphsage_rep_mode == 'concat' else hidden_dim
        else:
            # 使用简单GCN编码器
            self.encoder = SimpleGCNEncoder(
                n_dim=n_dim,
                e_dim=e_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                n_layers=n_layers,
                activation=activation,
                feat_drop=feat_drop,
                negative_slope=negative_slope,
                norm=create_norm(norm)
            )
            # SimpleGCN 输出维度是 hidden_dim * n_layers (所有隐藏层的拼接)
            encoder_out_dim = hidden_dim * n_layers

        # 构建解码器
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
        )

        # encoder_to_decoder 的输入维度根据编码器类型调整
        self.encoder_to_decoder = nn.Linear(encoder_out_dim, dec_in_dim, bias=False)

        # 设置损失函数
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

    def encoding_denoising(self, g, noise_rate=0.3, noise_std=0.1):
        """
        编码阶段的去噪处理
        
        Args:
            g: 输入图
            noise_rate: 噪声比例
            noise_std: 噪声标准差
            
        Returns:
            (处理后的图, (噪声节点, 保留节点))
        """
        new_g = g.clone()
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=g.device)

        # 随机选择噪声节点
        num_noise_nodes = int(noise_rate * num_nodes)
        noise_nodes = perm[: num_noise_nodes]
        keep_nodes = perm[num_noise_nodes:]

        # 添加高斯噪声到选定节点的特征
        noise = torch.randn_like(new_g.ndata["attr"][noise_nodes]) * noise_std
        new_g.ndata["attr"][noise_nodes] = new_g.ndata["attr"][noise_nodes] + noise

        return new_g, (noise_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        """
        计算损失函数
        
        包含两部分：
        1. 特征重建损失（仅在使用Denoising时）
        2. 结构重建损失
        """
        # 特征重建损失 (仅在使用Denoising时)
        if self._use_denoising:
            pre_use_g, (noise_nodes, keep_nodes) = self.encoding_denoising(
                g, self._noise_rate, self._noise_std
            )
        else:
            pre_use_g = g.clone()
            num_nodes = g.num_nodes()
            noise_nodes = torch.tensor([], dtype=torch.long, device=g.device)
            keep_nodes = torch.arange(num_nodes, device=g.device)
        
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g
        
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        
        # 选择用于后续模块的表示：
        # - GraphSAGE: 根据 _graphsage_rep_mode 选择 'concat' 或 'last'
        # - SimpleGCN: 使用所有隐藏层拼接 (维度 hidden_dim * n_layers)
        if self._use_graphsage:
            if getattr(self, '_graphsage_rep_mode', 'concat') == 'concat':
                enc_input = torch.cat(all_hidden, dim=1)
            else:
                enc_input = enc_rep
        else:
            enc_input = torch.cat(all_hidden, dim=1)
        
        rep = self.encoder_to_decoder(enc_input)

        recon = self.decoder(pre_use_g, rep)
        
        # 特征重建损失
        if self._use_denoising and len(noise_nodes) > 0:
            x_init = g.ndata['attr'][noise_nodes]  # 原始干净特征
            x_rec = recon[noise_nodes]  # 重建特征
            loss = self.criterion(x_rec, x_init)
        else:
            # 如果不使用Denoising，对所有节点计算重建损失
            loss = self.criterion(recon, g.ndata['attr'])

        # 结构重建损失（仅在启用时计算）
        if self._use_edge_recon:
            threshold = min(10000, g.num_nodes())

            negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
            positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
            positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])
            
            # 使用节点表示进行边重建（GraphSAGE: 最后一层；SimpleGCN: 所有层拼接）
            node_repr = enc_input
            sample_src = node_repr[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
            sample_dst = node_repr[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)
            y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)
            y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
                g.device)
            loss += self.recon_loss(y_pred, y)
        
        return loss

    def embed(self, g):
        """获取图的嵌入表示（与训练时的表示逻辑对齐）"""
        x = g.ndata['attr'].to(g.device)
        rep, all_hidden = self.encoder(g, x, return_hidden=True)
        if self._use_graphsage:
            if getattr(self, '_graphsage_rep_mode', 'concat') == 'concat':
                node_repr = torch.cat(all_hidden, dim=1)
            else:
                node_repr = rep
        else:
            node_repr = torch.cat(all_hidden, dim=1)
        return node_repr

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

