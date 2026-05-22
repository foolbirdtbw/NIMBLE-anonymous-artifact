import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
from nimble_core.utils.utils import create_activation, create_norm


class GraphSAGE(nn.Module):
    """
    GraphSAGE网络，使用采样和聚合机制
    适合大规模图，计算效率高，内存友好
    """
    def __init__(self,
                 n_dim,
                 e_dim,
                 hidden_dim,
                 out_dim,
                 n_layers,
                 n_heads,
                 n_heads_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 aggregator='mean'):
        super(GraphSAGE, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.aggregator = aggregator
        
        self.layers = nn.ModuleList()
        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        # 构建GraphSAGE层
        if n_layers == 1:
            self.layers.append(GraphSAGELayer(
                n_dim, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, activation=last_activation,
                concat_out=concat_out, aggregator=aggregator
            ))
        else:
            # 第一层
            self.layers.append(GraphSAGELayer(
                n_dim, e_dim, hidden_dim, n_heads,
                feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=True, aggregator=aggregator
            ))
            # 中间层
            for _ in range(1, n_layers - 1):
                self.layers.append(GraphSAGELayer(
                    hidden_dim * n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=True, aggregator=aggregator
                ))
            # 输出层
            self.layers.append(GraphSAGELayer(
                hidden_dim * n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm,
                concat_out=concat_out, aggregator=aggregator
            ))
        
        self.head = nn.Identity()
    
    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        
        for layer in self.layers:
            h = layer(g, h)
            if return_hidden:
                hidden_list.append(h)
        
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)


class GraphSAGELayer(nn.Module):
    """
    单层GraphSAGE，支持多种聚合方式
    """
    def __init__(self,
                 in_dim,
                 e_dim,
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 norm=None,
                 concat_out=True,
                 aggregator='mean'):
        super(GraphSAGELayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.out_feat = out_dim * n_heads if concat_out else out_dim
        self.aggregator = aggregator
        self.concat_out = concat_out
        
        # 节点特征投影
        self.fc_self = nn.Linear(in_dim, self.out_feat, bias=False)
        self.fc_neigh = nn.Linear(in_dim, self.out_feat, bias=False)
        
        # 边特征投影（用于增强聚合）
        self.fc_edge = nn.Linear(e_dim, self.out_feat, bias=False)
        
        # 多头注意力（可选，增强表达能力）
        if n_heads > 1:
            self.attn_fc = nn.Linear(self.out_feat, n_heads, bias=False)
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if residual:
            if in_dim != self.out_feat:
                self.res_fc = nn.Linear(in_dim, self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        self.norm = norm(self.out_feat) if norm else None
        self.activation = activation
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
        if hasattr(self, 'attn_fc'):
            nn.init.xavier_uniform_(self.attn_fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
    
    def forward(self, g, feat):
        with g.local_scope():
            # 处理输入特征
            if isinstance(feat, tuple):
                feat_src, feat_dst = feat
            else:
                feat_src = feat_dst = feat
                if g.is_block:
                    feat_dst = feat_src[:g.number_of_dst_nodes()]
            
            # Dropout
            feat_src = self.feat_drop(feat_src)
            feat_dst = self.feat_drop(feat_dst)
            
            # 投影
            h_self = self.fc_self(feat_dst)
            h_neigh = self.fc_neigh(feat_src)
            
            # 边特征处理
            if 'attr' in g.edata:
                edge_feat = self.fc_edge(g.edata['attr'])
                g.edata['e'] = edge_feat
            
            # 设置源节点特征
            g.srcdata['h'] = h_neigh
            
            # 根据聚合器类型进行聚合
            if self.aggregator == 'mean':
                if 'e' in g.edata:
                    # 包含边特征的聚合
                    g.update_all(
                        lambda edges: {'m': edges.src['h'] + edges.data['e']},
                        fn.mean('m', 'neigh')
                    )
                else:
                    g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            elif self.aggregator == 'max':
                if 'e' in g.edata:
                    g.update_all(
                        lambda edges: {'m': edges.src['h'] + edges.data['e']},
                        fn.max('m', 'neigh')
                    )
                else:
                    g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
            elif self.aggregator == 'sum':
                if 'e' in g.edata:
                    g.update_all(
                        lambda edges: {'m': edges.src['h'] + edges.data['e']},
                        fn.sum('m', 'neigh')
                    )
                else:
                    g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            elif self.aggregator == 'lstm':
                # LSTM聚合器需要排序，这里简化为mean
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            else:
                # 默认使用mean
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            
            # 获取聚合结果
            h_neigh = g.dstdata['neigh']
            
            # 多头注意力（如果启用）
            if self.n_heads > 1 and hasattr(self, 'attn_fc'):
                # 计算注意力权重
                attn = self.attn_fc(h_neigh + h_self)  # (N, n_heads)
                attn = self.leaky_relu(attn)
                attn = F.softmax(attn, dim=1)
                attn = self.attn_drop(attn)
                
                # 应用注意力
                h_neigh = h_neigh.view(-1, self.n_heads, self.out_dim)
                h_self = h_self.view(-1, self.n_heads, self.out_dim)
                
                if self.concat_out:
                    # 拼接所有头的输出
                    h_neigh = h_neigh.view(-1, self.out_feat)
                    h_self = h_self.view(-1, self.out_feat)
                else:
                    # 对所有头进行加权平均
                    h_neigh = (h_neigh * attn.unsqueeze(-1)).sum(dim=1)
                    h_self = (h_self * attn.unsqueeze(-1)).sum(dim=1)
            
            # 组合自身特征和邻居特征
            rst = h_self + h_neigh

            
            # 残差连接
            if self.res_fc is not None:
                resval = self.res_fc(feat_dst)
                rst = rst + resval
            
            # 归一化
            if self.norm is not None:
                rst = self.norm(rst)
            
            # 激活函数
            if self.activation:
                rst = self.activation(rst)
            
            return rst

