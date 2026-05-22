import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
from nimble_core.utils.utils import create_activation, create_norm


class GCNII(nn.Module):
    """
    GCNII网络，解决深度GCN的过平滑问题
    通过初始残差和恒等映射，支持更深的网络
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
                 alpha=0.1,
                 lamda=0.5):
        super(GCNII, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.concat_out = concat_out
        self.alpha = alpha  # 初始残差权重
        self.lamda = lamda  # 恒等映射权重
        
        self.layers = nn.ModuleList()
        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None
        
        # 输入投影
        self.input_proj = nn.Linear(n_dim, hidden_dim * n_heads)
        
        # 构建GCNII层
        if n_layers == 1:
            self.layers.append(GCNIILayer(
                hidden_dim * n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, activation=last_activation,
                concat_out=concat_out, alpha=alpha, lamda=lamda, layer=0
            ))
        else:
            # 第一层
            self.layers.append(GCNIILayer(
                hidden_dim * n_heads, e_dim, hidden_dim, n_heads,
                feat_drop, attn_drop, negative_slope,
                residual, create_activation(activation),
                norm=norm, concat_out=True, alpha=alpha, lamda=lamda, layer=0
            ))
            # 中间层
            for i in range(1, n_layers - 1):
                self.layers.append(GCNIILayer(
                    hidden_dim * n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_activation(activation),
                    norm=norm, concat_out=True, alpha=alpha, lamda=lamda, layer=i
                ))
            # 输出层
            self.layers.append(GCNIILayer(
                hidden_dim * n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm,
                concat_out=concat_out, alpha=alpha, lamda=lamda, layer=n_layers-1
            ))
        
        self.head = nn.Identity()
    
    def forward(self, g, input_feature, return_hidden=False):
        # 输入投影
        h = self.input_proj(input_feature)
        h0 = h  # 保存初始特征用于残差连接
        
        hidden_list = []
        for layer in self.layers:
            h = layer(g, h, h0)
            if return_hidden:
                hidden_list.append(h)
        
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)


class GCNIILayer(nn.Module):
    """
    单层GCNII，包含初始残差和恒等映射
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
                 alpha=0.1,
                 lamda=0.5,
                 layer=0):
        super(GCNIILayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.out_feat = out_dim * n_heads if concat_out else out_dim
        self.concat_out = concat_out
        self.alpha = alpha
        self.lamda = lamda
        self.layer = layer
        
        # 特征变换
        self.fc = nn.Linear(in_dim, self.out_feat, bias=False)
        
        # 边特征投影
        self.fc_edge = nn.Linear(e_dim, self.out_feat, bias=False)
        
        # 恒等映射的权重（可学习）
        self.theta = nn.Parameter(torch.tensor(lamda))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 残差连接
        if residual:
            if in_dim != self.out_feat:
                self.res_fc = nn.Linear(in_dim, self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        # 归一化
        self.norm = norm(self.out_feat) if norm else None
        self.activation = activation
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.constant_(self.theta, self.lamda)
    
    def forward(self, g, feat, h0):
        """
        feat: 当前层输入
        h0: 初始输入（用于初始残差连接）
        """
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
            
            # 特征变换
            h = self.fc(feat_dst)
            
            # 先对源节点特征进行投影
            feat_src_proj = self.fc(feat_src)
            g.srcdata['h'] = feat_src_proj
            
            # 边特征处理
            if 'attr' in g.edata:
                edge_feat = self.fc_edge(g.edata['attr'])
                g.edata['e'] = edge_feat
            
            # 消息传递（均值聚合）
            if 'e' in g.edata:
                g.update_all(
                    lambda edges: {'m': edges.src['h'] + edges.data['e']},
                    fn.mean('m', 'neigh')
                )
            else:
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            
            # 获取聚合结果
            neigh = g.dstdata['neigh']
            
            # GCNII更新公式：
            # h^(l+1) = (1 - alpha) * P * h^(l) + alpha * h^(0) + lamda * h^(l)
            # 其中P是图卷积操作
            
            # 图卷积部分
            h_conv = (1 - self.alpha) * neigh + self.alpha * self.fc(h0[:h.size(0)])
            
            # 恒等映射
            h_identity = self.theta * h
            
            # 组合
            rst = h_conv + h_identity
            
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

