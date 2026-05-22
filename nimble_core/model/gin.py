import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn  # <-- This is the crucial line
# Assuming expand_as_pair, create_activation, and create_norm are defined elsewhere
def expand_as_pair(input_, allow_zero=False):
    """
    Converts a single dimension or a tuple of dimensions into a pair (src_dim, dst_dim).
    
    Args:
        input_: int or tuple[int, int].
        allow_zero: bool, whether to allow zero dimension.

    Returns:
        tuple[int, int]: (src_dim, dst_dim)
    """
    if isinstance(input_, tuple):
        src_dim, dst_dim = input_
    else:
        src_dim = dst_dim = input_
        
    if not allow_zero:
        if src_dim == 0 or dst_dim == 0:
            raise ValueError("Feature dimensions must be positive.")
            
    return src_dim, dst_dim
def create_activation(name):
    """
    Creates a PyTorch activation module based on its string name.
    
    Args:
        name: str, name of the activation function (e.g., 'relu', 'leakyrelu', 'sigmoid').

    Returns:
        nn.Module or None: The activation module or None if 'identity'.
    """
    if name is None or name.lower() == 'identity':
        return nn.Identity()
    elif name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'leakyrelu':
        # Default slope of 0.01 is common, but often GAT uses 0.2
        return nn.LeakyReLU(0.2) 
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
def create_norm(name):
    """
    Returns a function/class for creating a PyTorch normalization layer.
    
    Args:
        name: str, name of the normalization layer (e.g., 'batchnorm', 'layernorm').

    Returns:
        callable: A function (usually a class) that takes the feature dimension 
                  and returns an initialized normalization layer (nn.Module).
    """
    if name is None or name.lower() == 'none':
        return None
    elif name.lower() == 'batchnorm':
        # Returns the class nn.BatchNorm1d, which will be called later with the dimension
        return nn.BatchNorm1d 
    elif name.lower() == 'layernorm':
        # Returns the class nn.LayerNorm
        return nn.LayerNorm 
    else:
        raise ValueError(f"Unknown normalization layer: {name}")
class GIN(nn.Module):
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
                 encoding=False
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.gin_layers = nn.ModuleList() # Changed name from 'gats' to 'gin_layers' for clarity
        self.concat_out = concat_out

        # The input/output dimensions of the GAT-style multi-head layers must be matched.
        
        # Helper functions (assumed to be available)
        def create_act(act): return create_activation(act)
        def create_norm_fn(norm): return norm(hidden_dim * n_heads) if norm else None

        last_activation = create_act(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        # --- First Layer ---
        if self.n_layers == 1:
            self.gin_layers.append(GINConv(
                n_dim, e_dim, out_dim, n_heads_out, feat_drop, attn_drop, negative_slope,
                last_residual, norm=last_norm, activation=last_activation, concat_out=self.concat_out
            ))
        else:
            self.gin_layers.append(GINConv(
                n_dim, e_dim, hidden_dim, n_heads, feat_drop, attn_drop, negative_slope,
                residual, create_act(activation),
                norm=create_norm_fn(norm), concat_out=True # Intermediate layers MUST concatenate
            ))
            
            # --- Hidden Layers ---
            for _ in range(1, self.n_layers - 1):
                # Input dimension is hidden_dim * n_heads (from the previous concat layer)
                self.gin_layers.append(GINConv(
                    hidden_dim * self.n_heads, e_dim, hidden_dim, n_heads,
                    feat_drop, attn_drop, negative_slope,
                    residual, create_act(activation),
                    norm=create_norm_fn(norm), concat_out=True # Intermediate layers MUST concatenate
                ))
                
            # --- Output Layer ---
            self.gin_layers.append(GINConv(
                hidden_dim * self.n_heads, e_dim, out_dim, n_heads_out,
                feat_drop, attn_drop, negative_slope,
                last_residual, last_activation, norm=last_norm, concat_out=self.concat_out
            ))
            
        self.head = nn.Identity()

    def forward(self, g, input_feature, return_hidden=False):
        h = input_feature
        hidden_list = []
        for layer in range(self.n_layers):
            h = self.gin_layers[layer](g, h)
            hidden_list.append(h)
            
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        # Note: self.num_heads is not defined in the original GAT, 
        # using the final layer's output dimension logic.
        final_dim = self.out_dim * self.n_heads if self.concat_out else self.out_dim
        self.head = nn.Linear(final_dim, num_classes)

class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 e_dim, # Edge dimension is used by concatenating with node features
                 out_dim,
                 n_heads,
                 feat_drop=0.0,
                 attn_drop=0.0, # Ignored in standard GIN
                 negative_slope=0.2, # Ignored in standard GIN
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):

        super(GINConv, self).__init__()

        # --- Parameter Matching & Setup ---
        self.n_heads = n_heads
        self.src_feat, self.dst_feat = expand_as_pair(in_dim)
        self.edge_feat = e_dim
        self.out_feat = out_dim
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.concat_out = concat_out

        # GIN output dimension must match GAT's output for layer stacking
        self.gin_out_dim = self.out_feat * self.n_heads
        
        # GIN Learnable Epsilon
        self.eps = nn.Parameter(torch.zeros(1))

        # GIN's core: MLP applied *after* aggregation
        # Input to MLP is (self_feature + summed_neighbor_features).
        # To handle edge features (EGIN-style), we concatenate them before the final MLP.
        
        # 1. Linear layer for node features (same input/output dimension)
        self.fc_node = nn.Linear(self.src_feat, self.gin_out_dim, bias=bias)
        # 2. Linear layer for edge features (transform edges to match node output dim)
        self.fc_edge = nn.Linear(self.edge_feat, self.gin_out_dim, bias=False) 
        
        # 3. MLP: Used to transform the combined (self + neighbor + edge) features.
        # Standard GIN uses a 2-layer MLP. Input is self.gin_out_dim.
        self.mlp = nn.Sequential(
            nn.Linear(self.gin_out_dim, self.gin_out_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(self.gin_out_dim, self.gin_out_dim, bias=bias)
        )

        # --- Residual (GAT-style) ---
        if residual:
            if self.dst_feat != self.gin_out_dim:
                self.res_fc = nn.Linear(self.dst_feat, self.gin_out_dim, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # --- Normalization ---
        self.norm = norm(self.gin_out_dim) if norm else None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize epsilon to zero
        nn.init.constant_(self.eps, 0)
        
        # Initialize MLPs (Xavier initialization is common for GIN)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_node.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_edge.weight, gain=gain)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
        
        if isinstance(self.res_fc, nn.Linear):
             nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
         # Kept for signature matching
        pass

    def forward(self, graph, feat, get_attention=False):
        # The GINConv does not support 'get_attention=True' since it's not attention-based.
        # We handle it by returning None for attention.
        
        with graph.local_scope():
            # --- Feature Preparation ---
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                res_input = h_dst
            else:
                h_src = h_dst = self.feat_drop(feat)
                res_input = h_dst
                if graph.is_block:
                    res_input = h_dst[:graph.number_of_dst_nodes()]

            # Transform input features for projection
            h_src_proj = self.fc_node(h_src)
            graph.srcdata['h'] = h_src_proj
            
            # Edge feature transformation (EGIN-style)
            edge_feature = graph.edata['attr']
            edge_h = self.fc_edge(edge_feature)
            graph.edata['e'] = edge_h

            # --- Message Passing (Sum Aggregation) ---
            # Message is the projected neighbor feature (h) + projected edge feature (e)
            # The sum aggregator accumulates all incoming messages.
            graph.update_all(
                lambda edges: {'m': edges.src['h'] + edges.data['e']},
                fn.sum('m', 'neigh')
            )
            
            neigh_sum = graph.dstdata['neigh']
            
            # --- GIN Update Function ---
            # GIN formula: (1 + eps) * self_feature + summed_neighbor_features
            
            # 1. Get the destination node's *projected* feature for the self-loop
            if isinstance(feat, tuple):
                self_h_proj = self.fc_node(h_dst)
            else:
                # In homogeneous/block case, dst features are the first N nodes of src features
                self_h_proj = h_src_proj[:neigh_sum.shape[0]] 
            
            h_combined = (1 + self.eps) * self_h_proj + neigh_sum
            
            # 2. Apply MLP
            rst = self.mlp(h_combined)
            
            # --- Residual (Matching GAT Logic) ---
            if self.res_fc is not None:
                # The residual connection uses the original input feature (h_dst/res_input)
                # and matches the output dimension (self.gin_out_dim).
                resval = self.res_fc(res_input) 
                rst = rst + resval

            # --- Output Shaping & Final Ops ---
            # GIN output is already (N_dst, self.gin_out_dim).
            
            # If concat_out is True (N_heads * out_dim), we skip reshaping.
            if not self.concat_out and self.n_heads > 1:
                 # Emulate GAT's head averaging: (N, N_heads, out_feat) -> (N, out_feat)
                rst = rst.view(-1, self.n_heads, self.out_feat).mean(dim=1)
                
            if self.norm is not None:
                rst = self.norm(rst)

            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                 # GIN does not produce attention scores
                return rst, None
            else:
                return rst