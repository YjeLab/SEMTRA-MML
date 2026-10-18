import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
# --- PyG 占位符 ---
try:
    from torch_geometric.nn import GATConv
except ImportError:
    print("警告: 未找到 PyTorch Geometric。将使用一个简化的占位符代替GATConv。")
    class GATConv(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels * heads)
        def forward(self, x, edge_index):
            return self.linear(x)

# ============================= 1. 辅助模块 =======================================
# --- GAT预训练相关模块 ---
class RoadEmbeddingGenerator(nn.Module):
    """
    使用GAT从道路图的节点特征生成道路嵌入。
    """
    def __init__(self, in_features, embedding_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim), 
            nn.ReLU(inplace=True), 
            nn.LayerNorm(embedding_dim)
        )
        # GAT层列表
        self.gat_layers = nn.ModuleList(
            [GATConv(embedding_dim, embedding_dim, heads=n_heads, dropout=dropout, concat=False) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x_feat, edge_index):
        # 初始投影
        road_embeddings = self.feature_projection(x_feat)
        # 通过多层GAT
        for gat_layer in self.gat_layers:
            identity = road_embeddings
            road_embeddings = gat_layer(road_embeddings, edge_index)
            road_embeddings = self.dropout(road_embeddings)
            road_embeddings = self.layer_norm(road_embeddings + identity) # 残差连接
            road_embeddings = F.relu(road_embeddings)
        return road_embeddings

class StagedGAT(nn.Module):
    """
    用于分阶段训练的 GAT 模型，专为链接预测预训练任务设计。
    """
    def __init__(self, in_features, embedding_dim, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        # 核心的GAT模块，用于生成嵌入
        self.gat = RoadEmbeddingGenerator(in_features, embedding_dim, n_heads, n_layers, dropout)
        
        # 链接预测的分类器：接收两条道路嵌入的拼接，输出一个logit
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, x_feat, edge_index):
        """只做 GAT 的前向传播，生成嵌入。"""
        return self.gat(x_feat, edge_index)
    
    def predict_links(self, road_embeddings, edge_indices_to_predict):
        """根据生成的嵌入来预测边的存在。"""
        # 从嵌入矩阵中取出对应边的起点和终点嵌入
        start_nodes_embeds = road_embeddings[edge_indices_to_predict[0]]
        end_nodes_embeds = road_embeddings[edge_indices_to_predict[1]]
        
        # 将起点和终点嵌入拼接起来
        edge_embeds = torch.cat([start_nodes_embeds, end_nodes_embeds], dim=1)
        
        # 通过分类器得到预测logit
        logits = self.link_predictor(edge_embeds)
        return logits.squeeze(-1)
    
class GridEmbedding(nn.Module):
    """使用CNN从城市栅格特征图中预计算嵌入，并提供批量查询功能。"""
    def __init__(self, in_channels, embedding_dim, grid_map_shape, pad_token_id):
        super().__init__()
        self.grid_map_shape = grid_map_shape
        self.num_total_grids = grid_map_shape[0] * grid_map_shape[1]
        self.embedding_dim = embedding_dim
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),
        )
        
        self.vocab_size = self.num_total_grids + 2 # CLS, PAD
        self.embedding_lookup = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=pad_token_id)
        # 权重将被预计算填充，但可以选择在训练中微调
        self.embedding_lookup.weight.requires_grad = True 

    def precompute_embeddings(self, grid_feature_map, device):
        print("正在预计算CNN栅格嵌入...")
        self.cnn.to(device)
        grid_feature_map = grid_feature_map.to(device)
        with torch.no_grad():
            conv_output = self.cnn(grid_feature_map.unsqueeze(0))
            flattened_embeddings = conv_output.permute(0, 2, 3, 1).squeeze(0).view(-1, self.embedding_dim)
            self.embedding_lookup.weight.data[0:self.num_total_grids] = flattened_embeddings
            print("CNN栅格嵌入预计算完成。")
        self.cnn.to('cpu') # 释放GPU显存

    def forward(self, grid_indices):
        return self.embedding_lookup(grid_indices)

class GPSEmbedding(nn.Module):
    """将4维GPS特征(lon, lat, cos_bearing, sin_bearing)投影到嵌入空间。"""
    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, gps_points):
        return self.projection(gps_points)

class PositionalEncoding(nn.Module):
    """标准的Transformer位置编码，适配batch_first=True。"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)
    
# 一个通用的交叉注意力层
class CrossAttentionLayer(nn.Module):
    """
    一个通用的交叉注意力模块，用于一个序列(Query)关注另一个序列(Key/Value)。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # 交叉注意力计算
        attn_output, _ = self.cross_attn(
            query=query, 
            key=key, 
            value=value,
            key_padding_mask=key_padding_mask
        )
        # 残差连接和层归一化
        return self.norm(query + self.dropout(attn_output))

class GroupedMotionProbe(nn.Module):
    """Evidence-conditioned assignment head for time-free motion anchors.

    This head does not modify the trajectory representation.  It is used to
    train and probe the 12 geometric/context anchors while the representation
    path can use the more stable anchor cross-attention injection.
    """

    def __init__(self, d_model, num_anchors, evidence_dim, dropout=0.1,
                 evidence_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.evidence_scale = evidence_scale
        self.e_proj = nn.Linear(evidence_dim, d_model)
        self.anchor_evidence = nn.Parameter(
            torch.empty(num_anchors, evidence_dim)
        )
        nn.init.xavier_uniform_(self.anchor_evidence)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, evidence):
        # ``query`` is accepted for API symmetry, but labels are predicted from
        # local time-free evidence so that probing does not perturb the TTE path.
        e = self.dropout(self.e_proj(evidence))
        ae = self.e_proj(self.anchor_evidence.to(query.dtype))
        logits = self.evidence_scale * torch.einsum(
            'bld,ad->bla', e, ae
        ) / math.sqrt(self.d_model)
        direction = torch.softmax(logits[..., :4], dim=-1)
        context = torch.sigmoid(logits[..., 4:])
        assignment = torch.cat([direction, context], dim=-1)
        return assignment, logits


class GroupedMotionResidualAdapter(nn.Module):
    """Grouped motion semantics with a gated evidence residual.

    Direction anchors are mutually exclusive.  Road-context and speed-regime
    anchors are multi-label.  Unlike a semantic bottleneck, the main Road-GPS
    token representation is preserved and receives only a gated residual.
    """
    GROUP_BOUNDS = {
        'direction': (0, 4),
        'road_context': (4, 10),
        'speed_context': (10, 12),
    }

    def __init__(self, d_model, num_anchors, evidence_dim, dropout=0.1,
                 evidence_scale=1.0, residual_scale_init=1e-3):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.evidence_dim = evidence_dim
        self.evidence_scale = evidence_scale
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.e_proj = nn.Linear(evidence_dim, d_model)
        self.anchor_evidence = nn.Parameter(torch.empty(num_anchors, evidence_dim))
        nn.init.xavier_uniform_(self.anchor_evidence)

        context_dim = d_model * 3
        self.residual_mlp = nn.Sequential(
            nn.Linear(d_model + evidence_dim + context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model + evidence_dim + context_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.gate_bias = nn.Parameter(torch.full((d_model,), -2.0))

    def _group_evidence_logits(self, evidence, anchor_evidence, start, end):
        e = self.e_proj(evidence)
        ae = self.e_proj(anchor_evidence[start:end])
        logits = self.evidence_scale * torch.einsum(
            'bld,ad->bla', e, ae
        ) / math.sqrt(self.d_model)
        return logits

    def forward(self, query, evidence, anchors):
        q = self.q_proj(query)
        k = self.k_proj(anchors)
        v = self.v_proj(anchors)
        ae_raw = self.anchor_evidence.to(query.dtype)

        assignments = []
        assignment_logits = []
        contexts = []

        # Direction group: mutually exclusive geometric motion states.
        d_start, d_end = self.GROUP_BOUNDS['direction']
        d_logits = self._group_evidence_logits(evidence, ae_raw, d_start, d_end)
        d_probs = torch.softmax(d_logits, dim=-1)
        d_context = torch.einsum('bla,ad->bld', d_probs, v[d_start:d_end])
        assignments.append(d_probs)
        assignment_logits.append(d_logits)
        contexts.append(d_context)

        # Multi-label context groups.
        for group in ('road_context', 'speed_context'):
            start, end = self.GROUP_BOUNDS[group]
            logits = self._group_evidence_logits(evidence, ae_raw, start, end)
            probs = torch.sigmoid(logits)
            context = torch.einsum('bla,ad->bld', probs, v[start:end])
            assignments.append(probs)
            assignment_logits.append(logits)
            contexts.append(context)

        assignment = torch.cat(assignments, dim=-1)
        assignment_logits = torch.cat(assignment_logits, dim=-1)
        context = torch.cat(contexts, dim=-1)

        adapter_input = torch.cat([query, evidence, context], dim=-1)
        residual = self.out_proj(self.residual_mlp(adapter_input))
        gate = torch.sigmoid(self.gate_mlp(adapter_input) + self.gate_bias)
        gated_residual = self.residual_scale * gate * self.dropout(residual)
        out = self.norm(query + gated_residual)
        return out, assignment, assignment_logits, gate


class GeoMotionAttention(nn.Module):
    """Evidence-conditioned motion-semantic attention.

    ``injection_mode='direction'`` restricts the representation residual to the
    four geometric direction anchors; the full assignment is still returned for
    quantitative probing.
    """
    def __init__(self, d_model, num_anchors, evidence_dim, n_heads=4, dropout=0.1,
                 evidence_scale=1.0, injection_scale_init=1.0, injection_mode='all'):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.evidence_scale = evidence_scale
        self.injection_mode = injection_mode
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.e_proj = nn.Linear(evidence_dim, d_model)
        self.anchor_evidence = nn.Parameter(torch.empty(num_anchors, evidence_dim))
        nn.init.xavier_uniform_(self.anchor_evidence)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.injection_scale = nn.Parameter(torch.tensor(float(injection_scale_init)))

    def forward(self, query, evidence, anchors):
        q = self.q_proj(query)
        k = self.k_proj(anchors)
        v = self.v_proj(anchors)
        e = self.e_proj(evidence)
        ae = self.e_proj(self.anchor_evidence.to(query.dtype))

        semantic_logits = torch.einsum('bld,ad->bla', q, k) / math.sqrt(self.d_model)
        evidence_logits = self.evidence_scale * torch.einsum(
            'bld,ad->bla', e, ae
        ) / math.sqrt(self.d_model)
        attn_logits = semantic_logits + evidence_logits

        direction_logits = attn_logits[..., :4]
        direction_probs = torch.softmax(direction_logits, dim=-1)
        context_probs = torch.sigmoid(attn_logits[..., 4:])
        assignment = torch.cat([direction_probs, context_probs], dim=-1)

        if self.injection_mode == 'direction':
            attn = direction_probs
            context = torch.einsum('bla,ad->bld', attn, v[:4])
        else:
            attn = torch.softmax(attn_logits, dim=-1)
            context = torch.einsum('bla,ad->bld', attn, v)

        attn = self.attn_dropout(attn)
        out = self.norm(query + self.injection_scale * self.dropout(self.out_proj(context)))
        return out, assignment, attn_logits


# # ============================= 3. 主干模型 =========================================
def _target_similarity_from_words(words):
    """Deterministic textual similarity prior used by the stable v2 bank."""
    size = len(words)
    target_matrix = torch.eye(size)
    synonyms = {
        'fast': ['rapid'], 'slow': ['sluggish'], 'steady': ['smooth', 'constant'],
        'stop': ['stationary'], 'turn': ['u-turn'],
        'accelerate': [], 'decelerate': ['brake']
    }
    antonyms = {
        'fast': ['slow', 'sluggish', 'stationary', 'stop'],
        'accelerate': ['decelerate', 'brake', 'stop'],
        'steady': ['erratic', 'dynamic']
    }
    for i in range(size):
        for j in range(i + 1, size):
            w1, w2 = words[i], words[j]
            if w2 in synonyms.get(w1, []) or w1 in synonyms.get(w2, []):
                sim = 0.8
            elif w2 in antonyms.get(w1, []) or w1 in antonyms.get(w2, []):
                sim = -0.8
            else:
                sim = 0.0
            target_matrix[i, j] = target_matrix[j, i] = sim
    return target_matrix


class SEMTRA_MML(nn.Module):
    """
    运动语义增强的多模态轨迹学习框架。
    - 双流架构: (Road+GPS)流 和 Grid流
    - 运动语义锚点: 通过交叉注意力注入运动先验知识
    - 自监督损失: 拉近模态间表示的相关性，优化运动语义损失
    """
    def __init__(self, num_roads, num_grids, grid_map_shape, grid_channels, 
                 embedding_dim, nhead_tf, num_shallow_layers, num_deep_layers,    
                 dim_feedforward_tf, num_classes, dropout=0.1, max_len=1000,device=None,pre_flag=False,
                 motion_version='v2', motion_feature_dim=40, num_motion_anchors=12, lambda_motion=0.1,
                 residual_scale_init=1e-3, motion_supervision='full',
                 injection_scale_init=1.0, injection_mode='all',
                 motion_injection_mode='geo', anchor_structure_weight=0.0,
                 perturb_structure_target='none'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.motion_version = motion_version
        self.lambda_motion = lambda_motion
        self.motion_supervision = motion_supervision
        self.motion_injection_mode = motion_injection_mode
        self.anchor_structure_weight = anchor_structure_weight
        self.perturb_structure_target = perturb_structure_target
        self.motion_loss_scale = 1.0
        # --- 1. 初始化各模态的嵌入层 ---
        # --- 【关键修正】在这里为错误的 pad_token 留出冗余空间 ---
        self.num_roads = num_roads
        self.unk_token_id = num_roads
        self.cls_token_id = num_roads + 1
        self.pad_token_id = num_roads + 2  # 这是我们【期望】的正确pad token
        self.mask_token_id = num_roads + 3
        
        # 旧的 vocab_size 是 num_roads + 4，最大有效索引是 num_roads + 3
        # 错误的 pad_token 会产生 num_roads + 3 + 1 = num_roads + 4 的索引
        # 所以我们将 vocab_size 再加1，变成 num_roads + 5
        self.road_vocab_size = num_roads + 5 # 0..N-1, UNK, CLS, PAD, MASK, [BAD_PAD]
        
        print(f"警告: 为兼容有问题的collate_fn，road_vocab_size被设置为{self.road_vocab_size}以提供冗余空间。")
        
        # 使用新的、更大的词汇表大小来创建嵌入层
        self.road_embedding_layer = nn.Embedding(
            self.road_vocab_size, 
            embedding_dim, 
            padding_idx=self.pad_token_id # padding_idx 仍然使用我们期望的正确值
        )
        self.grid_pad_token_id = num_grids + 1
        self.grid_embedding_layer = GridEmbedding(grid_channels, embedding_dim, grid_map_shape, self.grid_pad_token_id)
        self.gps_embedding_layer = GPSEmbedding(embedding_dim, dropout)
        self.road_gps_fusion_layer = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU(inplace=True))
        
        # --- 运动语义模块 ---
        if motion_version == 'v4':
            self.motion_words = [
                'straight', 'gentle_turn', 'sharp_turn', 'u_turn',
                'intersection', 'road_class_transition',
                'short_segment', 'long_corridor',
                'highway_context', 'local_road_context',
                'deceleration_prone_context', 'acceleration_prone_context'
            ]
            self.num_motion_words = num_motion_anchors
            self.motion_feature_dim = motion_feature_dim
            self.motion_anchor_embeddings = nn.Parameter(
                torch.empty(self.num_motion_words, self.embedding_dim)
            )
            nn.init.xavier_uniform_(self.motion_anchor_embeddings)
            self.road_gps_geo_motion_attn = GroupedMotionResidualAdapter(
                self.embedding_dim,
                self.num_motion_words,
                motion_feature_dim,
                dropout,
                evidence_scale=1.0,
                residual_scale_init=residual_scale_init,
            )
            self.register_buffer(
                'target_similarity_matrix',
                torch.eye(self.num_motion_words)
            )
        elif motion_version in ('v3', 'v5'):
            self.motion_words = [
                'straight', 'gentle_turn', 'sharp_turn', 'u_turn', 'intersection',
                'road_class_transition', 'short_segment', 'long_corridor',
                'highway_context', 'local_road_context',
                'deceleration_prone_context', 'acceleration_prone_context'
            ]
            self.num_motion_words = num_motion_anchors
            self.motion_feature_dim = motion_feature_dim
            self.motion_anchor_embeddings = nn.Parameter(
                torch.empty(self.num_motion_words, self.embedding_dim)
            )
            nn.init.xavier_uniform_(self.motion_anchor_embeddings)
            self.road_gps_geo_motion_attn = GeoMotionAttention(
                embedding_dim, self.num_motion_words, motion_feature_dim,
                nhead_tf, dropout, evidence_scale=1.0,
                injection_scale_init=injection_scale_init,
                injection_mode=injection_mode
            )
            # v3.1: separate the stable injection path from the quantitative
            # explanation path.  The probe is always created so checkpoints
            # retain the same interface; it is only used in anchor mode.
            self.motion_probe_head = GroupedMotionProbe(
                embedding_dim, self.num_motion_words, motion_feature_dim,
                dropout, evidence_scale=1.0
            )
            if motion_injection_mode == 'v2_anchor':
                # Main-task injection keeps the original 14 textual anchors,
                # while the 12-anchor probe provides grouped, time-free
                # quantitative explanations without perturbing the TTE path.
                self.injection_motion_words = [
                    'straight', 'turn', 'u-turn', 'brake', 'accelerate',
                    'decelerate', 'stop', 'steady', 'smooth', 'fast', 'slow',
                    'rapid', 'stationary', 'sluggish'
                ]
                self.num_injection_anchors = len(self.injection_motion_words)
                self.injection_anchor_embeddings = nn.Parameter(
                    torch.empty(self.num_injection_anchors, self.embedding_dim)
                )
                nn.init.xavier_uniform_(self.injection_anchor_embeddings)
                self.register_buffer(
                    'injection_target_similarity_matrix',
                    _target_similarity_from_words(self.injection_motion_words)
                )
            # v3 replaces the hand-crafted textual similarity matrix with
            # deterministic geometry/context assignment supervision.
            self.register_buffer('target_similarity_matrix', torch.eye(self.num_motion_words))
        else:
            self.motion_words = ['straight', 'turn', 'u-turn', 'brake', 'accelerate', 'decelerate', 'stop', 'steady', 'smooth', 'fast', 'slow', 'rapid', 'stationary', 'sluggish']
            self.num_motion_words = len(self.motion_words)
            self.motion_anchor_embeddings = nn.Parameter(torch.empty(self.num_motion_words, self.embedding_dim)); nn.init.xavier_uniform_(self.motion_anchor_embeddings)
            target_similarity_matrix = self._create_target_similarity_matrix()
            self.register_buffer('target_similarity_matrix', target_similarity_matrix)

        # R4-Q4 T 敏感性实验:按 severity 把 target_similarity_matrix 的非对角
        # 元素以固定概率替换为 [-0.8, 0.8] 随机先验(模拟人工定义出错);
        # 对角线(自相似=1)保持不变。扰动由 severity 级别确定性生成,
        # 与 run seed 无关,保证同一 severity 在所有 run 中扰动同一矩阵。
        sev = str(getattr(self, 'perturb_structure_target', 'none') or 'none')
        if sev != 'none' and hasattr(self, 'target_similarity_matrix'):
            severity = {'flip10': 0.10, 'flip25': 0.25, 'flip50': 0.50,
                        'flip100': 1.0}.get(sev)
            if severity is None:
                raise ValueError(f'未知 perturb_structure_target: {sev}')
            T = self.target_similarity_matrix.clone()
            n = T.shape[0]
            gen = torch.Generator().manual_seed(20261017)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            n_flip = int(round(severity * len(pairs)))
            perm = torch.randperm(len(pairs), generator=gen).tolist()
            for idx in perm[:n_flip]:
                i, j = pairs[idx]
                T[i, j] = T[j, i] = (torch.rand(1, generator=gen).item() * 1.6 - 0.8)
            self.target_similarity_matrix = T
            print(f'[R4-Q4] target_similarity_matrix 扰动 severity={sev}: '
                  f'{n_flip}/{len(pairs)} 个非对角元素被替换为随机先验。')
        
        # --- 编码器与注意力模块 ---
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)
        def create_encoder(num_layers):
            layer = nn.TransformerEncoderLayer(embedding_dim, nhead_tf, dim_feedforward_tf, dropout, batch_first=True)
            return nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # 使用通用的 CrossAttentionLayer
        self.road_gps_motion_attn = CrossAttentionLayer(embedding_dim, nhead_tf, dropout)
        self.grid_motion_attn = CrossAttentionLayer(embedding_dim, nhead_tf, dropout)
        self.road_gps_encoder_shallow, self.road_gps_encoder_deep = create_encoder(num_shallow_layers), create_encoder(num_deep_layers)
        self.grid_encoder_shallow, self.grid_encoder_deep = create_encoder(num_shallow_layers), create_encoder(num_deep_layers)
        
        # --- 后期融合与任务头 ---
        self.fusion_projector = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.time_head = nn.Sequential(nn.Linear(embedding_dim, 1))
        self.class_head = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None
        
        # --- 预训练任务模块 ---
        if pre_flag:
            self.temperature = nn.Parameter(torch.tensor(0.07))
            # 为MLM融合创建一个独立的交叉注意力实例
            self.mlm_fusion_attn = CrossAttentionLayer(embedding_dim, nhead_tf, dropout)
            self.mlm_classification_head = nn.Linear(embedding_dim, self.road_vocab_size)
        
    def _create_target_similarity_matrix(self):
        """辅助函数，用于创建我们理想中的语义相似度矩阵。"""
        words = self.motion_words
        size = len(words)
        target_matrix = torch.eye(size) # 对角线为1
        
        # 定义近义词和反义词对 (可以根据需要扩展)
        synonyms = {
            'fast': ['rapid'], 'slow': ['sluggish'], 'steady': ['smooth', 'constant'],
            'stop': ['stationary'], 'turn':['u-turn'], 'accelerate':[], 'decelerate':['brake']
        }
        antonyms = {
            'fast': ['slow', 'sluggish', 'stationary', 'stop'],
            'accelerate': ['decelerate', 'brake', 'stop'],
            'steady': ['erratic', 'dynamic'] # 假设我们词汇表里有这些词
        }
        
        # 填充矩阵
        for i in range(size):
            for j in range(i, size):
                w1, w2 = words[i], words[j]
                if i == j: continue
                
                sim = 0.0 # 默认为不相关
                if w2 in synonyms.get(w1, []) or w1 in synonyms.get(w2, []):
                    sim = 0.8 # 近义词
                elif w2 in antonyms.get(w1, []) or w1 in antonyms.get(w2, []):
                    sim = -0.8 # 反义词
                
                target_matrix[i, j] = target_matrix[j, i] = sim
                
        return target_matrix
    
    def _apply_span_mask(self, token_ids, pad_mask):
        """
        在模型内部对一个批次的token_ids进行连续段落掩码。
        pad_mask: [B, L], True表示填充位置。
        """
        # clone 输入以避免原地修改
        masked_token_ids = token_ids.clone()
        labels = torch.full_like(token_ids, -100) # 初始化标签为-100

        for i in range(token_ids.size(0)): # 遍历batch中的每个样本
            # 找到非填充部分的有效长度 (不包括[CLS])
            # 我们假设[CLS]总是在第一个位置
            seq_len = (~pad_mask[i, 1:]).sum().item()
            if seq_len < 3: continue # 序列太短，不进行掩码
            
            num_to_mask = int(math.ceil(seq_len * 0.15))
            if num_to_mask == 0: continue

            # 使用几何分布确定掩码段长度
            span_lengths = np.random.geometric(p=0.2, size=num_to_mask)
            
            masked_indices = set()
            attempts = 0
            # 增加尝试次数以确保能尽可能多地掩码
            while len(masked_indices) < num_to_mask and attempts < 100:
                # 随机选择一个掩码段的长度
                span_len = min(np.random.choice(span_lengths), seq_len - len(masked_indices))
                if span_len == 0:
                    attempts += 1
                    continue
                
                # 随机选择一个起始位置（在有效长度内，即1到seq_len）
                start_idx = random.randint(1, seq_len - span_len + 1)

                # 检查这个区域是否与已有的掩码区域重叠
                is_valid_span = True
                for j in range(span_len):
                    if (start_idx + j) in masked_indices:
                        is_valid_span = False
                        break
                
                if is_valid_span:
                    for j in range(span_len):
                        idx = start_idx + j
                        if len(masked_indices) < num_to_mask:
                            masked_indices.add(idx)
                attempts += 1
            
            # 应用掩码
            for idx in masked_indices:
                # labels的索引与token_ids完全一致
                labels[i, idx] = masked_token_ids[i, idx]
                masked_token_ids[i, idx] = self.mask_token_id

        return masked_token_ids, labels
        
    def load_static_road_embeddings(self, path, freeze=False):
        print(f"正在加载静态道路嵌入: {path}...")
        embeds = torch.load(path, map_location='cpu')
        self.road_embedding_layer.weight.data[0:self.num_roads] = embeds
        if freeze:
            self.road_embedding_layer.weight.requires_grad = False; self.road_embedding_layer.weight.data[self.num_roads:].requires_grad = True
            print("已知道路的嵌入已冻结，特殊tokens可学习。")
        else:
            print("所有道路嵌入（包括特殊tokens）都将进行微调。")

    def get_motion_anchors(self):
        """辅助函数，用于获取经过投影（如果需要的话）的运动锚点。"""
        # .weight 直接给出了 [num_words, D_raw] 的张量
        return self.motion_anchor_embeddings

    def forward(self, road_indices, road_mask, grid_indices, grid_mask, gps_points, gps_mask,
                task='classification', mtm_labels=None, motion_features=None, motion_labels=None,
                collect_attention=False):
        
        if task == 'pretrain':
            road_indices, mtm_labels = self._apply_span_mask(road_indices, road_mask)
            
        road_embed, grid_embed, gps_embed = self.road_embedding_layer(road_indices), self.grid_embedding_layer(grid_indices), self.gps_embedding_layer(gps_points)
        road_gps_fused_embed = self.road_gps_fusion_layer(torch.cat([road_embed, gps_embed], dim=-1))
        road_gps_embed_pos, grid_embed_pos = self.pos_encoder(road_gps_fused_embed), self.pos_encoder(grid_embed)
        road_gps_shallow_out, grid_shallow_out = self.road_gps_encoder_shallow(road_gps_embed_pos, src_key_padding_mask=road_mask), self.grid_encoder_shallow(grid_embed_pos, src_key_padding_mask=grid_mask)
        
        # 注入运动语义
        motion_anchors_expanded = self.motion_anchor_embeddings.unsqueeze(0).expand(road_indices.size(0), -1, -1)
        road_gps_motion_weights = None
        road_gps_motion_logits = None
        road_gps_motion_gate = None
        if self.motion_version == 'v4' and motion_features is not None:
            road_gps_motion_fused, road_gps_motion_weights, road_gps_motion_logits, road_gps_motion_gate = self.road_gps_geo_motion_attn(
                road_gps_shallow_out, motion_features, self.motion_anchor_embeddings
            )
        elif self.motion_version in ('v3', 'v5') and motion_features is not None:
            injection_mode = getattr(self, 'motion_injection_mode', 'geo')
            if injection_mode == 'v2_anchor':
                injection_anchors_expanded = self.injection_anchor_embeddings.unsqueeze(
                    0
                ).expand(road_indices.size(0), -1, -1)
                road_gps_motion_fused = self.road_gps_motion_attn(
                    road_gps_shallow_out,
                    injection_anchors_expanded,
                    injection_anchors_expanded,
                )
                road_gps_motion_weights, road_gps_motion_logits = self.motion_probe_head(
                    road_gps_shallow_out, motion_features
                )
                grid_motion_fused = self.grid_motion_attn(
                    grid_shallow_out,
                    injection_anchors_expanded,
                    injection_anchors_expanded,
                )
            elif injection_mode == 'anchor':
                # Stable performance path: reuse the v2 anchor cross-attention.
                road_gps_motion_fused = self.road_gps_motion_attn(
                    road_gps_shallow_out, motion_anchors_expanded, motion_anchors_expanded
                )
                road_gps_motion_weights, road_gps_motion_logits = self.motion_probe_head(
                    road_gps_shallow_out, motion_features
                )
            else:
                road_gps_motion_fused, road_gps_motion_weights, road_gps_motion_logits = self.road_gps_geo_motion_attn(
                    road_gps_shallow_out, motion_features, self.motion_anchor_embeddings
                )
        else:
            road_gps_motion_fused = self.road_gps_motion_attn(
                road_gps_shallow_out, motion_anchors_expanded, motion_anchors_expanded
            )
        grid_motion_fused = self.grid_motion_attn(grid_shallow_out, motion_anchors_expanded, motion_anchors_expanded)
        
        self.last_motion_weights = road_gps_motion_weights if self.motion_version != 'v2' else road_gps_motion_weights
        self.last_motion_assignment_logits = road_gps_motion_logits if self.motion_version in ('v3', 'v4', 'v5') else None
        self.last_motion_gate = road_gps_motion_gate if self.motion_version == 'v4' else None
        road_gps_deep_out = self.road_gps_encoder_deep(road_gps_motion_fused, src_key_padding_mask=road_mask)
        grid_deep_out = self.grid_encoder_deep(grid_motion_fused, src_key_padding_mask=grid_mask)
        
        if task == 'pretrain':
            # 1. 对比损失
            road_gps_cls = F.normalize(road_gps_deep_out[:, 0], p=2, dim=1)
            grid_cls = F.normalize(grid_deep_out[:, 0], p=2, dim=1)
            logits = torch.matmul(road_gps_cls, grid_cls.T) / self.temperature
            labels = torch.arange(logits.size(0)).to(logits.device)
            loss_cl = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            # 2. MLM损失
            fused_sequence_for_mlm = self.mlm_fusion_attn(road_gps_deep_out, grid_deep_out, grid_deep_out, key_padding_mask=grid_mask)
            masked_positions = (mtm_labels != -100)
            if not masked_positions.any():
                loss_mlm = torch.tensor(0.0, device=loss_cl.device)
            else:
                masked_output = fused_sequence_for_mlm[masked_positions]
                prediction_logits = self.mlm_classification_head(masked_output)
                true_labels = mtm_labels[masked_positions]
                loss_mlm = F.cross_entropy(prediction_logits, true_labels)
            
            return loss_cl, loss_mlm
            
        else: # 下游任务
            road_gps_cls, grid_cls = road_gps_deep_out[:, 0], grid_deep_out[:, 0]
            loss_ssl = None
            if self.training:
                loss_align = 1 - F.cosine_similarity(road_gps_cls, grid_cls, dim=-1).mean()
                if self.motion_version in ('v3', 'v4', 'v5'):
                    loss_ssl = loss_align
                    if road_gps_motion_weights is not None and motion_labels is not None and self.motion_supervision != 'none':
                        logits = self.last_motion_assignment_logits
                        # Direction anchors are mutually exclusive, so use CE.
                        dir_logits = logits[..., :4]
                        dir_labels = motion_labels[..., :4]
                        dir_mask = dir_labels.sum(dim=-1) > 0
                        if dir_mask.any():
                            dir_target = dir_labels[dir_mask].argmax(dim=-1)
                            loss_direction = F.cross_entropy(dir_logits[dir_mask], dir_target)
                        else:
                            loss_direction = torch.tensor(0.0, device=logits.device)

                        if self.motion_supervision == 'direction':
                            loss_motion = loss_direction
                        else:
                            # Context anchors are multi-label, so use weighted BCE.
                            # 空选择保护:某批可能没有任何上下文正例,
                            # 空 tensor 的 BCE 会返回 NaN 并毒化训练。
                            ctx_logits = logits[..., 4:]
                            ctx_labels = motion_labels[..., 4:]
                            ctx_mask = motion_labels.sum(dim=-1) > 0
                            selected_ctx_logits = ctx_logits[ctx_mask]
                            selected_ctx_labels = ctx_labels[ctx_mask]
                            if selected_ctx_logits.numel() == 0:
                                loss_motion = loss_direction
                            else:
                                positive = selected_ctx_labels.sum(dim=0)
                                negative = selected_ctx_labels.numel() - positive
                                pos_weight = (negative / positive.clamp(min=1.0)).clamp(1.0, 20.0)
                                loss_context = F.binary_cross_entropy_with_logits(
                                    selected_ctx_logits,
                                    selected_ctx_labels,
                                    pos_weight=pos_weight,
                                    reduction='mean'
                                )
                                loss_motion = loss_direction + loss_context
                        loss_scale = getattr(self, 'motion_loss_scale', 1.0)
                        loss_ssl = loss_ssl + loss_scale * self.lambda_motion * loss_motion
                    # Anchor diversity keeps the stable injection branch from
                    # collapsing while grouped anchors remain semantically
                    # distinguishable for probing.
                    structure_weight = getattr(self, 'anchor_structure_weight', 0.0)
                    if structure_weight > 0:
                        if getattr(self, 'motion_injection_mode', 'geo') == 'v2_anchor':
                            anchors_norm = F.normalize(
                                self.injection_anchor_embeddings, p=2, dim=1
                            )
                            target_sim = self.injection_target_similarity_matrix
                        else:
                            anchors_norm = F.normalize(
                                self.motion_anchor_embeddings, p=2, dim=1
                            )
                            target_sim = self.target_similarity_matrix
                        actual_sim = torch.matmul(anchors_norm, anchors_norm.t())
                        loss_ssl = loss_ssl + structure_weight * F.mse_loss(
                            actual_sim, target_sim
                        )
                else:
                    anchors_norm = F.normalize(self.motion_anchor_embeddings, p=2, dim=1)
                    actual_sim = torch.matmul(anchors_norm, anchors_norm.t())
                    loss_structure = F.mse_loss(actual_sim, self.target_similarity_matrix)
                    lambda_structure = 0.1
                    loss_ssl = loss_align + lambda_structure * loss_structure
            
            fused_representation = self.fusion_projector(torch.cat([road_gps_cls, grid_cls], dim=1))
            
            if task == 'classification': output = self.class_head(fused_representation)
            elif task == 'regression': output = self.time_head(fused_representation).squeeze(-1)
            else: raise ValueError(f"未知的下游任务: {task}")
            
            return (output, loss_ssl) if self.training else output

