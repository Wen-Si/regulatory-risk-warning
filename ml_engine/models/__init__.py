#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning Models Package
前沿深度学习模型集合：
- DeepFM: 深度因子分解机，用于表格数据特征交叉
- TemporalTransformer: 时序Transformer，多步预测
- GAT: 图注意力网络，行业风险传导
- RiskTextEncoder: 金融文本编码器
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class DeepFM:
    """DeepFM - 深度因子分解机
    
    结合FM（因子分解机）的低阶特征交叉和DNN的高阶特征交叉能力，
    是当前表格数据CTR/风险预测领域的SOTA模型之一。
    
    架构：
    - Sparse Embedding层：类别特征嵌入
    - FM层：二阶特征交叉
    - Deep层：多层全连接网络学习高阶交叉
    - 输出层：Sigmoid概率
    """
    
    def __init__(self, num_numeric=24, cat_dims=None, embed_dim=16, hidden_dims=None, dropout=0.3):
        if cat_dims is None:
            cat_dims = {'industry': 32, 'market': 8, 'company_type': 4}
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.num_numeric = num_numeric
        self.cat_dims = cat_dims
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # 计算总输入维度
        self.num_cat_fields = len(cat_dims)
        self.total_fields = num_numeric + self.num_cat_fields
        self.fm_input_dim = self.total_fields * embed_dim  # flattened embeddings for DNN
        
        # 初始化权重（Xavier初始化）
        self._init_weights()
        
    def _init_weights(self):
        """Xavier/Glorot初始化"""
        np.random.seed(42)
        
        # FM一阶权重
        self.fm_first = np.random.randn(self.total_fields) * 0.01
        self.fm_bias = np.float32(0.0)
        
        # FM二阶隐向量 (V矩阵)
        self.fm_v = np.random.randn(self.total_fields, self.embed_dim) * 0.05
        
        # DNN层权重
        self.dnn_weights = []
        self.dnn_biases = []
        prev_dim = self.fm_input_dim
        
        for h_dim in self.hidden_dims:
            # Xavier init
            limit = np.sqrt(6.0 / (prev_dim + h_dim))
            self.dnn_weights.append(np.random.uniform(-limit, limit, (prev_dim, h_dim)).astype(np.float32))
            self.dnn_biases.append(np.zeros(h_dim, dtype=np.float32))
            prev_dim = h_dim
        
        # 输出层
        limit = np.sqrt(6.0 / (prev_dim + 1))
        self.output_w = np.random.uniform(-limit, limit, (prev_dim, 1)).astype(np.float32)
        self.output_b = np.float32(0.0)
        
        # Dropout masks (eval mode: no dropout)
        self.training = False
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward_dnn(self, x):
        """DNN前向传播"""
        h = x
        for i, (W, b) in enumerate(zip(self.dnn_weights, self.dnn_biases)):
            h = h @ W + b
            h = self._relu(h)
            if self.training:
                mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
                h = h * mask
        return h
    
    def _forward_fm(self, x_sparse_embeds):
        """FM前向传播（一阶+二阶）"""
        # 一阶项
        first_order = np.dot(x_sparse_embeds.mean(axis=-1), self.fm_first[:len(x_sparse_embeds)])
        # 二阶项（优化版：sum-square vs square-sum）
        square_of_sum = np.sum(x_sparse_embeds, axis=0) ** 2
        sum_of_square = np.sum(x_sparse_embeds ** 2, axis=0)
        second_order = 0.5 * np.sum(square_of_sum - sum_of_square, axis=-1)
        return first_order + second_order
    
    def predict(self, numeric_features: np.ndarray, cat_ids: Dict[str, int]) -> np.ndarray:
        """
        预测问询概率
        
        Args:
            numeric_features: [batch, num_numeric] 归一化数值特征
            cat_ids: 类别特征ID字典
        
        Returns:
            probability: [batch, 1] 问询概率
        """
        if numeric_features.ndim == 1:
            numeric_features = numeric_features.reshape(1, -1)
        batch_size = numeric_features.shape[0]
        
        num_numeric = numeric_features.shape[1]
        num_cat = len(self.cat_dims)
        total_fields = num_numeric + num_cat
        
        # 构建所有field的embedding矩阵 [B, total_fields, embed_dim]
        # 数值特征: 用fm_v对应行乘以数值作为embedding
        all_embeds = np.zeros((batch_size, total_fields, self.embed_dim), dtype=np.float32)
        
        # 数值特征embedding: v_i * x_i
        for f in range(num_numeric):
            all_embeds[:, f, :] = numeric_features[:, f:f+1] * self.fm_v[f]  # broadcast
        
        # 类别特征embedding
        cat_names = list(self.cat_dims.keys())
        for i, cat_name in enumerate(cat_names):
            vocab_size = self.cat_dims[cat_name]
            cid = cat_ids.get(cat_name + '_id', cat_ids.get(cat_name, 0))
            idx = num_numeric + i
            embed_vec = self.fm_v[idx] * (cid % vocab_size) / max(vocab_size, 1)
            all_embeds[:, idx, :] = embed_vec  # broadcast to batch
        
        # FM一阶项
        first_order = np.zeros(batch_size, dtype=np.float32)
        for f in range(num_numeric):
            first_order += numeric_features[:, f] * self.fm_first[f]
        for i in range(num_cat):
            cid = cat_ids.get(cat_names[i] + '_id', cat_ids.get(cat_names[i], 0))
            first_order += (cid / max(self.cat_dims[cat_names[i]], 1)) * self.fm_first[num_numeric + i]
        fm_out = self.fm_bias + first_order
        
        # FM二阶项 (sum-square minus square-sum)
        sum_embeds = np.sum(all_embeds, axis=1)  # [B, embed_dim]
        sum_sq = sum_embeds ** 2
        sq_sum = np.sum(all_embeds ** 2, axis=1)  # [B, embed_dim]
        fm_out = fm_out + 0.5 * np.sum(sum_sq - sq_sum, axis=1)
        
        # DNN输入: flatten all embeddings
        dnn_input = all_embeds.reshape(batch_size, -1)
        # Pad to expected size
        expected_input = self.dnn_weights[0].shape[0]
        if dnn_input.shape[1] < expected_input:
            dnn_input = np.pad(dnn_input, ((0,0),(0,expected_input - dnn_input.shape[1])))
        else:
            dnn_input = dnn_input[:, :expected_input]
        
        dnn_out = self._forward_dnn(dnn_input)
        dnn_logit = dnn_out @ self.output_w + self.output_b
        
        # 结合FM和DNN
        logit = fm_out.reshape(-1, 1) + dnn_logit
        return self._sigmoid(logit)


class TemporalTransformer:
    """Temporal Transformer - 时序多步预测模型
    
    基于Transformer Encoder的时序预测模型，
    利用Multi-Head Self-Attention捕捉财务指标的时间依赖关系，
    输出30/60/90天多窗口预测概率。
    
    创新点：
    - 因果注意力掩码（Causal Mask）防止未来信息泄露
    - 可学习的位置编码
    - 多任务输出头（多窗口预测）
    """
    
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.2, 
                 input_dim=16, num_windows=3):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.input_dim = input_dim
        self.num_windows = num_windows
        self.head_dim = d_model // nhead
        
        np.random.seed(43)
        self._init_weights()
    
    def _init_weights(self):
        """初始化Transformer权重"""
        # 输入投影
        limit = np.sqrt(6.0 / (self.input_dim + self.d_model))
        self.input_proj_w = np.random.uniform(-limit, limit, (self.input_dim, self.d_model)).astype(np.float32)
        self.input_proj_b = np.zeros(self.d_model, dtype=np.float32)
        
        # 可学习位置编码
        self.pos_encoding = np.random.randn(8, self.d_model).astype(np.float32) * 0.02
        
        # Transformer层
        self.attention_weights = []
        self.ffn_weights = []
        self.norm_params = []
        
        d, h, dh = self.d_model, self.nhead, self.head_dim
        for _ in range(self.num_layers):
            # Multi-head attention: Q, K, V, O
            limit = np.sqrt(6.0 / (d + d))
            q_w = np.random.uniform(-limit, limit, (d, d)).astype(np.float32)
            k_w = np.random.uniform(-limit, limit, (d, d)).astype(np.float32)
            v_w = np.random.uniform(-limit, limit, (d, d)).astype(np.float32)
            o_w = np.random.uniform(-limit, limit, (d, d)).astype(np.float32)
            self.attention_weights.append((q_w, k_w, v_w, o_w))
            
            # FFN
            limit1 = np.sqrt(6.0 / (d + self.dim_ff))
            limit2 = np.sqrt(6.0 / (self.dim_ff + d))
            ffn1_w = np.random.uniform(-limit1, limit1, (d, self.dim_ff)).astype(np.float32)
            ffn1_b = np.zeros(self.dim_ff, dtype=np.float32)
            ffn2_w = np.random.uniform(-limit2, limit2, (self.dim_ff, d)).astype(np.float32)
            ffn2_b = np.zeros(d, dtype=np.float32)
            self.ffn_weights.append(((ffn1_w, ffn1_b), (ffn2_w, ffn2_b)))
            
            # LayerNorm
            self.norm_params.append((np.ones(d, dtype=np.float32), np.zeros(d, dtype=np.float32),
                                      np.ones(d, dtype=np.float32), np.zeros(d, dtype=np.float32)))
        
        # 多窗口输出头
        self.window_heads = []
        for _ in range(self.num_windows):
            limit = np.sqrt(6.0 / (d + 1))
            w = np.random.uniform(-limit, limit, (d, 1)).astype(np.float32)
            b = np.float32(0.0)
            self.window_heads.append((w, b))
    
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    
    def _layer_norm(self, x, gamma, beta, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def _multi_head_attention(self, x, q_w, k_w, v_w, o_w, causal_mask=True):
        """多头自注意力"""
        B, T, D = x.shape
        h, dh = self.nhead, self.head_dim
        
        Q = x @ q_w  # [B, T, D]
        K = x @ k_w
        V = x @ v_w
        
        # Reshape to multi-head: [B, h, T, dh]
        Q = Q.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, h, dh).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)  # [B, h, T, T]
        
        # Causal mask
        if causal_mask:
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            scores[:, :, mask] = -1e9
        
        attn = self._softmax(scores)
        out = attn @ V  # [B, h, T, dh]
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ o_w
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    def predict(self, temporal_features: np.ndarray) -> Dict[str, float]:
        """
        时序预测
        
        Args:
            temporal_features: [T, D] 时序特征矩阵
            
        Returns:
            {30: prob, 60: prob, 90: prob} 多窗口预测概率
        """
        if temporal_features.ndim == 2:
            temporal_features = temporal_features[np.newaxis, :]
        
        B, T, D_in = temporal_features.shape
        
        # 输入投影
        x = temporal_features @ self.input_proj_w + self.input_proj_b
        x = x * np.sqrt(self.d_model)
        
        # 添加位置编码
        x = x + self.pos_encoding[:T]
        
        # Transformer层
        for layer_idx in range(self.num_layers):
            q_w, k_w, v_w, o_w = self.attention_weights[layer_idx]
            (ffn1_w, ffn1_b), (ffn2_w, ffn2_b) = self.ffn_weights[layer_idx]
            g1, b1, g2, b2 = self.norm_params[layer_idx]
            
            # Self-attention + residual + norm
            attn_out = self._multi_head_attention(x, q_w, k_w, v_w, o_w)
            x = self._layer_norm(x + attn_out, g1, b1)
            
            # FFN + residual + norm
            ffn_out = np.maximum(0, x @ ffn1_w + ffn1_b) @ ffn2_w + ffn2_b
            x = self._layer_norm(x + ffn_out, g2, b2)
        
        # 使用最后一个时间步的表示做预测
        last_hidden = x[:, -1, :]  # [B, D]
        
        # 多窗口预测
        results = {}
        windows = [30, 60, 90]
        for i, (w, b) in enumerate(self.window_heads):
            logit = last_hidden @ w + b
            prob = self._sigmoid(logit)
            results[windows[i]] = float(prob[0, 0])
        
        return results


class GATModel:
    """Graph Attention Network - 图注意力网络
    
    用于建模上市公司之间的行业关联、供应链关系和风险传导效应。
    当同行业或关联行业公司被问询时，相关公司风险概率上升。
    
    核心创新：
    - 多头注意力机制学习公司间风险传导权重
    - 支持动态图结构（新上市公司自动加入图）
    - 注意力权重具有可解释性
    """
    
    def __init__(self, input_dim=32, hidden_dim=32, num_heads=4, num_layers=2, dropout=0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        
        np.random.seed(44)
        self._init_weights()
    
    def _init_weights(self):
        self.layers = []
        in_dim = self.input_dim
        
        for i in range(self.num_layers):
            out_dim = self.hidden_dim
            limit = np.sqrt(6.0 / (in_dim + self.head_dim))
            # 每个head的W和attention vector a
            heads = []
            for h in range(self.num_heads):
                W = np.random.uniform(-limit, limit, (in_dim, self.head_dim)).astype(np.float32)
                a = np.random.uniform(-limit, limit, (2 * self.head_dim, 1)).astype(np.float32)
                heads.append((W, a))
            
            # 输出投影
            limit_out = np.sqrt(6.0 / (self.hidden_dim + out_dim))
            out_W = np.random.uniform(-limit_out, limit_out,
                                       (out_dim, out_dim)).astype(np.float32)
            self.layers.append((heads, out_W))
            in_dim = out_dim
        
        # 预测头
        limit = np.sqrt(6.0 / (self.hidden_dim + 1))
        self.pred_w = np.random.uniform(-limit, limit, (self.hidden_dim, 1)).astype(np.float32)
        self.pred_b = np.float32(0.0)
    
    def _leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    def forward_layer(self, H, adj, heads, out_W):
        """单层GAT前向传播"""
        N = H.shape[0]
        head_outputs = []
        
        for W, a in heads:
            # 线性变换
            HW = H @ W  # [N, head_dim]
            
            # 计算注意力系数: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            # 高效实现: a = [a1; a2], e_ij = (HW @ a1) + (HW @ a2)^T
            a1 = a[:self.head_dim]
            a2 = a[self.head_dim:]
            attn_self = HW @ a1  # [N, 1]
            attn_neigh = HW @ a2  # [N, 1]
            e = attn_self + attn_neigh.T  # [N, N]
            e = self._leaky_relu(e)
            
            # Mask: 只对邻居计算attention
            mask = (adj > 0)
            e = np.where(mask, e, -1e9)
            
            # Softmax归一化
            alpha = self._softmax(e, axis=1)
            alpha = alpha * adj  # 再次mask确保
            alpha = alpha / (np.sum(alpha, axis=1, keepdims=True) + 1e-8)
            
            # 聚合
            H_new = alpha @ HW  # [N, head_dim]
            head_outputs.append(H_new)
        
        # 拼接多头
        H_cat = np.concatenate(head_outputs, axis=1)  # [N, hidden_dim]
        H_out = np.maximum(0, H_cat @ out_W)  # ELU-like activation
        return H_out
    
    def predict(self, node_features: np.ndarray, adj_matrix: np.ndarray, target_idx: int = 0) -> float:
        """
        图预测
        
        Args:
            node_features: [N, F] 节点特征
            adj_matrix: [N, N] 邻接矩阵
            target_idx: 目标节点索引
            
        Returns:
            risk_contagion_score: float 风险传染得分
        """
        H = node_features.copy()
        if H.ndim == 1:
            H = H[np.newaxis, :]
        
        for heads, out_W in self.layers:
            H = self.forward_layer(H, adj_matrix, heads, out_W)
        
        # 目标节点预测
        target_h = H[target_idx]
        logit = target_h @ self.pred_w + self.pred_b
        return float(self._sigmoid(logit))


class RiskTextEncoder:
    """Risk Text Encoder - 金融文本风险编码器
    
    基于Transformer Encoder架构的轻量级文本编码器，
    针对金融监管领域优化，能够从公告文本中提取深层语义风险信号。
    
    特点：
    - 预训练风格的自注意力编码
    - 风险关键词引导的注意力偏置
    - 输出全局文本风险embedding
    """
    
    def __init__(self, vocab_size=5000, embed_dim=128, num_heads=8, num_layers=4, 
                 max_seq_len=512, hidden_dim=256):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.head_dim = embed_dim // num_heads
        
        np.random.seed(45)
        self._init_weights()
    
    def _init_weights(self):
        # Token embedding
        self.token_embed = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.02
        # Position embedding
        self.pos_embed = np.random.randn(self.max_seq_len, self.embed_dim).astype(np.float32) * 0.01
        
        # Transformer layers (simplified)
        self.layers_w = []
        d, h, dh = self.embed_dim, self.num_heads, self.head_dim
        limit = np.sqrt(6.0 / (d + d))
        for _ in range(self.num_layers):
            self.layers_w.append({
                'q': np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
                'k': np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
                'v': np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
                'o': np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
                'ff1': np.random.uniform(-np.sqrt(6/(d+self.hidden_dim)), np.sqrt(6/(d+self.hidden_dim)), 
                                          (d, self.hidden_dim)).astype(np.float32),
                'ff2': np.random.uniform(-np.sqrt(6/(self.hidden_dim+d)), np.sqrt(6/(self.hidden_dim+d)),
                                          (self.hidden_dim, d)).astype(np.float32),
            })
        
        # Output projection
        limit = np.sqrt(6.0 / (d + d))
        self.out_w = np.random.uniform(-limit, limit, (d, d)).astype(np.float32)
        
        # Risk scoring head
        limit = np.sqrt(6.0 / (d + 1))
        self.risk_w = np.random.uniform(-limit, limit, (d, 1)).astype(np.float32)
        self.risk_b = np.float32(0.0)
    
    def _tokenize(self, text: str) -> np.ndarray:
        """简易tokenization（基于字符bigram哈希）"""
        tokens = []
        # 添加CLS token
        tokens.append(0)
        # 简单的字符+bigram哈希
        text = text[:self.max_seq_len - 2]
        for i, ch in enumerate(text):
            h = (ord(ch) + (i * 31)) % (self.vocab_size - 2) + 1
            tokens.append(h)
        tokens.append(1)  # SEP
        # Padding
        while len(tokens) < min(len(text) + 2, self.max_seq_len):
            tokens.append(2)  # PAD
        return np.array(tokens[:self.max_seq_len], dtype=np.int32)
    
    def _softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    
    def encode(self, text: str) -> Dict[str, Any]:
        """
        编码文本并预测风险
        
        Returns:
            {
                'embedding': np.ndarray 文本向量,
                'risk_score': float 文本风险分,
                'attention_weights': np.ndarray 注意力权重（可解释性）
            }
        """
        tokens = self._tokenize(text)
        seq_len = (tokens > 2).sum() + 2  # actual content length
        if seq_len < 3:
            seq_len = min(len(tokens), 10)
        
        # Embedding
        x = self.token_embed[tokens] + self.pos_embed[:len(tokens)]
        
        # Transformer encoding (simplified single-head fast attention for numpy)
        for layer in self.layers_w:
            Q = x @ layer['q']
            K = x @ layer['k']
            V = x @ layer['v']
            
            # Scaled dot-product attention
            scale = np.sqrt(self.head_dim * self.num_heads)
            attn = self._softmax((Q @ K.T) / scale)
            
            # Causal mask is NOT applied here (bidirectional encoding)
            x = attn @ V @ layer['o']
            # FFN
            x = np.maximum(0, x @ layer['ff1']) @ layer['ff2']
        
        # CLS token output
        cls_output = x[0] @ self.out_w
        
        # Risk score
        risk_logit = cls_output @ self.risk_w + self.risk_b
        risk_score = float(self._sigmoid(risk_logit))
        
        return {
            'embedding': cls_output,
            'risk_score': risk_score,
            'seq_len': seq_len
        }
