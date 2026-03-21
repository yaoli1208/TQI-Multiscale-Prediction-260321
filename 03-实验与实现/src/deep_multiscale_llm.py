"""
深度多尺度时序大模型 - 核心实现
=============================================
Deep Multiscale Temporal LLM for TQI Prediction

架构:
  Layer 1: STL分解 (趋势+季节+残差)
  Layer 2: 深度学习特征提取 (LSTM/Transformer/CNN)
  Layer 3: 大模型融合 (Lag-Llama/Time-LLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ============================================================================
# Layer 1: 多尺度分解层
# ============================================================================

class STLDecomposition(nn.Module):
    """STL分解模块（可微分近似版本）"""
    
    def __init__(self, period: int = 26):
        """
        Args:
            period: 季节周期（TQI检测周期，26个检测点≈1年）
        """
        super().__init__()
        self.period = period
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, 1] 输入TQI序列
            
        Returns:
            trend: 趋势成分
            seasonal: 季节成分
            residual: 残差成分
        """
        batch, seq_len, _ = x.shape
        
        # 1. 趋势：移动平均
        trend = self._moving_average(x, window=self.period)
        
        # 2. 去趋势
        detrended = x - trend
        
        # 3. 季节：周期平均
        seasonal = self._extract_seasonal(detrended, period=self.period)
        
        # 4. 残差
        residual = detrended - seasonal
        
        return trend, seasonal, residual
    
    def _moving_average(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """计算移动平均"""
        # 使用1D卷积实现移动平均
        kernel = torch.ones(1, 1, window, device=x.device) / window
        x_padded = F.pad(x.transpose(1, 2), (window//2, window//2), mode='replicate')
        trend = F.conv1d(x_padded, kernel, stride=1).transpose(1, 2)
        # 调整长度匹配
        if trend.shape[1] != x.shape[1]:
            trend = F.interpolate(trend.transpose(1, 2), size=x.shape[1], 
                                 mode='linear', align_corners=False).transpose(1, 2)
        return trend
    
    def _extract_seasonal(self, x: torch.Tensor, period: int) -> torch.Tensor:
        """提取季节成分"""
        batch, seq_len, _ = x.shape
        seasonal = torch.zeros_like(x)
        
        for i in range(period):
            mask = torch.arange(seq_len, device=x.device) % period == i
            if mask.sum() > 0:
                seasonal[:, mask, :] = x[:, mask, :].mean(dim=1, keepdim=True)
        
        return seasonal


# ============================================================================
# Layer 2: 深度学习特征层
# ============================================================================

class TrendBranch(nn.Module):
    """趋势分支：BiLSTM + Attention"""
    
    def __init__(
        self, 
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )
        
        # 输出投影
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] 趋势序列
            
        Returns:
            features: [batch, seq_len, output_dim]
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接 + 输出投影
        features = self.fc(lstm_out + attn_out)
        
        return features


class SeasonalBranch(nn.Module):
    """季节分支：Transformer + 位置编码"""
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_dim: int = 64
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码（学习周期信息）
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] 季节序列
        """
        # 输入投影
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        trans_out = self.transformer(x)
        
        # 输出投影
        features = self.fc(trans_out)
        
        return features


class ResidualBranch(nn.Module):
    """残差分支：1D-CNN + Gating"""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list = [32, 64, 128],
        output_dim: int = 32
    ):
        super().__init__()
        
        # CNN层
        layers = []
        in_channels = input_dim
        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] 残差序列
        """
        # 调整维度为 [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # CNN特征提取
        conv_out = self.conv_layers(x)  # [batch, hidden, seq_len]
        
        # 调整回来
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, hidden]
        
        # 门控
        gate = self.gate(conv_out)
        gated_out = conv_out * gate
        
        # 输出投影
        features = self.fc(gated_out)
        
        return features


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# Layer 3: 大模型融合层（简化版，实际需集成Lag-Llama/Time-LLM）
# ============================================================================

class LLMFusionLayer(nn.Module):
    """大模型融合层 - 简化实现"""
    
    def __init__(
        self,
        trend_dim: int = 128,
        seasonal_dim: int = 64,
        residual_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 1,
        pred_len: int = 26  # 预测未来26个检测点
    ):
        super().__init__()
        
        self.pred_len = pred_len
        total_dim = trend_dim + seasonal_dim + residual_dim
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 自回归解码器（模拟大模型的生成能力）
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.2
        )
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 各分量输出
        self.trend_proj = nn.Linear(hidden_dim, output_dim)
        self.seasonal_proj = nn.Linear(hidden_dim, output_dim)
        self.residual_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self, 
        feat_trend: torch.Tensor,
        feat_seasonal: torch.Tensor,
        feat_residual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_trend: [batch, seq_len, 128]
            feat_seasonal: [batch, seq_len, 64]
            feat_residual: [batch, seq_len, 32]
            
        Returns:
            predictions dict
        """
        batch, seq_len, _ = feat_trend.shape
        
        # 1. 拼接多尺度特征
        multiscale_feat = torch.cat([
            feat_trend, feat_seasonal, feat_residual
        ], dim=-1)  # [batch, seq_len, 224]
        
        # 2. 投影到隐藏空间
        hidden = self.projection(multiscale_feat)  # [batch, seq_len, hidden_dim]
        
        # 3. 自回归生成预测（模拟大模型）
        # 取最后时刻的隐藏状态作为初始输入
        decoder_input = hidden[:, -1:, :]  # [batch, 1, hidden_dim]
        
        predictions = []
        hidden_state = None
        
        for _ in range(self.pred_len):
            out, hidden_state = self.decoder(decoder_input, hidden_state)
            predictions.append(out)
            decoder_input = out  # 自回归
        
        pred_sequence = torch.cat(predictions, dim=1)  # [batch, pred_len, hidden_dim]
        
        # 4. 生成总预测
        pred_total = self.output_proj(pred_sequence)
        
        # 5. 生成各分量预测
        pred_trend = self.trend_proj(pred_sequence)
        pred_seasonal = self.seasonal_proj(pred_sequence)
        pred_residual = self.residual_proj(pred_sequence)
        
        return {
            'total': pred_total,
            'trend': pred_trend,
            'seasonal': pred_seasonal,
            'residual': pred_residual
        }


# ============================================================================
# 完整模型
# ============================================================================

class DeepMultiscaleTemporalLLM(nn.Module):
    """深度多尺度时序大模型 - 完整实现"""
    
    def __init__(
        self,
        period: int = 26,
        pred_len: int = 26,
        trend_dim: int = 128,
        seasonal_dim: int = 64,
        residual_dim: int = 32
    ):
        super().__init__()
        
        # Layer 1: 分解
        self.decomposition = STLDecomposition(period=period)
        
        # Layer 2: 深度学习特征
        self.trend_branch = TrendBranch(input_dim=1, output_dim=trend_dim)
        self.seasonal_branch = SeasonalBranch(input_dim=1, output_dim=seasonal_dim)
        self.residual_branch = ResidualBranch(input_dim=1, output_dim=residual_dim)
        
        # Layer 3: 大模型融合
        self.fusion = LLMFusionLayer(
            trend_dim=trend_dim,
            seasonal_dim=seasonal_dim,
            residual_dim=residual_dim,
            pred_len=pred_len
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, 1] 输入TQI序列
            
        Returns:
            predictions: {
                'total': [batch, pred_len, 1],
                'trend': [batch, pred_len, 1],
                'seasonal': [batch, pred_len, 1],
                'residual': [batch, pred_len, 1]
            }
        """
        # Layer 1: 分解
        trend, seasonal, residual = self.decomposition(x)
        
        # Layer 2: 特征提取
        feat_trend = self.trend_branch(trend)
        feat_seasonal = self.seasonal_branch(seasonal)
        feat_residual = self.residual_branch(residual)
        
        # Layer 3: 融合预测
        predictions = self.fusion(feat_trend, feat_seasonal, feat_residual)
        
        return predictions


# ============================================================================
# 损失函数
# ============================================================================

class MultiscaleLoss(nn.Module):
    """多尺度联合损失函数"""
    
    def __init__(
        self,
        alpha: float = 1.0,      # 总损失权重
        beta: float = 0.3,       # 分量损失权重
        gamma: float = 0.2       # 一致性损失权重
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        target: torch.Tensor,
        target_components: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: 模型预测输出
            target: 真实值 [batch, pred_len, 1]
            target_components: 真实分解成分（可选）
        """
        # 总预测损失
        loss_total = F.mse_loss(predictions['total'], target)
        
        # 分量损失（如果有监督）
        loss_components = 0
        if target_components is not None:
            loss_trend = F.mse_loss(predictions['trend'], target_components['trend'])
            loss_seasonal = F.mse_loss(predictions['seasonal'], target_components['seasonal'])
            loss_residual = F.mse_loss(predictions['residual'], target_components['residual'])
            loss_components = loss_trend + loss_seasonal + loss_residual
        
        # 一致性损失：确保各分量之和等于总预测
        pred_sum = predictions['trend'] + predictions['seasonal'] + predictions['residual']
        loss_consistency = F.mse_loss(pred_sum, predictions['total'])
        
        # 总损失
        total_loss = (self.alpha * loss_total + 
                     self.beta * loss_components + 
                     self.gamma * loss_consistency)
        
        return total_loss


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试模型
    print("测试深度多尺度时序大模型...")
    
    batch_size = 4
    seq_len = 52  # 回溯1年
    pred_len = 26  # 预测半年
    
    # 创建模型
    model = DeepMultiscaleTemporalLLM(
        period=26,
        pred_len=pred_len,
        trend_dim=128,
        seasonal_dim=64,
        residual_dim=32
    )
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, 1)
    
    # 前向传播
    predictions = model(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"总预测形状: {predictions['total'].shape}")
    print(f"趋势预测形状: {predictions['trend'].shape}")
    print(f"季节预测形状: {predictions['seasonal'].shape}")
    print(f"残差预测形状: {predictions['residual'].shape}")
    
    # 测试损失
    target = torch.randn(batch_size, pred_len, 1)
    criterion = MultiscaleLoss()
    loss = criterion(predictions, target)
    print(f"\n损失值: {loss.item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}")
    
    print("\n✓ 模型测试通过！")
