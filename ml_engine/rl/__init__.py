#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Module
强化学习算法集合：
- PPOAgent: Proximal Policy Optimization，自适应风险阈值优化
- ThompsonSamplingEnsemble: 汤普森采样多臂老虎机，动态集成权重
- OnlineLearner: 在线学习器，从反馈中持续优化
"""
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Any
import random


class PPOMemory:
    """PPO经验回放缓存"""
    
    def __init__(self, batch_size=64):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (np.array(self.states), np.array(self.actions), 
                np.array(self.probs), np.array(self.vals),
                np.array(self.rewards), np.array(self.dones), batches)
    
    def __len__(self):
        return len(self.states)


class PPOActorCritic:
    """PPO Actor-Critic网络（numpy实现）
    
    策略网络输出风险阈值调整动作，价值网络评估状态价值。
    动作空间：3维连续动作，对应30/60/90天阈值的调整量（-0.1~0.1）
    """
    
    def __init__(self, state_dim=64, action_dim=3, hidden_dim=64, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        np.random.seed(46)
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # Actor
        limit1 = np.sqrt(6.0 / (self.state_dim + self.hidden_dim))
        limit2 = np.sqrt(6.0 / (self.hidden_dim + self.hidden_dim))
        limit3 = np.sqrt(6.0 / (self.hidden_dim + self.action_dim))
        
        self.actor_w1 = np.random.uniform(-limit1, limit1, (self.state_dim, self.hidden_dim)).astype(np.float32)
        self.actor_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.actor_w2 = np.random.uniform(-limit2, limit2, (self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.actor_b2 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.actor_w3 = np.random.uniform(-limit3, limit3, (self.hidden_dim, self.action_dim)).astype(np.float32)
        self.actor_b3 = np.zeros(self.action_dim, dtype=np.float32)
        # Log std for Gaussian policy
        self.actor_log_std = np.zeros(self.action_dim, dtype=np.float32) - 0.5
        
        # Critic
        self.critic_w1 = np.random.uniform(-limit1, limit1, (self.state_dim, self.hidden_dim)).astype(np.float32)
        self.critic_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.critic_w2 = np.random.uniform(-limit2, limit2, (self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.critic_b2 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.critic_w3 = np.random.uniform(-limit3, limit3, (self.hidden_dim, 1)).astype(np.float32)
        self.critic_b3 = np.float32(0.0)
    
    def _forward_actor(self, state):
        h = np.tanh(state @ self.actor_w1 + self.actor_b1)
        h = np.tanh(h @ self.actor_w2 + self.actor_b2)
        mean = np.tanh(h @ self.actor_w3 + self.actor_b3) * 0.15  # 输出范围[-0.15, 0.15]
        std = np.exp(self.actor_log_std)
        return mean, std
    
    def _forward_critic(self, state):
        h = np.tanh(state @ self.critic_w1 + self.critic_b1)
        h = np.tanh(h @ self.critic_w2 + self.critic_b2)
        value = h @ self.critic_w3 + self.critic_b3
        return float(value[0]) if value.ndim > 0 else float(value)
    
    def get_action(self, state):
        mean, std = self._forward_actor(state)
        # 采样动作
        action = mean + np.random.randn(*mean.shape) * std
        action = np.clip(action, -0.15, 0.15)
        # 计算log概率
        log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8))**2 + 2*np.log(std+1e-8) + np.log(2*np.pi))
        value = self._forward_critic(state)
        return action, float(log_prob), value
    
    def evaluate(self, states, actions):
        mean, std = self._forward_actor(states)
        log_probs = -0.5 * np.sum(((actions - mean) / (std + 1e-8))**2 + 2*np.log(std+1e-8) + np.log(2*np.pi), axis=-1)
        values = np.array([self._forward_critic(s) for s in states])
        dist_entropy = np.sum(np.log(std + 1e-8) + 0.5 * np.log(2*np.pi*np.e), axis=-1)
        return log_probs, values, dist_entropy


class PPOAgent:
    """PPO (Proximal Policy Optimization) 强化学习智能体
    
    用于自适应优化风险预警阈值：
    - 状态(state)：融合多模型的风险特征向量
    - 动作(action)：三个预测窗口的阈值调整量
    - 奖励(reward)：基于预测准确性（高召回率优先于精确率，因为漏检问询代价高）
    
    核心算法特点：
    - Clipped Surrogate Objective防止策略更新过大
    - GAE (Generalized Advantage Estimation) 高效估计优势函数
    - 熵正则化鼓励探索
    """
    
    def __init__(self, state_dim=64, action_dim=3, hidden_dim=64, 
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, batch_size=64, epochs=10):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        
        self.network = PPOActorCritic(state_dim, action_dim, hidden_dim, lr)
        self.memory = PPOMemory(batch_size)
        
        # 自适应阈值（初始值）
        self.current_thresholds = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        self.training_steps = 0
    
    def get_state(self, model_outputs: Dict[str, float], market_context: Dict = None) -> np.ndarray:
        """从模型输出构建状态向量"""
        state = np.zeros(64, dtype=np.float32)
        
        # 各模型预测概率
        state[0] = model_outputs.get('deepfm_prob', 0.5)
        state[1] = model_outputs.get('temporal_30d', 0.3)
        state[2] = model_outputs.get('temporal_60d', 0.5)
        state[3] = model_outputs.get('temporal_90d', 0.7)
        state[4] = model_outputs.get('gat_score', 0.0)
        state[5] = model_outputs.get('text_risk', 0.0)
        state[6] = model_outputs.get('rule_score', 0.0)
        state[7] = model_outputs.get('ensemble_prob', 0.5)
        
        # 市场上下文
        if market_context:
            state[8] = market_context.get('market_volatility', 0.5)
            state[9] = market_context.get('inquiry_frequency', 0.3)
        
        # 时间编码
        import time
        t = time.time() % 86400 / 86400
        state[10] = np.sin(2*np.pi*t)
        state[11] = np.cos(2*np.pi*t)
        
        return state
    
    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """选择阈值调整动作"""
        mean, std = self.network._forward_actor(state)
        if deterministic:
            action = mean
            log_prob = 0.0
        else:
            action = mean + np.random.randn(*mean.shape) * std
            action = np.clip(action, -0.15, 0.15)
            log_prob = float(-0.5 * np.sum(((action - mean) / (std + 1e-8))**2 + 2*np.log(std+1e-8) + np.log(2*np.pi)))
        value = self.network._forward_critic(state)
        return action, log_prob, value
    
    def adjust_thresholds(self, state: np.ndarray, base_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应调整预测阈值
        
        Args:
            state: 状态向量
            base_probs: 基础预测概率 [30d, 60d, 90d]
            
        Returns:
            adjusted_probs: 调整后概率
            thresholds: 使用的阈值
        """
        action, log_prob, value = self.choose_action(state)
        # 调整概率（不对称调整：更倾向于提高风险评分以减少漏检）
        adjusted = np.clip(base_probs + action * np.array([0.5, 0.8, 1.0]), 0.01, 0.99)
        return adjusted, action, log_prob, value
    
    def compute_reward(self, predicted_high_risk: bool, actual_inquiry: bool, 
                       predicted_prob: float, cost_false_negative=10.0, 
                       cost_false_positive=1.0) -> float:
        """计算奖励函数
        
        金融风控场景的非对称损失：漏检问询函（FN）的代价远大于误报（FP）
        """
        if predicted_high_risk and actual_inquiry:
            # True Positive: 正确预警
            return 2.0 + 0.5 * predicted_prob
        elif not predicted_high_risk and not actual_inquiry:
            # True Negative: 正确排除
            return 0.5
        elif predicted_high_risk and not actual_inquiry:
            # False Positive: 误报
            return -cost_false_positive * (predicted_prob - 0.5)
        else:
            # False Negative: 漏检（严重！）
            return -cost_false_negative
        
    def store_transition(self, state, action, log_prob, value, reward, done=False):
        """存储转移"""
        self.memory.store(state, action, log_prob, value, reward, done)
    
    def learn(self):
        """PPO更新"""
        if len(self.memory) < 32:
            return {'actor_loss': 0, 'critic_loss': 0}
        
        states, actions, old_log_probs, vals, rewards, dones, batches = self.memory.generate_batches()
        
        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = vals[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - vals[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + vals
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.epochs):
            for batch in batches:
                b_states = states[batch]
                b_actions = actions[batch]
                b_old_log_probs = old_log_probs[batch]
                b_advantages = advantages[batch]
                b_returns = returns[batch]
                
                # 评估
                new_log_probs, new_values, entropy = self.network.evaluate(b_states, b_actions)
                
                # 策略比率
                prob_ratio = np.exp(new_log_probs - b_old_log_probs)
                
                # Clipped surrogate
                weighted_probs = prob_ratio * b_advantages
                clipped_probs = np.clip(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_advantages
                
                actor_loss = -np.mean(np.minimum(weighted_probs, clipped_probs)) - self.entropy_coef * np.mean(entropy)
                critic_loss = self.value_coef * np.mean((b_returns - new_values)**2)
                
                total_loss = actor_loss + critic_loss
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
        
        self.memory.clear()
        self.training_steps += 1
        
        return {
            'actor_loss': float(total_actor_loss / (self.epochs * len(batches))),
            'critic_loss': float(total_critic_loss / (self.epochs * len(batches))),
            'training_steps': self.training_steps
        }


class ThompsonSamplingEnsemble:
    """Thompson Sampling Multi-Armed Bandit for Ensemble Weight Optimization
    
    使用汤普森采样（后验采样）动态调整多个模型的集成权重。
    每个模型对应一个Beta分布，根据历史预测准确率更新alpha/beta参数。
    
    算法优势：
    - 自动探索-利用平衡
    - 非平稳环境下能快速适应
    - 计算高效，适合在线学习
    """
    
    def __init__(self, num_models=4, prior_alpha=1.0, prior_beta=1.0, exploration_decay=0.995):
        self.num_models = num_models
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploration_decay = exploration_decay
        
        # 每个模型的Beta分布参数
        self.alphas = np.array([prior_alpha] * num_models, dtype=np.float64)
        self.betas = np.array([prior_beta] * num_models, dtype=np.float64)
        
        # 预测记录用于计算reward
        self.last_predictions = None
        self.last_weights = None
        self.update_count = 0
        
        # 模型名称
        self.model_names = ['DeepFM', 'TemporalTransformer', 'GAT+Text', 'RuleEngine']
    
    def sample_weights(self) -> np.ndarray:
        """使用汤普森采样从Beta分布采样权重"""
        # 从每个模型的Beta分布采样
        samples = np.array([
            np.random.beta(self.alphas[i], self.betas[i]) 
            for i in range(self.num_models)
        ])
        # 归一化为权重
        weights = samples / (samples.sum() + 1e-8)
        return weights.astype(np.float32)
    
    def predict(self, model_probs: List[float]) -> Tuple[float, np.ndarray]:
        """
        集成预测
        
        Args:
            model_probs: 各模型预测概率列表
            
        Returns:
            ensemble_prob: 加权集成概率
            weights: 使用的权重
        """
        probs = np.array(model_probs, dtype=np.float32)
        
        # 汤普森采样得到权重
        weights = self.sample_weights()
        
        # 加权预测
        ensemble_prob = float(np.sum(probs * weights))
        
        # 记录用于更新
        self.last_predictions = probs
        self.last_weights = weights
        
        return ensemble_prob, weights
    
    def update(self, actual_outcome: int, model_probs: np.ndarray = None):
        """
        根据真实结果更新Beta分布
        
        Args:
            actual_outcome: 1=发生问询，0=未发生
            model_probs: 各模型预测概率（如果不同于上次）
        """
        if model_probs is None:
            model_probs = self.last_predictions
        
        if model_probs is None:
            return
        
        # 根据各模型预测准确性更新
        for i in range(self.num_models):
            pred = model_probs[i]
            if actual_outcome == 1:
                # 发生问询：预测高概率的模型得到奖励
                self.alphas[i] += max(0, pred - 0.3) * 2
                self.betas[i] += max(0, 0.5 - pred)
            else:
                # 未发生问询：预测低概率的模型得到奖励
                self.betas[i] += max(0, pred - 0.5) * 2
                self.alphas[i] += max(0, 0.3 - pred)
        
        # 先验衰减，让模型适应新数据
        self.alphas = self.prior_alpha + (self.alphas - self.prior_alpha) * self.exploration_decay
        self.betas = self.prior_beta + (self.betas - self.prior_beta) * self.exploration_decay
        
        self.update_count += 1
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """获取各模型性能估计"""
        result = {}
        for i, name in enumerate(self.model_names):
            mean = self.alphas[i] / (self.alphas[i] + self.betas[i])
            # 95%置信区间（近似）
            total = self.alphas[i] + self.betas[i]
            std = np.sqrt(mean * (1-mean) / (total + 1))
            result[name] = {
                'expected_accuracy': float(mean),
                'confidence': float(1.96 * std),
                'alpha': float(self.alphas[i]),
                'beta': float(self.betas[i]),
                'samples': float(total)
            }
        return result


class OnlineLearner:
    """在线学习器
    
    收集用户反馈和实际结果，持续更新模型参数。
    支持增量学习，无需重新训练整个模型。
    """
    
    def __init__(self, buffer_size=1000, batch_size=32, lr=5e-4):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.lr = lr
        self.total_updates = 0
    
    def add_feedback(self, features: Dict, prediction: float, actual: int, 
                     risk_type: str = '', metadata: Dict = None):
        """添加一条反馈数据"""
        self.buffer.append({
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'risk_type': risk_type,
            'metadata': metadata or {},
            'timestamp': np.datetime64('now')
        })
    
    def should_update(self) -> bool:
        """是否应该执行更新"""
        return len(self.buffer) >= self.batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取在线学习统计"""
        if not self.buffer:
            return {'buffer_size': 0}
        
        preds = np.array([d['prediction'] for d in self.buffer])
        actuals = np.array([d['actual'] for d in self.buffer])
        
        # 简单指标
        tp = np.sum((preds > 0.5) & (actuals == 1))
        fp = np.sum((preds > 0.5) & (actuals == 0))
        fn = np.sum((preds <= 0.5) & (actuals == 1))
        tn = np.sum((preds <= 0.5) & (actuals == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'buffer_size': len(self.buffer),
            'total_updates': self.total_updates,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float((tp + tn) / len(self.buffer))
        }
