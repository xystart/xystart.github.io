---
title: "Transformer 推理：Prefill vs Decode 与 KV Cache（为什么它决定性能）"
description: "从计算图与张量形状出发，解释 prefill/decode、KV cache 的数据结构与性能瓶颈，并给出工程侧的观测指标。"
published: 2026-06-17
slug: "transformer-prefill-decode-kv-cache"
date: 2026-04-07 00:00:00+0800
categories:
  - AI Infra
series:
  - Transformer 基础
tags:
  - Transformer
  - Inference
  - KV Cache
  - Performance
draft: true
---

## 这篇文章解决什么问题
- 你能从“计算图 + 张量形状”的角度解释：为什么推理会变慢、为什么 KV Cache 必须做、以及性能瓶颈通常在哪里
- 你能把这些概念对齐到系统指标：TTFT/TPOT、吞吐、显存占用

## Transformer 速览（把概念对齐）
### 最小结构
- embedding → N 层 block → lm head
- block：(RMS)Norm + Attention + MLP（带残差）

### 注意力的核心张量
- Q/K/V：来自同一个 hidden，经不同线性层投影
- multi-head：把 hidden 分成多个 head 以并行表达

### 训练 vs 推理（不仅 infra）
- 训练：全序列并行（teacher forcing），一次算完所有 token 的 loss
- 推理：自回归生成（token-by-token），每步依赖历史上下文
- dropout、label smoothing、梯度、optimizer：训练特有；推理只保留前向

## 两阶段：Prefill vs Decode（推理视角的“计算图分解”）
### Prefill（上下文阶段）
- 输入长度 \(L\) 大、并行度高
- 主要成本：Attention \(O(L^2)\) + MLP（大 GEMM）
- 直觉：prefill 更容易吃满算力（矩阵大、并行度高）

### Decode（生成阶段）
- 每步只生成 1 个 token，但要利用全部历史上下文
- 主要成本：读历史 KV + 小算子（小 GEMM / reduce / elementwise）的 launch 与访存
- 直觉：decode 更容易被“延迟 + 带宽”支配

## KV Cache：存什么、怎么存
### 先把 shape 写清楚（建议用具体符号）
- 设：layers = \(N\)，heads = \(H\)，head_dim = \(D\)，hidden = \(H \cdot D\)
- batch = \(B\)，prefill 序列长 = \(L\)，decode 步数 = \(T\)
- KV（单层）常见形状：`[B, H, L, D]`（实现里可能会换轴）

### KV Cache 到底缓存了什么
- 对于每一层 attention：缓存历史 token 的 K、V
- 目的：decode 每一步只需要算当前 token 的 Q/K/V，然后用缓存的 K/V 做 attention，不用重复算历史

### Layout 与访问模式（为什么它影响性能）
- 你关注两个访问：
  - 写入：prefill 或 decode 时把新 K/V append 到 cache
  - 读取：decode 时按 head 读取历史 K/V 做 attention
- 不同 layout 会影响：
  - contiguous 读写与 coalescing
  - 是否容易做 paging（paged KV）
  - 与 attention kernel（FlashAttention / paged attention）的匹配程度

### KV Cache 的显存账本（写清楚“钱花在哪”）
- KV 显存规模近似：\( \text{bytes} \approx 2 \cdot N \cdot B \cdot H \cdot L \cdot D \cdot \text{dtype\_bytes} \)
- 解释为什么长上下文/高并发会把显存吃爆

## 生成质量相关（别只讲性能）
- sampling：greedy / temperature / top-p / top-k
- repetition penalty、stop tokens、max_new_tokens
- 这些会改变 decode 步数与分布，从而影响 TPOT/吞吐

## 性能瓶颈：算力 vs 带宽
### Prefill：大 GEMM 主导（更偏 compute-bound）
- attention 的 QK^T 与后续投影、MLP 的两次大 GEMM
- 常见优化关键词：tensor core、fused kernel、FlashAttention

### Decode：访存与同步点主导（更偏 memory/latency-bound）
- 读 KV、做 softmax/reduce、写输出
- 常见优化关键词：paged KV、减少 kernel launch、融合算子、overlap 通信

## 工程指标：怎么量化问题
- TTFT（time to first token）
- TPOT（time per output token）
- 吞吐：tokens/s（按请求、按 GPU、按实例）
- 显存：weights / activations / KV（分项观测）

## 把指标落到系统里：你应该在引擎里打哪些点
- 在 router/scheduler/worker 分段统计（排队、prefill、decode）
- 记录上下文长度与输出长度（对 TTFT/TPOT 解释力很强）
- KV 相关：allocated/used、回收耗时、碎片化趋势

## 常见优化方向（为后续推理引擎/CUDA铺垫）
- Continuous batching / scheduling
- Paged KV / KV cache 管理
- FlashAttention / fused kernels（减少访存与 launch）

## 延伸阅读路线（你可以在后续文章逐个展开）
- Transformer 结构细节：RoPE、SwiGLU、RMSNorm、GQA/MQA
- 推理引擎：continuous batching、prefill/decode 混排、SLA
- CUDA/Triton：RMSNorm/RoPE/attention 相关 kernel + profile

## 小结
- Prefill/Decode 的分解，是理解“为什么推理慢”和“该优化哪里”的地图
- KV Cache 是把自回归推理从“重复计算历史”变成“高效读取历史”的关键，但也带来显存与访存瓶颈

