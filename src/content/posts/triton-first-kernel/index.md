---
title: "Triton"
description: "推理链路算子（待更新）"
published: 2026-06-17
slug: "triton-first-kernel"
date: 2026-04-07 00:00:00+0800
pinned: true
priority: 3
categories:
  - AI Infra
series:
  - CUDA / Triton 实战
tags:
  - CUDA
  - Triton
  - Kernel
  - Profiling
draft: false
---

## 为什么 Triton 值得学
- 推理引擎的性能瓶颈，经常不是“模型原理”，而是“算子实现与访存”
- Triton 适合用来写/改小算子（尤其是希望做 fused 的场景）

## Triton 编程模型速通
- program / program id
- block/tile 的思维方式
- `num_warps` / `num_stages` 的直觉意义

## 选择 RMSNorm：为什么它适合作为第一篇
- 在 LLM 推理链路里出现频繁（attention/MLP 前后）
- 典型特征：reduce + elementwise，容易被访存/并行策略影响
- 很适合练“正确性→性能→迭代”的方法论

## 先把问题说清楚：RMSNorm 的公式与张量形状
- 公式：\(y = x \cdot \text{rsqrt}(\text{mean}(x^2) + \epsilon) \cdot w\)
- 典型 shape：`[batch, seq, hidden]`（推理时经常被 reshape 成 2D）

## 实现 0：最小可跑版本（正确性优先）
- grid 如何映射到 row（每个 program 处理一行 hidden）
- mask 与边界处理

## 实现 1：性能第一轮迭代（问题驱动）
- 访存：合并读取、减少重复读
- 并行：block size、`num_warps` 的选择
- 数值：epsilon 放置位置、fp16/bf16 的注意点

## 性能分析：用数据说明“为什么快/慢”
- 先定义指标：吞吐/延迟、带宽利用率、SM 利用率
- 再看工具：Nsight Systems / Nsight Compute（后续可单独开工具篇）
- 对照基线：PyTorch eager / torch.compile / 现成 fused（写清楚对比对象）

## 小结
- 你的 kernel 迭代 checklist（正确性、基线、profile、瓶颈、优化、回归）

