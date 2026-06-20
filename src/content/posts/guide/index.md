---
title: AI Infra 学习记录
description: 这个博客会记录自己的学习过程。
published: 2026-06-16
slug: getting-started
date: 2026-04-07 00:00:00+0800
categories:
    - AI Infra
series:
    - AI Infra 学习路线
tags:
    - Roadmap
    - 复盘
draft: true
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

这个博客主要记录我在 **AI Infra** 方向的学习与实践过程。

## 学习路线
- **Transformer 基础**：prefill/decode、KV cache、指标（TTFT/TPOT）怎么落到系统观测
- **推理引擎拆解**：调度与 batching、KV/显存管理、多卡通信、可观测与调试闭环
- **CUDA / Triton 实战**：从推理瓶颈出发写/改算子（例如 RMSNorm、RoPE、attention 相关）

由于自己是初学，更新博客内容可能会有误解，如有发现，欢迎沟通：`https://github.com/xystart`