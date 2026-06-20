---
title: "推理引擎拆解 01：从一次请求看完整链路（调度、KV、并行与可观测）"
description: "用“请求进入→token产出”的视角拆解推理引擎核心模块：router、scheduler、worker、KV 管理、并行通信与指标。"
published: 2026-06-17
slug: "inference-engine-architecture"
date: 2026-04-07 00:00:00+0800
categories:
  - AI Infra
series:
  - 推理引擎拆解
tags:
  - LLM Serving
  - Scheduler
  - Batching
  - Observability
draft: true
---

## 这篇文章解决什么问题
- 你能画出“请求进入 → token 流式输出”的全链路架构图，并解释每一段的职责
- 你能把性能与稳定性问题落到“指标 → 定位点 → 常见原因 → 可行修复”
- 你能把可观测、压测与容量规划当成系统的一等公民

## 一条请求的生命周期（建议配一张时序图）
1) API / Gateway 接入（鉴权、限流、队列）
2) Router（多实例、多模型路由）
3) Scheduler（batching、优先级、SLA、抢占）
4) Model Worker（执行 prefill/decode）
5) KV / Memory Manager（分配、回收、碎片治理）
6) Stream 输出（SSE/WebSocket/GRPC streaming）

## 模块拆解（按职责写清“边界”和“输入输出”）
### API / Gateway
- **职责**：鉴权、限流、配额、多租户隔离、审计
- **输入**：prompt/messages、采样参数（temperature/top_p/top_k）、max_new_tokens、stop
- **输出**：标准化后的推理请求（带请求 id、SLA/优先级、租户信息）
- **常见坑**：长文本/超大上下文导致资源异常；参数组合导致输出不可控

### Router（路由层）
- **职责**：多模型/多版本/多实例的选择；健康检查；熔断与回退
- **策略**：round-robin、least-loaded、基于 KV/显存的 capacity-aware、基于 SLA 的优先级路由
- **常见坑**：健康检查不可靠导致抖动；回退策略引发质量不一致

### Scheduler（调度层，核心）
- **职责**：把请求组织成可执行的 batch，决定 prefill/decode 的执行顺序
- **输入**：队列中的请求（带上下文长度、已生成长度、SLA）
- **输出**：一个个执行步（prefill batch / decode step batch）

### Worker（执行层）
- **职责**：模型加载、权重管理、执行 prefill/decode、流式输出 token
- **边界**：tokenizer 是否在 worker（CPU）还是在 gateway；stream 回传路径
- **常见坑**：CPU tokenizer 成瓶颈；流式回传阻塞导致尾延迟放大

### KV / Memory Manager（显存与缓存）
- **职责**：KV cache 的分配/回收/上限控制；碎片治理；（可选）paged KV
- **常见坑**：碎片化引发“还有显存但分配失败”；长上下文请求把并发打穿

### Communication（多卡/多机）
- **职责**：TP/PP 同步与通信；对拓扑/带宽/同步点敏感
- **常见坑**：同步点导致 tail latency；跨机网络抖动把 TPOT 放大

## 从第一天就把指标挂上（把“可观测”变成结构的一部分）
- 请求侧：QPS、并发、排队时间、超时率
- 体验侧：TTFT、TPOT、尾延迟（p95/p99）
- 资源侧：GPU 利用率、显存（weights/KV/fragmentation）、CPU/网络
- 引擎侧：batch size 分布、prefill/decode 占比、KV 命中/回收

## Scheduler：知识库最值得写的核心
### batching 的几种形态
- **static batching**：固定时间窗口聚合，简单但吞吐/延迟折中较差
- **continuous batching**：主流方案，吞吐高但实现复杂（队列/抢占/混排）
- **microbatch/pipeline**：配合 TP/PP 的并行策略（影响实现与稳定性）

### prefill/decode 混排（决定 TTFT 与吞吐）
- **prefill 优先**：TTFT 好，但 decode 可能抖
- **decode 优先**：TPOT 好，但新请求 TTFT 可能变差
- **分层队列**：按上下文长度/优先级/SLA 分队列，降低长尾

### fairness 与 tail latency（为什么工程里“很难”）
- 长上下文/大输出请求会“霸占”资源
- 常用手段：配额、抢占、最大上下文限制、分级服务（SLA tiers）

## KV 与显存：为什么“能跑”≠“能稳”
- KV cache 的预算与上限控制
- OOM 发生点与定位（prefill、decode、allocator）

### 容量规划的最小模型（先给读者一个算账框架）
- 变量：并发 \(C\)、上下文 \(L\)、输出 \(T\)、层数 \(N\)、heads/head_dim、dtype
- 目标：估算安全并发、最大上下文、最大输出 token 的边界
- 产出：把“显存 OOM”从玄学变成可预测的约束

## 多卡与通信（先给读者一个地图）
- TP/PP 的基本开销
- NCCL 相关：带宽、拓扑、同步点

## 可观测性：必须从第一天就有
- 关键指标：TTFT/TPOT/tokens/s/queue latency/KV usage
- tracing：一次请求跨 router/scheduler/worker
- profiling：GPU 利用率与 kernel 级别热点

## 调试闭环：从“慢/抖/炸”到定位点
- TTFT 高：prefill 变慢？队列堆积？tokenizer/网络？
- TPOT 高：decode kernel/通信/小 batch？KV 访存？
- OOM：KV 预算、碎片、并发上限、异常请求（超长上下文）

## 压测与回归（把“上线前”变成流程）
- **负载模型**：短请求/长请求、不同 temperature、不同 max_new_tokens、不同上下文分布
- **关键曲线**：并发-吞吐、并发-尾延迟、KV 使用-时间、OOM 边界
- **回归策略**：模型升级/驱动升级/引擎升级的 A/B 对比基线与报警阈值

## 安全与成本（工程必须面对）
- 多租户隔离：配额、限流、审计、资源上限
- 成本口径：tokens/$、GPU-hour 利用率、cache 命中带来的节省

## 选型导读：vLLM / TensorRT-LLM / 自研
- 你未来文章会展开的对比维度列表

## 推荐的后续文章拆分（保证你能持续写下去）
- 02：Continuous batching 与 prefill/decode 混排（含策略对比与指标）
- 03：KV 管理（paged KV、碎片、OOM 定位与治理）
- 04：多卡通信与拓扑（NCCL 同步点、tail latency、overlap）
- 05：可观测与压测体系（从 metrics/tracing 到 kernel profiling）

## 小结
- 用 3–5 条 bullet 总结“推理引擎的关键复杂度来自哪里”

