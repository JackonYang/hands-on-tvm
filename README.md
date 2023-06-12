# Hands on TVM

## Step 0: TVM Overview

[冯思远-Learning-based Deep Learning Compiler](https://www.bilibili.com/video/BV1T5411W7o8/): b 站视频。

30 分钟，讲了为什么要做 TVM，& AutoTVM 的基本原理。

简明扼要，新手友好。质量很高。

冯思远是 TVM 的核心贡献者之一，上交大毕业。这个视频是给上交 2020 级的学生讲的。

## Step 1: 陈天奇 - MLC 课程 Jupyter notebooks

### 对原 notebook 的改修

修改后的 code：[陈天奇 - MLC 课程 jupyter notebooks](mlc-ai-notebooks)

改动点：

1. 基于 tvm 仓库的代码。原课程基于 mlc-ai-nightly 库，虽然稳定，但与最新的 tvm API 不一样。
2. 精简注释 & 代码。
    - 目标：要点突出，适合复习、快速查阅代码。
    - 原课程，细节详细，更适合萌新学习 tvm。

备注：

1. 课程使用 relax，还未合入 tvm 主分支。实测，切到 unity 分支后，可用。
2. 代码走读发现，unity 分支，主要就是新增了 relax 相关的 5 万行代码。

### 推荐 3 个 notebook

[2_tensor_program_abstraction.ipynb](mlc-ai-notebooks/2_tensor_program_abstraction.ipynb)

1. 用 TVM 的 2 种 DSL 写算子。
    - TVMScript 的 TIR 写 矩阵加法
    - Tensor Expression（简称 TE）写矩阵乘法
2. 程序优化程序。tvm scheduler 把矩阵乘的性能提升 ~10+ 倍。

[4_Build_End_to_End_Model.ipynb](mlc-ai-notebooks/4_Build_End_to_End_Model.ipynb)

1. 用 TVM 手写模型。用到了 Relax 的 2 个 API: Dataflow, call_dps_packed / call_tir
2. 可以注册外部算子，并在模型里混合使用。使用 call_dps_packed 调用外部算子。
3. 把模型权重 bind 到 IRModule 上，运行时不用再传入模型权重。

[5_Automatic_Program_Optimization.ipynb](mlc-ai-notebooks/5_Automatic_Program_Optimization.ipynb)

1. 这是 tvm 的精髓：autotvm & ansor。
2. 使用 `meta_schedule.tune_tir` API，可以自动搜索最优的 schedule。
3. 调用 `tune_tir` 时，可以手动指定 search space，但，如果写的太简单了，效果很可能不如默认版本。
4. 可以把 Model 里的特定算子拿出来，单独调用 `tune_tir` 优化，并把优化后的算子，放回 Model。

备注：

1. 原 notebook 花了大量篇幅讲采样搜索的最优化方法。实际，对应的 TVM API 很简单。熟悉最优化理论的，直接看 API 就够了。
2. 搜索算法，依赖 xgboost。 `pip install xgboost`

### 其他章节

一共 10 节课。1 和 10 都是 ppt，没有 code。

其他章节的 code，个人看法如下

[3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action.ipynb](mlc-ai-notebooks/3_TensorIR_Tensor_Program_Abstraction_Case_Study_Action.ipynb)

1. 针对高性能计算新手的科普。原课程，花了大量篇幅讲解，为什么要 tiling, reorder。都是与 TVM 关系不大的背景知识。
2. TVM API 的使用，基本是课程 2 的翻版。用的也是 TVMScript (TIR)，简单讲了 TE。新东西不多。
3. 特点是，例子是深度学习模型里的基础结构：线性层 + ReLU 激活。

整体来讲，信息量不大，跳过。暂不细致整理。

[6_Integration_with_Machine_Learning_Frameworks.ipynb](mlc-ai-notebooks/6_Integration_with_Machine_Learning_Frameworks.ipynb)

1. 个人理解，属于 TVM 定制开发，而且是针对 relax 前端的。普通用户不需要。
2. 现在用的都是 relay，前端的算子注册跟 relax 也不一样。学会了也没啥用。
3. 如果是新手，通读一下，了解图前端如何转模型和算子的，还不错。
4. 注意：高性能算子的开发和注册，跟这个不一样。要写 Tensor Expression 和 TVMScript，见 2、3 节。

[7_GPU_and_Specialized_Hardware.ipynb](mlc-ai-notebooks/7_GPU_and_Specialized_Hardware.ipynb)

1. 值得看。讲了 TVM 如何支持 GPU。
2. 花了大量篇幅介绍 GPU。实际 TVM API 相关的，内容应该很少。
3. 手头的 GPU 环境没配，先跳过。

[8_GPU_and_Specialized_Hardware_part2.ipynb](mlc-ai-notebooks/8_GPU_and_Specialized_Hardware_part2.ipynb)

1. 讲的是 NPU 和 FPGA。
2. 个人的经验，坑还是很多的，而且有些坑是 TVM 架构难以处理的。完全不像陈天奇说的这么简单。
3. 华为海思使用 TVM，据说是一二百人做了两三年才开始产生商业价值。

[9_Computational_Graph_Optimization.ipynb](mlc-ai-notebooks/9_Computational_Graph_Optimization.ipynb)

1. 讲的主要是如何用 pattern match and rewrite 做算子融合。
2. 此处的算子融合，还是基于规则的。信息量不大。
3. pattern match and rewrite 是编译器的基本功。没做过编译器的话，可以练练手。

## Step 2: 用 TVM 跑模型 - 官方 docs 整理

说明：

1. 代码均来自 TVM 官方文档的 [User Tutorial](https://tvm.apache.org/docs/tutorial/index.html) 和 [How To Guides](https://tvm.apache.org/docs/how_to/index.html)
2. 稍作修改和整理，目标，与 step 1 一样：要点突出，适合复习、快速查阅代码。
3. 原文档，细节详细，更适合萌新学习 tvm。


[01-build-run-dl-models-tvm.ipynb](01-build-run-dl-models-tvm.ipynb)

茴香豆的茴有几种写法? -- 跟 TVM 编译模型的方法一样多。

1. TVM build & run 模型，有多种 API 组合可选。粗读代码发现，一套 API 组合对应一套轮子。
2. 功能基本一样，用起来也基本一样。我感觉，掌握一套即可。
3. 我选了 `relay.build` + `tvm.contrib.graph_executor`，可以 `export_library` API 把编译后的模型保存为 `.so` 文件，方便 deploy 分发。

TODO:

1. Auto-Tune with Templates and AutoTVM
2. Use AutoScheduler for Template-Free Scheduling
3. dynamic shape, relax, unity & Nimble
4. https://github.com/mlc-ai/mlc-llm
5. TVM Papers: [https://tvm.apache.org/docs/reference/publications.html](https://tvm.apache.org/docs/reference/publications.html)

optional:

1. microTVM: TVM on bare-metal
2. VTA: Versatile Tensor Accelerator
