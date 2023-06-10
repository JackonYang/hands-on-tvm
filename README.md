# Hands on TVM

## Step 1: 陈天奇 - MLC 课程 Jupyter notebooks

我修改后的 code：[陈天奇 - MLC 课程 jupyter notebooks](mlc-ai-notebooks)

改动点：

1. 基于 tvm 仓库的代码。原课程基于 mlc-ai-nightly 库，虽然稳定，但跟最终要用的 tvm 并不一样。
2. 重写注释。新注释的目标群体是，简洁明了，要点突出，适合有编译器基础的学习，以及复习、快速查阅代码。原课程，细节详细，更适合编译器新手学习 tvm。
3. 课程大量使用了 relax，还未合入 tvm 主分支。实测，切到  unity 分支后，可用。通过代码走读分析，unity 分支的改动，主要就是新增了 relax 相关的 5 万行代码。
