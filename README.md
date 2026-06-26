# 自适应超图传播动力学复现代码

这是论文 **Adaptive Epidemic Dynamics on Hypergraphs with Group-Level Immunization and Rewiring** 的独立复现代码仓库。

需要先说清楚：当前工作区和 arXiv 源文件里都没有作者原始仿真代码。因此本仓库不是“搬运作者代码”，而是根据论文正文中的模型方程、算法描述、参数设置重新实现的一套可运行复现框架。

## 仓库包含什么

- 自适应 simplicial SIS 模型的同质 MMCA 数值迭代。
- 随机均匀超图上的 Monte Carlo / 准稳态 QS 仿真。
- 三类超边层免疫策略：
  - 随机超边免疫；
  - 按感染压力排序的 targeted immunization；
  - 按活动阈值触发的 spontaneous isolation。
- 两类重连机制：
  - 随机重连；
  - 度优先重连。
- 对应论文 Fig. 2-6 风格实验的复现脚本。
- `outputs/` 下保留了一版 quick run 的 CSV 数据和 PNG 图。
- 一份中文科研分析报告：`adaptive_hypergraph_research_report.html`。

## 安装环境

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

依赖很少：核心仿真主要用 `numpy`，画图用 `matplotlib`，`networkx` 预留给后续拓扑分析和经验网络扩展。

## 快速验证

```bash
python scripts/smoke_test.py
python scripts/audit_no_figure_inputs.py
python scripts/run_all.py --quick
```

输出目录：

```text
outputs/data/      quick run 生成的 CSV 数据
outputs/figures/   quick run 生成的 PNG 图
```

## 正式复现

论文写的是 200 次独立 MC 平均。正式跑时可以去掉 `--quick`，并显式设置重复次数：

```bash
python scripts/reproduce_fig2.py --reps 200
python scripts/reproduce_fig4_fig5.py --reps 200
python scripts/reproduce_fig6.py
```

完整 MC 扫描会比较耗时，尤其是免疫策略和 HIT 扫描。

## 重要说明

- 公开仓库没有上传出版社 PDF，避免版权风险。
- 复现脚本不会读取、数字化、复制论文图中的数据点；它们从论文方程和文字参数重新计算。可运行 `python scripts/audit_no_figure_inputs.py` 检查这个约束。
- 论文 Fig. 7 使用的国会议案共署经验超图没有随论文或当前工作区公开，因此本仓库暂不复现 Fig. 7。
- 论文中的 spontaneous isolation 有一个小歧义：算法写的是候选超边按活动水平降序排序，但文字解释有时像是在说移除低活动高风险超边。本仓库默认按算法实现，并提供 `--si-sort ascending` 做敏感性检查。
- 对 Fig. 4-5，论文没有完全说明免疫是一次性、持续性还是累计性。本仓库采用“burn-in 后一次性干预”的约定，并在 `README_reproduction.md` 中记录原因。

## 目录结构

```text
src/adaptive_hypergraph/     模型、MMCA、MC、免疫和重连核心代码
scripts/                     每张图/每类实验的命令行入口
outputs/data/                quick run 的 CSV 输出
outputs/figures/             quick run 的 PNG 输出
docs/original_request.md     原始研究需求
docs/reproduction_audit.md   防止“读图复现”的审计说明
README_reproduction.md       更细的复现说明和约定
```
