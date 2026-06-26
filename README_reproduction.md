# 自适应超图传播模型复现说明

本仓库没有作者原始仿真代码。这里的代码是根据论文中的方程、算法和参数设置重新写的一套独立复现框架：

`Adaptive Epidemic Dynamics on Hypergraphs with Group-Level Immunization and Rewiring`

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

核心模型只依赖 `numpy`；绘图脚本需要 `matplotlib`；`networkx` 已加入依赖，方便后续做拓扑统计和经验网络扩展。

## 快速检查

```bash
python scripts/smoke_test.py
python scripts/audit_no_figure_inputs.py
```

其中：

- `smoke_test.py` 检查核心动力学、MC、免疫和重连能否跑通；
- `audit_no_figure_inputs.py` 检查源码是否存在读取论文图像/PDF 提取物的行为，避免“读图复现”。

## 复现脚本

`scripts/` 下的脚本都支持 `--quick`，用于快速生成一版小规模结果。正式复现时去掉 `--quick`，并把 `--reps` 提高到论文的 200 次 MC 平均。

```bash
python scripts/reproduce_fig2.py --quick
python scripts/reproduce_fig3.py --quick
python scripts/reproduce_fig4_fig5.py --quick
python scripts/reproduce_fig6.py --quick
python scripts/run_all.py --quick
```

输出会写入：

```text
outputs/data/
outputs/figures/
```

## 关键复现约定

- 本仓库不会数字化或读取论文中已经画好的图。所有结果由模型方程、算法规则和论文文字/图注给出的参数重新计算。详见 `docs/reproduction_audit.md`。
- 论文写到 QS 仿真使用 50 个历史活跃构型和概率 0.2。这里采用常见 QS 约定：一旦进入全健康吸收态，就从历史活跃构型中恢复；`0.2` 用作刷新历史池的概率。
- 论文的 spontaneous isolation 算法写的是按活动水平降序排序候选超边，但正文解释有时更像“低活动高风险超边优先”。默认实现严格跟随算法，即 `descending`；如果想检查另一种解释，可以使用 `--si-sort ascending`。
- 对 Fig. 4-5，论文没有明确说明免疫是一次性施加、持续施加，还是随时间累计施加。这里采用“先 burn-in，让超边活动完成适应；然后按策略一次性施加干预”的约定。这样 SI 策略才有非平凡意义，因为活动水平已经分化。
- 论文 Fig. 7 的国会议案共署经验超图没有在当前材料中公开，因此本仓库只给 Fig. 2-6 的合成超图复现，以及一个均匀超图 edge-list 加载接口。
