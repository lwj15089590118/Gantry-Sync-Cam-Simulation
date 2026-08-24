# -*- coding: utf-8 -*-
"""
参数扫描批量测试（Batch Testbench）—— 项目核心产出
====================================================

实验内容
--------
1. 龙门矩阵：交叉耦合补偿 开/关 × 两轴增益失配 5% / 10% / 20% / 30%
   → 每格统计同步偏差 均值 / P95 / 最大值 (µm)，并计算补偿带来的
     P95 下降百分比。
2. 同步报警联锁演示：30% 失配、阈值收紧到 2mm 时，无补偿触发停机、
   有补偿正常完成 —— 展示真实龙门"超差即停"的安全逻辑。
3. 飞剪带速扫描：400 / 700 / 1000 / 1300 mm/s
   → 同步段速度误差 均值/最大、循环节拍（实测 vs 理论）。

输出
----
- docs/测试报告.md        （全部指标为"仿真验证值"，含免责声明）
- docs/data/gantry_matrix.csv / flying_shear_sweep.csv （原始数据）

运行方式：
    python -m testbench.batch_test      或    python testbench/batch_test.py

免责说明：本项目所有结果均为 Python 虚拟轴数值仿真所得（仿真验证值），
不涉及真实驱动器调试；但实验方法与控制思想与真实系统一致。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# 支持包运行与直接脚本运行两种方式
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from gantry.gantry import GantryController, GantryParams
    from cam.electronic_cam import FlyingShear, ShearParams
else:
    from gantry.gantry import GantryController, GantryParams
    from cam.electronic_cam import FlyingShear, ShearParams


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"


# ---------------------------------------------------------------------------
# 实验 1：龙门矩阵
# ---------------------------------------------------------------------------
@dataclass
class MatrixCell:
    """矩阵中的一个单元格结果"""

    mismatch: float
    comp: bool
    mean_um: float
    p95_um: float
    max_um: float
    samples: int
    alarm_sync: bool


def run_gantry_matrix(
    mismatches: list[float] | None = None,
    vel_mm_s: float = 300.0,
) -> list[MatrixCell]:
    """
    运行 补偿开/关 × 失配 5%~30% 矩阵，返回全部单元格结果。
    除失配与补偿开关外，其余参数固定，保证对比公平。
    """
    mismatches = mismatches or [0.05, 0.10, 0.20, 0.30]
    cells: list[MatrixCell] = []
    total = len(mismatches) * 2
    done = 0
    for comp in (False, True):          # 外层按补偿分组，便于阅读
        for mis in mismatches:
            prm = GantryParams(
                mismatch=mis,
                comp_enabled=comp,
                vel_mm_s=vel_mm_s,
            )
            ctl = GantryController(prm)
            res = ctl.run_positioning()
            st = res.stats()
            cells.append(
                MatrixCell(
                    mismatch=mis,
                    comp=comp,
                    mean_um=st["mean_um"],
                    p95_um=st["p95_um"],
                    max_um=st["max_um"],
                    samples=st["samples"],
                    alarm_sync=res.alarm_sync,
                )
            )
            done += 1
            print(f"[龙门矩阵 {done}/{total}] 失配={mis:.0%} 补偿={'开' if comp else '关'} "
                  f"P95={st['p95_um']:.1f}µm max={st['max_um']:.1f}µm")
    return cells


def _p95_improvements(cells: list[MatrixCell]) -> dict[float, float]:
    """按失配度计算：补偿使 P95 下降的百分比"""
    result: dict[float, float] = {}
    off = {c.mismatch: c.p95_um for c in cells if not c.comp}
    on = {c.mismatch: c.p95_um for c in cells if c.comp}
    for mis in sorted(off):
        if on.get(mis):
            result[mis] = (off[mis] - on[mis]) / off[mis] * 100.0
    return result


# ---------------------------------------------------------------------------
# 实验 2：同步报警联锁演示
# ---------------------------------------------------------------------------
def run_alarm_demo() -> tuple[bool, bool]:
    """
    收紧阈值到 2mm、失配 30%：
      无补偿 → 应触发同步报警停机；
      有补偿 → 应正常完成定位。
    返回 (无补偿是否报警, 有补偿是否报警)
    """
    alarm_off = alarm_on = False
    for comp in (False, True):
        prm = GantryParams(
            mismatch=0.30, comp_enabled=comp, sync_threshold_mm=2.0
        )
        res = GantryController(prm).run_positioning()
        if comp:
            alarm_on = res.alarm_sync
        else:
            alarm_off = res.alarm_sync
        print(f"[报警演示] 失配=30% 阈值=2mm 补偿={'开' if comp else '关'} → "
              f"{'触发停机' if res.alarm_sync else '正常完成'}")
    return alarm_off, alarm_on


# ---------------------------------------------------------------------------
# 实验 3：飞剪带速扫描
# ---------------------------------------------------------------------------
@dataclass
class ShearRow:
    """飞剪扫描中的一行结果"""

    belt_speed: float
    cuts: int
    err_mean: float
    err_max: float
    cycle_mean_ms: float
    cycle_theory_ms: float
    tables_used: str


def run_shear_sweep(belt_speeds: list[float] | None = None) -> list[ShearRow]:
    """不同带速下运行飞剪，统计同步段速度误差与循环节拍"""
    belt_speeds = belt_speeds or [400.0, 700.0, 1000.0, 1300.0]
    rows: list[ShearRow] = []
    for v in belt_speeds:
        fs = FlyingShear(ShearParams(belt_speed_mm_s=v, sim_cycles=4))
        m = fs.run().metrics
        rows.append(
            ShearRow(
                belt_speed=v,
                cuts=m["cuts"],
                err_mean=m["sync_err_mean_mm_s"],
                err_max=m["sync_err_max_mm_s"],
                cycle_mean_ms=m["cycle_mean_ms"],
                cycle_theory_ms=m["cycle_theory_ms"],
                tables_used=m["tables_used"],
            )
        )
        print(f"[飞剪扫描] 带速={v}mm/s 速度误差均值={m['sync_err_mean_mm_s']:.2f} "
              f"最大={m['sync_err_max_mm_s']:.2f}mm/s 节拍={m['cycle_mean_ms']:.1f}ms")
    return rows


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def _fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# 数据一致性自检：报告结论必须能被本次数据支撑（评审整改项）
# ---------------------------------------------------------------------------
def run_self_checks(
    cells: list[MatrixCell],
    improvements: dict[float, float],
    alarm_result: tuple[bool, bool],
    shear_rows: list[ShearRow],
) -> list[tuple[str, bool, str]]:
    """
    对实验结果做实时一致性校验，返回 (名称, 是否通过, 数据说明) 列表。
    任何一项不通过都意味着"报告结论可能与数据脱节"，主流程会以非零码退出。
    """
    checks: list[tuple[str, bool, str]] = []

    # ① 龙门：补偿对 P95 的压缩比应符合理论 (1 − 1/(1+2·Kcc))
    off = {c.mismatch: c.p95_um for c in cells if not c.comp}
    on = {c.mismatch: c.p95_um for c in cells if c.comp}
    if off and on:
        avg_off = float(np.mean(list(off.values())))
        avg_on = float(np.mean(list(on.values())))
        measured = (avg_off - avg_on) / avg_off * 100.0
        kcc = GantryParams().kcc
        theory = (1.0 - 1.0 / (1.0 + 2.0 * kcc)) * 100.0
        ok = abs(measured - theory) <= 5.0   # 容差：加减速段动态分量
        checks.append((
            "龙门补偿P95压缩比 vs 理论公式",
            ok,
            f"实测 {measured:.1f}% vs 理论 {theory:.1f}%（Kcc={kcc:g}），容差 ±5 个百分点",
        ))

    # ② 龙门：无补偿 P95 应随失配度单调递增（模型合理性）
    seq = [off[m] for m in sorted(off)]
    ok_mono = len(seq) >= 2 and all(b >= a for a, b in zip(seq, seq[1:]))
    checks.append((
        "无补偿P95随失配度单调递增",
        ok_mono,
        f"P95 序列 = {[round(v) for v in seq]} µm",
    ))

    # ③ 报警演示的方向性：无补偿应停机、有补偿应通过
    a_off, a_on = alarm_result
    checks.append((
        "报警联锁演示方向性",
        (a_off is True) and (a_on is False),
        f"无补偿触发={a_off}，有补偿触发={a_on}（失配30%/阈值2mm 工况）",
    ))

    # ④ 飞剪：稳态节拍与理论 L/v 的最大偏差应很小
    devs = [
        abs(r.cycle_mean_ms - r.cycle_theory_ms)
        for r in shear_rows if r.cuts >= 2
    ]
    dev_max = max(devs) if devs else float("nan")
    ok_dev = bool(devs) and dev_max < 5.0
    checks.append((
        "飞剪节拍偏差(实测-理论) < 5ms",
        ok_dev,
        f"各带速最大偏差 {dev_max:.2f} ms",
    ))

    return checks


def build_report_md(
    cells: list[MatrixCell],
    improvements: dict[float, float],
    alarm_result: tuple[bool, bool],
    shear_rows: list[ShearRow],
    checks: list[tuple[str, bool, str]] | None = None,
) -> str:
    """把全部实验结果拼装成 Markdown 测试报告文本"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- 龙门矩阵表 ----
    lines = []
    lines.append("# 测试报告：龙门双轴同步 与 电子凸轮·飞剪 虚拟轴仿真\n")
    lines.append(f"> 生成时间：{now}  ")
    lines.append("> **免责声明**：本报告全部指标均为 **仿真验证值**——由本项目 ")
    lines.append("> Python 虚拟伺服轴数值仿真得出，非真实驱动器调试结果。")
    lines.append("> 同步/凸轮的控制思想与真实系统一致，数值量级仅供算法对比参考。\n")

    lines.append("## 1 实验环境与方法\n")
    lines.append("| 项目 | 内容 |")
    lines.append("| --- | --- |")
    lines.append("| 平台 | Windows 10 + Python 3.12 |")
    lines.append("| 依赖 | numpy（仿真内核）、flask（看板） |")
    lines.append("| 控制周期 | 1 ms（与主流工业总线循环周期同构） |")
    lines.append("| 被控对象 | 虚拟伺服轴：位置环一阶闭环 + 梯形/S 曲线规划 + 脉冲当量量化 |")
    lines.append("| 龙门工况 | 定位行程 300mm，规划速度 300mm/s，基准增益 Kp=60(1/s) |")
    lines.append("| 失配建模 | Kp2 = Kp1 × (1 − 失配度)，两轴分别独立闭环 |")
    lines.append("")
    lines.append("统计口径：只取运动区段（含 0.3s 收敛尾），同步偏差 |Δ| = |act1 − act2|。\n")

    lines.append("## 2 龙门双轴同步矩阵实验（补偿 × 失配）\n")
    lines.append("| 失配度 | 补偿 | 同步偏差均值(µm) | P95(µm) | 最大(µm) | 触发报警 |")
    lines.append("| ---: | :---: | ---: | ---: | ---: | :---: |")
    for c in cells:
        lines.append(
            f"| {c.mismatch:.0%} | {'开' if c.comp else '关'} "
            f"| {_fmt(c.mean_um)} | {_fmt(c.p95_um)} | {_fmt(c.max_um)} "
            f"| {'是' if c.alarm_sync else '否'} |"
        )
    lines.append("")
    avg_p95_off = float(np.mean([c.p95_um for c in cells if not c.comp]))
    avg_p95_on = float(np.mean([c.p95_um for c in cells if c.comp]))
    avg_impr = (avg_p95_off - avg_p95_on) / avg_p95_off * 100.0
    lines.append("### 结论\n")
    lines.append(
        f"- 全部工况中，交叉耦合同步补偿平均使 P95 同步偏差下降 "
        f"**{_fmt(avg_impr)}%（仿真验证值）**；各失配度下的下降比例为："
        + "、".join(f"{k:.0%}→{v:.1f}%" for k, v in improvements.items())
        + "。\n"
    )
    kcc = GantryParams().kcc   # 矩阵实验使用默认参数，理论压缩比由此而来
    theory_drop = (1.0 - 1.0 / (1.0 + 2.0 * kcc)) * 100.0
    lines.append(f"- 机理解释：理论上补偿把稳态偏差压缩为 Δ*/(1+2·Kcc)。本实验 Kcc={kcc:g}，"
                 f"对应理论下降 {theory_drop:.1f}%；实测平均下降 {_fmt(avg_impr)}%，"
                 f"两者相差 {abs(avg_impr - theory_drop):.1f} 个百分点"
                 "（差异来自加减速段的动态分量）。\n")
    lines.append("- 失配越大，无补偿偏差近似线性增大（Δ ≈ v/Kp·失配度），补偿后基本被钳制在同一量级。\n")

    a_off, a_on = alarm_result
    lines.append("## 3 同步偏差报警联锁演示\n")
    lines.append("条件：失配 30%，同步偏差阈值收紧至 2mm。\n")
    lines.append("| 工况 | 结果 |")
    lines.append("| --- | --- |")
    lines.append(f"| 补偿关 | {'⚠ 触发同步报警并双轴停机' if a_off else '正常完成'} |")
    lines.append(f"| 补偿开 | {'⚠ 触发同步报警并双轴停机' if a_on else '✔ 正常完成'} |")
    lines.append("")
    lines.append(
        "> 说明：真实龙门横梁两侧严重失步会造成机械卡死/横梁扭曲，"
        "> 因此控制器必须提供“同步偏差超限即停”的安全联锁；补偿算法则从控制上避免进入该工况。\n"
    )

    lines.append("## 4 电子凸轮·飞剪带速扫描\n")
    lines.append("| 带速(mm/s) | 切割次数 | 误差均值(mm/s) | 占带速% | 误差最大(mm/s) | 节拍均值(ms) | 节拍理论(ms) | 节拍偏差(ms) | 所用凸轮表 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in shear_rows:
        ratio = r.err_mean / r.belt_speed * 100.0
        dev = abs(r.cycle_mean_ms - r.cycle_theory_ms)
        lines.append(
            f"| {r.belt_speed:.0f} | {r.cuts} | {_fmt(r.err_mean, 2)} | {_fmt(ratio, 2)} "
            f"| {_fmt(r.err_max, 2)} | {_fmt(r.cycle_mean_ms)} | {_fmt(r.cycle_theory_ms)} "
            f"| {_fmt(dev, 2)} | {r.tables_used} |"
        )
    lines.append("")
    # 结论全部由本次数据计算得出，不写死数值（评审整改项）
    worst = max(shear_rows, key=lambda r: r.err_mean / r.belt_speed)
    worst_ratio = worst.err_mean / worst.belt_speed * 100.0
    cycle_devs = [abs(r.cycle_mean_ms - r.cycle_theory_ms) for r in shear_rows]
    lines.append("### 结论\n")
    lines.append(
        f"- 同步段速度误差均值最大出现在带速 {worst.belt_speed:.0f}mm/s 工况："
        f"**{_fmt(worst.err_mean, 2)}mm/s，占该带速的 {_fmt(worst_ratio, 2)}%（仿真验证值）**；"
        f"对应瞬时最大 {_fmt(worst.err_max, 2)}mm/s。"
    )
    lines.append(
        f"- 循环节拍实测与理论值 L/v 的最大偏差为 {_fmt(max(cycle_devs), 2)}ms"
        "（逐带速实时计算），定长逻辑正确。"
    )
    lines.append("- 速度误差随带速升高而增大：同步窗在时域上变短，入口收敛段占比上升；"
                 "主要来源为剪切轴闭环滞后（e≈v/Kp），工程上可通过提高位置环增益或引入速度前馈进一步压缩。\n")

    # ---- 数据一致性自检（由本次实验数据实时校验，防止结论与数据脱节）----
    if checks:
        lines.append("## 5 数据一致性自检\n")
        lines.append("> 下表由脚本对本次实验数据实时计算生成；任何一项未通过，"
                     "即表示报告结论失去数据支撑。\n")
        lines.append("| 校验项 | 结果 | 说明 |")
        lines.append("| --- | :---: | --- |")
        for name, ok, detail in checks:
            lines.append(f"| {name} | {'通过' if ok else '未通过'} | {detail} |")
        lines.append("")

    lines.append("## 6 复现方式\n")
    lines.append("```bash")
    lines.append("pip install -r requirements.txt")
    lines.append("python -m testbench.batch_test     # 重新生成本报告与 CSV 数据")
    lines.append("python axis/virtual_axis.py        # 单模块自测试")
    lines.append("python gantry/gantry.py            # 龙门同步自测试")
    lines.append("python cam/electronic_cam.py       # 电子凸轮·飞剪自测试")
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    """运行全部实验并输出报告与原始数据"""
    print("=" * 72)
    print("开始批量测试：龙门矩阵 + 报警演示 + 飞剪带速扫描")
    print("=" * 72)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cells = run_gantry_matrix()
    improvements = _p95_improvements(cells)
    alarm_result = run_alarm_demo()
    shear_rows = run_shear_sweep()

    # ---- 数据一致性自检（评审整改项）：结论必须能被数据支撑 ----
    checks = run_self_checks(cells, improvements, alarm_result, shear_rows)

    # ---- 写 CSV 原始数据 ----
    matrix_csv = DATA_DIR / "gantry_matrix.csv"
    with open(matrix_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["失配度", "补偿", "均值_um", "P95_um", "最大_um", "样本数", "触发报警"])
        for c in cells:
            w.writerow([f"{c.mismatch:.2f}", "on" if c.comp else "off",
                        f"{c.mean_um:.2f}", f"{c.p95_um:.2f}", f"{c.max_um:.2f}",
                        c.samples, "Y" if c.alarm_sync else "N"])

    shear_csv = DATA_DIR / "flying_shear_sweep.csv"
    with open(shear_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["带速_mm_s", "切割次数", "误差均值_mm_s", "误差均值占带速_%",
                    "误差最大_mm_s", "节拍均值_ms", "节拍理论_ms", "节拍偏差_ms"])
        for r in shear_rows:
            w.writerow([f"{r.belt_speed:.0f}", r.cuts, f"{r.err_mean:.3f}",
                        f"{r.err_mean / r.belt_speed * 100.0:.3f}",
                        f"{r.err_max:.3f}", f"{r.cycle_mean_ms:.2f}",
                        f"{r.cycle_theory_ms:.2f}",
                        f"{abs(r.cycle_mean_ms - r.cycle_theory_ms):.2f}"])

    # ---- 写 Markdown 报告 ----
    report = build_report_md(cells, improvements, alarm_result, shear_rows, checks)
    report_path = DOCS_DIR / "测试报告.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n[数据一致性自检]")
    failed = 0
    for name, ok, detail in checks:
        mark = "[通过]" if ok else "[未通过]"
        print(f"  {mark} {name}：{detail}")
        failed += 0 if ok else 1

    print("\n" + "=" * 72)
    print(f"完成！报告：{report_path}")
    print(f"数据：{matrix_csv}")
    print(f"数据：{shear_csv}")
    if failed:
        print(f"警告：{failed} 项自检未通过，报告已如实标注；"
              "结论暂不可信，请检查参数或实现后重跑。")
        sys.exit(2)


if __name__ == "__main__":
    # Windows 控制台 GBK 兜底
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
