# -*- coding: utf-8 -*-
"""
龙门双轴同步控制（Gantry Dual-Axis Synchronization）
====================================================

物理背景
--------
龙门机的 X 轴由两台伺服电机分别驱动横梁两侧（X1/X2）。两电机的
位置环增益 Kp、负载特性不可能完全一致（编码器/丝杠/摩擦差异），
恒速运动时各自的跟随误差 e ≈ v/Kp 不同，于是产生"同步偏差"：

        Δ = p_act1 - p_act2 = v/Kp1 - v/Kp2 ≠ 0   （增益失配时）

控制结构
--------
    MASTER(虚拟主轴,轨迹发生器) ──目标流──►  X1 从轴闭环 ──► act1
                        │                     ▲
                        └──目标流──►  X2 从轴闭环 ──► act2
                                      ▲         │
                    交叉耦合补偿：Δ 反馈给落后轴微调

交叉耦合同步补偿（Cross-Coupling Control）
------------------------------------------
不做各自"回到主轴"的强纠偏，而是把两轴之间的相对偏差 Δ 作为额外
反馈量，对称地修正两轴的目标：
        tgt1 = m - Kcc·Δ      tgt2 = m + Kcc·Δ
可以推导（见 docs/系统设计说明书.md）：稳态同步偏差被压缩为
        Δ* = v·(1/Kp2 - 1/Kp1) / (1 + 2·Kcc)
即补偿使偏差下降 (1 + 2·Kcc) 倍 —— 这是龙门同步的核心思想，
与真实系统中"双驱龙门 + 同步误差反馈"完全同构。

本文件可直接运行自测试：python gantry/gantry.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 允许作为包或直接脚本两种方式运行
try:
    from axis.virtual_axis import VirtualAxis, AxisConfig, AlarmCode
except ImportError:  # pragma: no cover - 直接脚本运行时的路径兜底
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from axis.virtual_axis import VirtualAxis, AxisConfig, AlarmCode


@dataclass
class GantryParams:
    """龙门同步实验的一组参数（矩阵扫描就是对不同参数组合各跑一次）"""

    kp_base: float = 60.0        # X1 轴位置环增益 (1/s) —— 基准增益
    mismatch: float = 0.20       # 两轴增益失配度：Kp2 = Kp1 * (1 - mismatch)
    load2_extra: float = 0.0     # X2 轴附加等效负载扰动 (mm/s)，模拟两侧摩擦不一致
    comp_enabled: bool = True    # 交叉耦合补偿开关
    kcc: float = 2.0             # 交叉耦合增益（无量纲），理论压偏倍数 = 1+2*Kcc
    sync_threshold_mm: float = 10.0  # 同步偏差报警阈值 (mm)，超过则双轴停机
    target_mm: float = 300.0     # 定位目标 (mm)
    vel_mm_s: float = 300.0      # 规划速度 (mm/s)
    acc_mm_s2: float = 3000.0    # 规划加速度 (mm/s^2)
    jerk_mm_s3: float = 60000.0  # S 曲线加加速度 (mm/s^3)
    use_s_curve: bool = True     # 梯形(False)/S 曲线(True)
    vel_ff: float = 0.0          # 速度前馈系数（0=纯比例位置环）

    def describe(self) -> str:
        """一行文字描述参数组合（用于报告表格行头）"""
        comp = "开" if self.comp_enabled else "关"
        return f"失配{self.mismatch:.0%}_补偿{comp}"


@dataclass
class GantryResult:
    """一次龙门定位过程的仿真结果"""

    params: GantryParams
    t: np.ndarray                 # 时间序列 (s)
    master_cmd: np.ndarray        # 主轴指令位置 (mm)
    act1: np.ndarray              # X1 实际位置 (mm)
    act2: np.ndarray              # X2 实际位置 (mm)
    sync_err: np.ndarray          # 同步偏差 act1-act2 (mm)
    alarm_sync: bool              # 是否触发同步偏差报警
    run_seconds: float            # 仿真总时长 (s)

    # ------------------------------------------------------------------
    def stats(self, tail_trim_s: float = 0.0) -> dict:
        """
        统计同步偏差 |Δ| 的 均值 / P95 / 最大值 (µm)。
        默认统计全过程；tail_trim_s 可裁掉尾部静止段。
        """
        mask = self.t <= (self.run_seconds - tail_trim_s + 1e-9)
        data = np.abs(self.sync_err[mask]) * 1000.0  # mm → µm
        return {
            "mean_um": float(np.mean(data)),
            "p95_um": float(np.percentile(data, 95)),
            "max_um": float(np.max(data)),
        }

    def to_plot_series(self, max_points: int = 600) -> dict:
        """降采样为适合 Web 图表绘制的数据系列"""
        n = len(self.t)
        step = max(1, n // max_points)
        idx = slice(0, n, step)
        return {
            "t": [round(v, 4) for v in self.t[idx]],
            "master_cmd": [round(v, 4) for v in self.master_cmd[idx]],
            "act1": [round(v, 4) for v in self.act1[idx]],
            "act2": [round(v, 4) for v in self.act2[idx]],
            "sync_err_mm": [round(v, 5) for v in self.sync_err[idx]],
            "sync_threshold_mm": self.params.sync_threshold_mm,
            "stats": {k: round(v, 2) for k, v in self.stats().items()},
            "alarm_sync": self.alarm_sync,
        }


class GantryController:
    """
    龙门双轴同步控制器。

    组成：
      - MASTER：虚拟主轴（只做轨迹发生器，其指令位置作为两从轴的共同目标）
      - X1/X2：两台虚拟伺服从轴，增益/负载按参数配置（天然产生失配）
      - 交叉耦合补偿器：把同步偏差对称反馈到两轴目标上（可开关）
      - 同步监测器：|Δ| 超阈值 → 双轴报警停机（模拟真实龙门的安全逻辑）
    """

    def __init__(self, params: GantryParams | None = None, dt: float = 0.001):
        self.p = params if params is not None else GantryParams()
        self.dt = dt
        self._build_axes()

    # ------------------------------------------------------------------
    def _build_axes(self) -> None:
        """按当前参数创建主轴与两个从轴"""
        p = self.p

        # 主轴：仅作轨迹发生器，规划参数与从轴一致，便于对齐速度曲线
        self.master = VirtualAxis(
            AxisConfig(
                name="MASTER",
                max_vel=p.vel_mm_s,
                max_acc=p.acc_mm_s2,
                max_jerk=p.jerk_mm_s3,
                use_s_curve=p.use_s_curve,
                following_error_alarm=0,  # 主轴是理想轨迹源，不参与超差保护
            ),
            dt=self.dt,
        )
        self.master.enable()

        # X1 从轴：基准增益
        self.axis1 = VirtualAxis(
            AxisConfig(
                name="X1",
                kp_pos=p.kp_base,
                vel_feedforward=p.vel_ff,
                max_vel=p.vel_mm_s * 1.5,
                max_acc=p.acc_mm_s2 * 2,
                max_jerk=p.jerk_mm_s3 * 2,
                use_s_curve=p.use_s_curve,
                following_error_alarm=50.0,  # 单轴超差保护放宽，重点看同步偏差
            ),
            dt=self.dt,
        )
        # X2 从轴：增益按失配度缩小 + 可配附加负载 → 同步偏差的根源
        self.axis2 = VirtualAxis(
            AxisConfig(
                name="X2",
                kp_pos=p.kp_base * (1.0 - p.mismatch),
                load_disturbance=p.load2_extra,
                vel_feedforward=p.vel_ff,
                max_vel=p.vel_mm_s * 1.5,
                max_acc=p.acc_mm_s2 * 2,
                max_jerk=p.jerk_mm_s3 * 2,
                use_s_curve=p.use_s_curve,
                following_error_alarm=50.0,
            ),
            dt=self.dt,
        )
        self.axis1.enable()
        self.axis2.enable()
        self.axis1.set_following_stream(True)
        self.axis2.set_following_stream(True)

        self.alarm_sync = False  # 同步报警锁存标志

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """全部复位后按当前参数重建（保证每次实验相互独立）"""
        self._build_axes()

    # ------------------------------------------------------------------
    def _step_once(self) -> None:
        """推进一个控制周期：主轴 → 补偿分配 → 从轴闭环 → 同步监测"""
        p = self.p

        # 1) 主轴轨迹推进
        self.master.step()

        # 2) 计算本周期两从轴的目标（是否叠加交叉耦合修正）
        m_cmd = self.master.cmd_pos
        delta = self.axis1.act_pos - self.axis2.act_pos  # 当前同步偏差
        if p.comp_enabled and not self.alarm_sync:
            tgt1 = m_cmd - p.kcc * delta   # 领先轴目标回拉
            tgt2 = m_cmd + p.kcc * delta   # 落后轴目标前推
        else:
            tgt1 = m_cmd
            tgt2 = m_cmd

        # 3) 从轴各自闭环跟随目标流
        self.axis1.set_stream_target(tgt1, vel=p.vel_mm_s)
        self.axis2.set_stream_target(tgt2, vel=p.vel_mm_s)
        self.axis1.step()
        self.axis2.step()

        # 4) 同步偏差监测（真实龙门的安全联锁逻辑）
        if not self.alarm_sync and abs(delta) > p.sync_threshold_mm:
            self.alarm_sync = True
            print(
                f"[GANTRY] 同步偏差 {delta:.3f}mm 超过阈值 "
                f"{p.sync_threshold_mm}mm → 双轴报警停机！"
            )

    # ------------------------------------------------------------------
    def run_positioning(
        self,
        target_mm: float | None = None,
        record_every: int = 5,
        settle_extra_s: float = 0.8,
    ) -> GantryResult:
        """
        执行一次完整的双轴定位过程并记录遥测曲线。

        参数
        ----
        target_mm     : 定位目标，None 则用参数默认值
        record_every  : 每 N 个控制周期记录一个点（降低数据量）
        settle_extra_s: 到位后再多仿真的时间（观察误差收敛过程）
        """
        if target_mm is not None and target_mm != self.p.target_mm:
            self.p.target_mm = target_mm
        self.reset()  # 独立复跑

        p = self.p
        self.master.move_abs(p.target_mm, vel=p.vel_mm_s)

        t_list, m_list, a1_list, a2_list, d_list = [], [], [], [], []
        done_t: float | None = None
        max_steps = int(120 / self.dt)  # 120s 硬超时保护
        k = 0

        while k < max_steps:
            self._step_once()
            k += 1

            if k % record_every == 0:
                t_list.append(k * self.dt)
                m_list.append(self.master.cmd_pos)
                a1_list.append(self.axis1.act_pos)
                a2_list.append(self.axis2.act_pos)
                d_list.append(self.axis1.act_pos - self.axis2.act_pos)

            # 到位判断：主轴完成 且 两从轴进入稳态（指令速度为 0）
            moving = (
                abs(self.master.cmd_vel) > 1e-6
                or abs(self.axis1.cmd_vel) > 1e-6
                or self.axis1.is_moving
                or self.axis2.is_moving
            )
            if not moving and not self.alarm_sync:
                if done_t is None:
                    done_t = k * self.dt
                elif k * self.dt - done_t >= settle_extra_s:
                    break
            else:
                done_t = None
            if self.alarm_sync:
                # 报警后再记录 200ms 收尾数据，便于图表展示停机时刻
                if done_t is None:
                    done_t = k * self.dt
                elif k * self.dt - done_t >= 0.2:
                    break

        result = GantryResult(
            params=self.p,
            t=np.array(t_list),
            master_cmd=np.array(m_list),
            act1=np.array(a1_list),
            act2=np.array(a2_list),
            sync_err=np.array(d_list),
            alarm_sync=self.alarm_sync,
            run_seconds=k * self.dt,
        )
        return result


# ---------------------------------------------------------------------------
# 自测试：直接运行本文件，对比"补偿关 vs 开"在 30% 增益失配下的表现
# python gantry/gantry.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("龙门双轴同步自测试：失配 30%，对比交叉耦合补偿 关/开")
    print("=" * 72)

    for comp in (False, True):
        prm = GantryParams(mismatch=0.30, comp_enabled=comp)
        ctl = GantryController(prm)
        res = ctl.run_positioning()
        st = res.stats()
        theory_no = (
            prm.vel_mm_s
            * (1.0 / (prm.kp_base * 0.7) - 1.0 / prm.kp_base)
            * 1000.0
        )  # µm
        theory = theory_no / (1 + 2 * prm.kcc) if comp else theory_no
        print(
            f"补偿={'开' if comp else '关'} | 同步偏差均值={st['mean_um']:8.1f}µm "
            f"P95={st['p95_um']:8.1f}µm 最大={st['max_um']:8.1f}µm "
            f"| 理论稳态≈{theory:7.1f}µm | 报警={res.alarm_sync}"
        )

    print("\n说明：理论稳态 Δ*=v(1/Kp2-1/Kp1)/(1+2Kcc)；补偿开启时应显著下降。")
