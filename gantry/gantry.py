# -*- coding: utf-8 -*-
"""
龙门双轴同步控制（Gantry Dual-Axis Synchronization）
====================================================

物理背景
--------
龙门机的 X 轴由两台伺服电机分别驱动横梁两侧（X1/X2）。两电机的
位置环增益 Kp、负载特性不可能完全一致（编码器/丝杠/摩擦差异），
恒速运动时各自的跟随误差 e ≈ v/Kp 不同，于是产生"同步偏差"：

        Δ = p_act1 - p_act2 = v/Kp2 - v/Kp1 ≠ 0   （增益失配时）

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

from dataclasses import dataclass

import numpy as np

# 允许作为包或直接脚本两种方式运行
try:
    from axis.virtual_axis import VirtualAxis, AxisConfig, AlarmCode, downsample_slice
except ImportError:  # pragma: no cover - 直接脚本运行时的路径兜底
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from axis.virtual_axis import VirtualAxis, AxisConfig, AlarmCode, downsample_slice


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
    alarm_sync: bool              # 是否触发同步偏差报警（联锁停机）
    run_seconds: float            # 仿真总时长 (s)
    vel1: np.ndarray | None = None    # X1 实际速度 (mm/s)，与 t 同长
    vel2: np.ndarray | None = None    # X2 实际速度 (mm/s)，与 t 同长
    alarm_time_s: float | None = None  # 报警联锁触发时刻 (s)；未报警为 None

    # ------------------------------------------------------------------
    def _active_end_index(self, settle_tail_s: float = 0.3) -> int:
        """
        计算"有效统计窗口"的结束索引：
        从开始到 主轴最后一次移动的时刻 再加 settle_tail_s 收敛尾段。
        静止等待段不参与统计，避免把零偏差样本混入稀释 均值/P95。
        """
        n = len(self.t)
        if n < 2:
            return n
        dm = np.abs(np.diff(self.master_cmd, append=self.master_cmd[-1]))
        moving_idx = np.nonzero(dm > 1e-7)[0]
        if len(moving_idx) == 0:
            return n
        # 用时间序列估计采样周期，换算尾段长度。
        # round 抑制浮点噪声：k*dt 的差值中位数会带最后一位浮点误差，
        # 直接 int() 截断会让尾段长度在 59/60 之间漂移，统计窗口
        # （从而 CSV 均值）随记录长度不可复现。
        dt_est = float(np.median(np.diff(self.t))) if n > 1 else 0.001
        tail = int(round(settle_tail_s / dt_est)) if dt_est > 0 else 0
        return min(n, int(moving_idx[-1]) + 1 + tail)

    def stats(self, use_active_window: bool = True) -> dict:
        """
        统计同步偏差 |Δ| 的 均值 / P95 / 最大值 (µm)。

        use_active_window=True 时只统计运动区段（含 0.3s 收敛尾），
        这是评价同步性能的合理口径；False 则统计全部记录样本。
        """
        end = self._active_end_index() if use_active_window else len(self.t)
        data = np.abs(self.sync_err[:end]) * 1000.0  # mm → µm
        if len(data) == 0:
            data = np.array([0.0])
        return {
            "mean_um": float(np.mean(data)),
            "p95_um": float(np.percentile(data, 95)),
            "max_um": float(np.max(data)),
            "samples": int(len(data)),
        }

    def stop_metrics(self) -> dict | None:
        """
        同步报警联锁停机过程的可验证数据（复审报告 05 N-P3-2：此前
        alarm_time/vel 序列仅服务 testbench 内部，看板与外部拿不到联锁证据）。

        报警触发时返回：
          alarm_time_s        触发时刻 (s)
          stop_ms             触发 → 双轴速度归零的耗时（与 testbench
                              _verify_interlock 同口径）
          stop_within_resolution  True=首个报警后记录样本速度已为 0
                              （停机耗时小于记录分辨率）
          dt_record_ms        记录分辨率 (ms)
          peak_sync_err_mm    停机前 |Δ| 峰值（锁存冻结值）
          act1/act2_stop_mm   双轴停机位置（未走完全程）
        未报警返回 None。记录分辨率为 record_every×dt（默认 5ms）。
        """
        if not (self.alarm_sync and self.alarm_time_s is not None) or len(self.t) < 2:
            return None
        idx = np.nonzero(self.t >= self.alarm_time_s)[0]
        if len(idx) == 0:
            return None
        dt_rec = float(np.median(np.diff(self.t)))
        stop_ms = 0.0
        if self.vel1 is not None and self.vel2 is not None and len(idx) >= 2:
            v_peak = np.maximum(np.abs(self.vel1[idx]), np.abs(self.vel2[idx]))
            moving = v_peak > 1e-6
            if moving.any():
                stop_ms = float((self.t[idx[moving][-1]] - self.alarm_time_s) * 1000.0)
        return {
            "alarm_time_s": round(self.alarm_time_s, 4),
            "stop_ms": round(stop_ms, 1),
            "stop_within_resolution": stop_ms == 0.0,
            "dt_record_ms": round(dt_rec * 1000.0, 1),
            "peak_sync_err_mm": round(float(np.max(np.abs(self.sync_err))), 3),
            "act1_stop_mm": round(float(self.act1[idx[0]]), 3),
            "act2_stop_mm": round(float(self.act2[idx[0]]), 3),
        }

    def to_plot_series(self, max_points: int = 600) -> dict:
        """降采样为适合 Web 图表绘制的数据系列（含联锁停机证据字段）"""
        idx = downsample_slice(len(self.t), max_points)
        return {
            "t": [round(v, 4) for v in self.t[idx]],
            "master_cmd": [round(v, 4) for v in self.master_cmd[idx]],
            "act1": [round(v, 4) for v in self.act1[idx]],
            "act2": [round(v, 4) for v in self.act2[idx]],
            "sync_err_mm": [round(v, 5) for v in self.sync_err[idx]],
            # 停机过程速度序列（报警后归零并冻结，供看板/外部验证联锁）
            "vel1": [round(v, 2) for v in self.vel1[idx]] if self.vel1 is not None else [],
            "vel2": [round(v, 2) for v in self.vel2[idx]] if self.vel2 is not None else [],
            "sync_threshold_mm": self.params.sync_threshold_mm,
            "stats": {k: round(v, 2) for k, v in self.stats().items()},
            "alarm_sync": self.alarm_sync,
            "alarm_time_s": (
                round(self.alarm_time_s, 4) if self.alarm_time_s is not None else None
            ),
            "stop_metrics": self.stop_metrics(),
        }


class GantryController:
    """
    龙门双轴同步控制器。

    组成：
      - MASTER：虚拟主轴（只做轨迹发生器，其指令位置作为两从轴的共同目标）
      - X1/X2：两台虚拟伺服从轴，增益/负载按参数配置（天然产生失配）
      - 交叉耦合补偿器：把同步偏差对称反馈到两轴目标上（可开关）
      - 同步监测器：|Δ| 超阈值 → 联锁停机（模拟真实龙门的安全逻辑）

    同步报警联锁（超差即停，真实执行的动作链）：
      ① 主轴停令：MASTER 按加速度限幅斜坡停车，目标流归零；
      ② 双从轴封锁输出：X1/X2 脱离目标流、去使能并锁存 SYNC_EXCEED
         报警码（实际位置/速度冻结，报警后 20ms 内双轴速度归零）；
      ③ 控制器进入 FAULTED（见 state 属性）：补偿停用、不再下发目标，
         须 reset() 复位重建后才能重新定位。
    """

    ALARM_TAIL_S = 0.2  # 报警后再记录的收尾数据时长 (s)，便于图表展示停机时刻

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

        self.alarm_sync = False              # 同步报警锁存标志（控制器级）
        self.alarm_code = AlarmCode.NONE     # 控制器级报警码（SYNC_EXCEED）
        self._alarm_time_s: float | None = None  # 报警触发时刻 (s)
        self._step_count = 0                 # _step_once 累计步数（推算报警时刻用）

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        """控制器状态：RUNNING（正常运行）/ FAULTED（同步报警联锁触发后）"""
        return "FAULTED" if self.alarm_sync else "RUNNING"

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """全部复位后按当前参数重建（保证每次实验相互独立）"""
        self._build_axes()

    # ------------------------------------------------------------------
    def _step_once(self) -> None:
        """推进一个控制周期：主轴 → 补偿分配 → 从轴闭环 → 同步监测"""
        p = self.p
        self._step_count += 1

        # 1) 主轴轨迹推进
        self.master.step()

        if self.alarm_sync:
            # FAULTED：联锁已触发，不再产生任何新目标 —— 主轴指令斜坡收尾，
            # 两从轴输出封锁（实际位置/速度冻结），仅推进模型保持冻结状态
            self.axis1.step()
            self.axis2.step()
            return

        # 2) 计算本周期两从轴的目标（是否叠加交叉耦合修正）
        m_cmd = self.master.cmd_pos
        delta = self.axis1.act_pos - self.axis2.act_pos  # 当前同步偏差
        if p.comp_enabled:
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
        if abs(delta) > p.sync_threshold_mm:
            self._trigger_sync_interlock(delta)

    # ------------------------------------------------------------------
    def _trigger_sync_interlock(self, delta: float) -> None:
        """
        同步偏差超限 → 执行真实的联锁停机动作链（FAULTED）：
          ① 主轴停令：MASTER 脱离定位任务，指令按加速度限幅斜坡归零；
          ② 双从轴脱离目标流 + 封锁输出（去使能）+ 锁存 SYNC_EXCEED 报警码；
          ③ 控制器置 alarm_sync / alarm_code，记录触发时刻；
             之后 _step_once 不再下发任何目标，须 reset() 复位才能重走。
        """
        self.alarm_sync = True
        self.alarm_code = AlarmCode.SYNC_EXCEED
        self._alarm_time_s = self._step_count * self.dt
        print(
            f"[GANTRY] 同步偏差 {delta:.3f}mm 超过阈值 "
            f"{self.p.sync_threshold_mm}mm → 触发联锁停机！"
        )
        self.master.stop()
        for ax in (self.axis1, self.axis2):
            ax.set_following_stream(False)
            ax.latch_alarm(
                AlarmCode.SYNC_EXCEED,
                f"龙门同步偏差 {delta:.3f}mm 超阈值({self.p.sync_threshold_mm}mm)",
            )
            ax.disable()  # 封锁输出：实际位置/速度立即冻结（见 VirtualAxis.step）

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
        v1_list, v2_list = [], []
        done_t: float | None = None
        max_steps = int(120 / self.dt)  # 120s 硬超时兜底（正常定位 ~2s 即收敛退出）
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
                v1_list.append(self.axis1.act_vel)
                v2_list.append(self.axis2.act_vel)

            now = k * self.dt
            if self.alarm_sync:
                # 报警联锁已触发：双轴已封锁停机，再记录 ALARM_TAIL_S 收尾
                # 数据用于图表展示停机时刻，随后退出
                if now - self._alarm_time_s >= self.ALARM_TAIL_S:
                    break
                continue

            # 到位判断：主轴与两从轴的指令速度均归零。
            # 注意不能用 axis.is_moving()：目标流跟随态从轴（set_stream_target）
            # 的 _mode 恒为 JOG，is_moving 恒为 True，用它判稳会导致循环
            # 永远跑满 120s 超时（死代码缺陷，详见审查报告 P1-2）。
            moving = (
                abs(self.master.cmd_vel) > 1e-6
                or abs(self.axis1.cmd_vel) > 1e-6
                or abs(self.axis2.cmd_vel) > 1e-6
            )
            if not moving:
                if done_t is None:
                    done_t = now
                elif now - done_t >= settle_extra_s:
                    break
            else:
                done_t = None

        result = GantryResult(
            params=self.p,
            t=np.array(t_list),
            master_cmd=np.array(m_list),
            act1=np.array(a1_list),
            act2=np.array(a2_list),
            sync_err=np.array(d_list),
            alarm_sync=self.alarm_sync,
            run_seconds=k * self.dt,
            vel1=np.array(v1_list),
            vel2=np.array(v2_list),
            alarm_time_s=self._alarm_time_s,
        )
        return result


# ---------------------------------------------------------------------------
# 自测试：直接运行本文件，对比"补偿关 vs 开"在 30% 增益失配下的表现，
# 并回归验证 到位判定收敛退出 与 同步报警联锁停机 两项关键行为
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
            f"| 理论稳态≈{theory:7.1f}µm | 报警={res.alarm_sync} "
            f"| 定位收敛耗时={res.run_seconds:.2f}s"
        )
        # 回归断言（审查报告 P1-2）：到位判定必须真实生效，
        # 修复前的死代码缺陷会让每次定位固定空跑 120s
        assert res.run_seconds < 10.0, (
            f"定位未及时收敛退出：run_seconds={res.run_seconds:.1f}s（疑似 120s 空跑回归）"
        )
        assert not res.alarm_sync, "默认阈值 10mm 下 30% 失配不应触发同步报警"

    # ---- 联锁回归（审查报告 P1-1）：阈值 2mm + 无补偿 → 双轴真实停机 ----
    print("\n[联锁回归] 失配 30%、阈值 2mm、无补偿 → 应触发联锁并双轴停机：")
    prm = GantryParams(mismatch=0.30, comp_enabled=False, sync_threshold_mm=2.0)
    ctl = GantryController(prm)
    res = ctl.run_positioning()
    assert res.alarm_sync and res.alarm_time_s is not None, "应触发同步报警联锁"
    assert ctl.state == "FAULTED", "报警后控制器应进入 FAULTED 状态"
    assert ctl.alarm_code == AlarmCode.SYNC_EXCEED, "控制器应锁存 SYNC_EXCEED 报警码"
    assert ctl.axis1.alarm == AlarmCode.SYNC_EXCEED and not ctl.axis1.is_enabled, (
        "X1 应锁存 SYNC_EXCEED 并封锁输出"
    )
    assert ctl.axis2.alarm == AlarmCode.SYNC_EXCEED and not ctl.axis2.is_enabled, (
        "X2 应锁存 SYNC_EXCEED 并封锁输出"
    )
    idx = np.nonzero(res.t >= res.alarm_time_s + 0.02)[0]  # 报警 20ms 后
    assert len(idx) > 1, "报警后应仍有收尾记录样本"
    assert np.max(np.abs(res.vel1[idx])) == 0.0 and np.max(np.abs(res.vel2[idx])) == 0.0, (
        "报警 20ms 后双轴速度必须归零并保持（真实停机，而非走完全程）"
    )
    drift = np.max(np.abs(np.diff(res.act1[idx]))) + np.max(np.abs(np.diff(res.act2[idx])))
    assert drift == 0.0, f"报警后双轴位置必须冻结（实测漂移 {drift:.6f}mm）"
    assert res.run_seconds < 1.0, (
        f"报警工况应在报警后 0.2s 收尾退出，实测 run_seconds={res.run_seconds:.2f}s"
    )
    print(
        f"  触发时刻 t={res.alarm_time_s:.3f}s，双轴停机于 "
        f"act1={res.act1[idx[0]]:.1f}mm / act2={res.act2[idx[0]]:.1f}mm（未走完全程 300mm），"
        f"20ms 后速度=0、位置零漂移，SYNC_EXCEED 已锁存 [OK]"
    )

    print("\n说明：理论稳态 Δ*=v(1/Kp2-1/Kp1)/(1+2Kcc)；补偿开启时应显著下降。")
    print("说明：同步报警联锁 = 主轴斜坡停 + 双从轴封锁输出并锁存 SYNC_EXCEED，")
    print("      控制器进入 FAULTED，须 reset() 复位后才能重走（自测试已断言）。")
