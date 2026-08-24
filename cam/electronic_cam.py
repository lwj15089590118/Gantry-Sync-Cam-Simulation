# -*- coding: utf-8 -*-
"""
电子凸轮（Electronic Cam）与飞剪（Flying Shear）应用仿真
==========================================================

电子凸轮原理
------------
电子凸轮用一张"位移表"代替机械凸轮：主轴位置 m（连续增长）按周期 L 取模
得到相位，查表插值得到从轴（剪切轴）目标位移：

        y_ref(m) = Table( mod(m, L) )      （表内线性插值）

只要改变表的数据，就能改变从轴的运动规律 —— 这就是"软件凸轮"。
本模块的飞剪表由程序按飞剪工艺自动生成：

        待机区 ──► 加速追赶 ──► 与带速同步(切刀) ──► 快速返回

关键推导：设输送带速度为 v_belt，主轴每周期前进定长 L（=切料长度），
相位 θ = mod(m,L)/L。刀的水平速度：
        v_knife = (dy/dθ)·(dθ/dt) = (dy/dθ)·(v_belt/L)
同步条件 v_knife = v_belt  ⟺  dy/dθ = L
即：**表中同步段的斜率必须等于定长 L** —— 这是飞剪凸轮表的核心设计规则。

指标口径
--------
同步段内速度误差 = | v_knife_actual(t) − v_belt |，取同步窗口中段统计
均值/最大值；循环节拍 = 相邻两次剪切的时间间隔（理论值 = L / v_belt）。

免责说明：全部为 Python 虚拟轴数值仿真，非真实驱动器调试；
但凸轮表构造、插值跟随、速度误差指标的计算思想与真实系统一致。

本文件可直接运行自测试：python cam/electronic_cam.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from axis.virtual_axis import VirtualAxis, AxisConfig, downsample_slice
except ImportError:  # pragma: no cover - 直接脚本运行时的路径兜底
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from axis.virtual_axis import VirtualAxis, AxisConfig, downsample_slice


# ---------------------------------------------------------------------------
# 工具函数：平滑 S 形过渡（五次多项式，端点斜率为零）
# ---------------------------------------------------------------------------
def smootherstep(x: float | np.ndarray) -> float | np.ndarray:
    """五次平滑阶跃 s(x)=6x^5-15x^4+10x^3，s(0)=0, s(1)=1, 两端导数为 0"""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


# ---------------------------------------------------------------------------
# 电子凸轮表
# ---------------------------------------------------------------------------
class CamTable:
    """
    凸轮表：n 点 主轴位移 → 从轴位移 映射，周期性循环使用。

    表头存储 master 点列（含首尾两点，xs[0]=0, xs[-1]=L），运行时对
    mod(主轴位置, L) 做线性插值。256 点是工程上常见的表长规格。
    """

    def __init__(self, name: str, master_points: np.ndarray, slave_points: np.ndarray):
        if len(master_points) != len(slave_points):
            raise ValueError("凸轮表主/从轴点数不一致")
        if len(master_points) < 8:
            raise ValueError("凸轮表点数过少（至少 8 点）")
        self.name = name
        self.xs = np.asarray(master_points, dtype=float)
        self.ys = np.asarray(slave_points, dtype=float)
        self.period = float(self.xs[-1] - self.xs[0])

    # ------------------------------------------------------------------
    def evaluate(self, master_pos: float) -> float:
        """查询：主轴位置 → 从轴目标位移（周期回绕 + 线性插值）"""
        phase = math.fmod(master_pos, self.period)
        if phase < 0:
            phase += self.period
        return float(np.interp(phase, self.xs, self.ys))

    # ------------------------------------------------------------------
    def scaled(self, ratio: float, name: str | None = None) -> "CamTable":
        """生成按比例缩放的新表（类似电子齿轮比叠加在凸轮上）"""
        return CamTable(
            name or f"{self.name}x{ratio:g}",
            self.xs.copy(),
            self.ys * ratio,
        )

    def resample(self, n_points: int) -> "CamTable":
        """重采样为指定点数的等距新表"""
        new_xs = np.linspace(self.xs[0], self.xs[-1], n_points)
        new_ys = np.interp(new_xs, self.xs, self.ys)
        return CamTable(f"{self.name}@{n_points}", new_xs, new_ys)


# ---------------------------------------------------------------------------
# 飞剪凸轮表生成器
# ---------------------------------------------------------------------------
def build_flying_shear_table(
    product_length_mm: float = 600.0,
    sync_window_mm: float = 120.0,
    standby_frac: float = 0.08,
    accel_frac: float = 0.12,
    decel_frac: float = 0.10,
    n_points: int = 256,
    name: str = "FLY_SHEAR",
) -> tuple[CamTable, dict]:
    """
    按飞剪工艺生成凸轮表。

    相位分段（θ = 输送带走过的定长比例）::

        [0, a1)                 待机区   刀具停在最前端等待位（w=dy/dθ=0）
        [a1, a2)                加速追赶 平滑加速到与带速对应斜率 L
        [a2, a3)                同步区   斜率恒为 L（刀与带同速，中点落刀）
        [a3, a4)                减速脱离 平滑减速到 0 并转入返回
        [a4, 1)                 快速返回 负速度梯形波，把刀送回起点

    其中 w = dy/dθ；同步段斜率必须等于定长 L（见文件头推导）。
    返回 (凸轮表, 分段信息 dict)，分段信息供图表标注与单元测试使用。
    """
    if not (0 < sync_window_mm < product_length_mm):
        raise ValueError("要求 0 < 同步窗长度 < 定长")
    L = float(product_length_mm)
    win_frac = sync_window_mm / L

    a1 = standby_frac
    a2 = a1 + accel_frac
    a3 = a2 + win_frac
    a4 = a3 + decel_frac
    if a4 >= 0.98:
        raise ValueError("分段总比例超界，请减小各段占比")

    # ---- 细网格上构造速度形状函数 w(θ)，再积分得到位移 ----
    n_fine = 8192
    theta = np.linspace(0.0, 1.0, n_fine, endpoint=False)

    def seg_ramp_up(th):  # 加速段形状 0→1
        return smootherstep((th - a1) / max(accel_frac, 1e-9))

    def seg_ramp_down(th):  # 减速段形状 1→0
        return 1.0 - smootherstep((th - a3) / max(decel_frac, 1e-9))

    w = np.zeros_like(theta)
    m_b = (theta >= a1) & (theta < a2)
    m_c = (theta >= a2) & (theta < a3)
    m_d = (theta >= a3) & (theta < a4)
    w[m_b] = L * seg_ramp_up(theta[m_b])
    w[m_c] = L                      # 同步区：斜率 = 定长 L（核心设计规则）
    w[m_d] = L * seg_ramp_down(theta[m_d])

    # 返回段：负向平滑梯形（两端斜率为零，峰值 R 由"回到原位"面积约束确定）
    ret_len = 1.0 - a4
    u = (theta[theta >= a4] - a4) / ret_len
    ramp = 0.35                     # 返回梯形的加减速段占整段比例
    shape = smootherstep(u / ramp) * smootherstep((1.0 - u) / ramp)
    dtheta = 1.0 / n_fine
    forward_area = float(np.sum(np.maximum(w, 0.0)) * dtheta)   # 前进总面积
    shape_area = float(np.sum(shape) * dtheta)
    R = forward_area / shape_area                                # 返回段峰值斜率
    w[theta >= a4] = -R * shape

    # 数值积分得位移 y(θ)；扣除累积漂移使 y(1)=y(0)=0（周期闭合）
    y = np.cumsum(w) * dtheta
    drift = np.linspace(0.0, y[-1], n_fine)
    y -= drift
    y -= y[0]

    seg_info = {
        "theta_standby_end": a1,
        "theta_sync_start": a2,
        "theta_cut": (a2 + a3) / 2.0,     # 落刀点：同步区中点
        "theta_sync_end": a3,
        "theta_return_start": a4,
        "sync_slope": L,
        "peak_blade_travel_mm": float(np.max(y)),
        "peak_return_slope": float(R),
        "product_length_mm": L,
        "sync_window_mm": float(sync_window_mm),
    }

    # ---- 采样成 n_points 规格的工程凸轮表（含首尾闭合点）----
    xs = np.linspace(0.0, L, n_points)
    ys = np.interp(xs, theta * L, y)
    ys[0] = ys[-1] = 0.0
    return CamTable(name, xs, ys), seg_info


# ---------------------------------------------------------------------------
# 电子凸轮管理器：多表存储 + 运行中切换
# ---------------------------------------------------------------------------
class ElectronicCam:
    """管理多张凸轮表；运行中调用 switch_table() 即可在下个周期生效"""

    def __init__(self):
        self.tables: dict[str, CamTable] = {}
        self.active_name: str | None = None

    def add_table(self, table: CamTable) -> None:
        self.tables[table.name] = table
        if self.active_name is None:
            self.active_name = table.name

    def switch_table(self, name: str) -> None:
        """切换当前激活表（模拟 HMI 上选择新凸轮表 / 换产）"""
        if name not in self.tables:
            raise KeyError(f"凸轮表 {name} 不存在，已注册：{list(self.tables)}")
        self.active_name = name

    @property
    def active(self) -> CamTable:
        if self.active_name is None:
            raise RuntimeError("尚未注册任何凸轮表")
        return self.tables[self.active_name]

    def evaluate(self, master_pos: float) -> float:
        return self.active.evaluate(master_pos)


# ---------------------------------------------------------------------------
# 飞剪参数 / 结果
# ---------------------------------------------------------------------------
@dataclass
class ShearParams:
    """飞剪实验参数"""

    product_length_mm: float = 600.0   # 定长：每个循环输送带走过的长度
    sync_window_mm: float = 120.0      # 同步区长度（切刀期间刀随带走的距离）
    belt_speed_mm_s: float = 700.0     # 输送带（虚拟主轴）速度
    kp_shear: float = 80.0             # 剪切轴位置环增益 (1/s)
    dt: float = 0.001                  # 控制周期 (s)
    sim_cycles: int = 3                # 仿真循环数（切割次数）
    switch_to_length_mm: float | None = None  # 若非 None：第 1 刀后切换到该定长的表
    standby_frac: float = 0.08         # 表分段参数（透传给生成器）
    accel_frac: float = 0.12
    decel_frac: float = 0.10
    n_table_points: int = 256          # 凸轮表点数


@dataclass
class ShearResult:
    """飞剪仿真结果：曲线数据 + 关键指标"""

    params: ShearParams
    t: np.ndarray                  # 时间 (s)
    master_pos: np.ndarray         # 输送带（主轴）位置 (mm，累计值)
    blade_cmd: np.ndarray          # 剪切轴指令位置（凸轮输出）
    blade_act: np.ndarray          # 剪切轴实际位置
    blade_vel: np.ndarray          # 剪切轴实际速度 (mm/s)
    in_sync: np.ndarray            # 是否处于同步窗口（布尔）
    cut_times: list                # 每次落刀时刻 (s)
    table_names: list              # 记录过程中用到的凸轮表名（含切换过程）
    switch_time: float | None = None  # 换表时刻 (s)；未换表为 None
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def finalize_metrics(self) -> dict:
        """
        计算同步段速度误差与节拍指标。

        节拍口径：相邻两次落刀的间隔为"一个周期"。若过程中发生过换表，
        则横跨换表点的那一段属于新旧配方的过渡期（既不是旧定长的节拍，
        也不是新定长的节拍），单独计数为 transition_cycles，不混入
        稳态节拍统计——否则会拉偏 min/max，得出错误的节拍结论。
        """
        p = self.params
        belt = p.belt_speed_mm_s
        err = np.abs(self.blade_vel[self.in_sync] - belt)
        cycle_theory = p.product_length_mm / belt * 1000.0  # ms（主表理论节拍）

        cuts = np.array(self.cut_times)
        transition_cycles = 0
        if len(cuts) >= 2:
            cycles = np.diff(cuts) * 1000.0
            if self.switch_time is not None:
                # 落刀区间 [cuts[i], cuts[i+1]) 包含换表时刻 → 判为过渡周期
                is_transition = (cuts[:-1] <= self.switch_time) & (
                    self.switch_time < cuts[1:]
                )
                transition_cycles = int(np.sum(is_transition))
                steady = cycles[~is_transition]
            else:
                steady = cycles
            if len(steady) > 0:
                cycle_mean = float(np.mean(steady))
                cycle_min = float(np.min(steady))
                cycle_max = float(np.max(steady))
            else:  # 极端情况：全部是过渡段
                cycle_mean = cycle_min = cycle_max = float("nan")
        else:
            cycle_mean = cycle_min = cycle_max = float("nan")

        # 换表后的新配方理论节拍（供对照；未换表则与主表一致）
        if p.switch_to_length_mm is not None and self.switch_time is not None:
            cycle_theory_switched = p.switch_to_length_mm / belt * 1000.0
        else:
            cycle_theory_switched = cycle_theory

        self.metrics = {
            "cuts": len(self.cut_times),
            "belt_speed_mm_s": belt,
            "sync_err_mean_mm_s": float(np.mean(err)) if len(err) else 0.0,
            "sync_err_max_mm_s": float(np.max(err)) if len(err) else 0.0,
            "cycle_theory_ms": cycle_theory,
            "cycle_mean_ms": cycle_mean,
            "cycle_min_ms": cycle_min,
            "cycle_max_ms": cycle_max,
            "transition_cycles": transition_cycles,
            "cycle_theory_switched_ms": cycle_theory_switched,
            "tables_used": "+".join(dict.fromkeys(self.table_names)),
        }
        return self.metrics

    # ------------------------------------------------------------------
    def to_plot_series(self, max_points: int = 1500) -> dict:
        """降采样为适合 Web 图表绘制的数据系列"""
        idx = downsample_slice(len(self.t), max_points)
        return {
            "t": [round(v, 4) for v in self.t[idx]],
            "master_pos": [round(v, 2) for v in self.master_pos[idx]],
            "blade_cmd": [round(v, 3) for v in self.blade_cmd[idx]],
            "blade_act": [round(v, 3) for v in self.blade_act[idx]],
            "blade_vel": [round(v, 2) for v in self.blade_vel[idx]],
            "in_sync": [bool(v) for v in self.in_sync[idx]],
            "cut_times": [round(v, 4) for v in self.cut_times],
            "switch_time": None if self.switch_time is None else round(self.switch_time, 4),
            "params": {
                "belt_speed_mm_s": self.params.belt_speed_mm_s,
                "product_length_mm": self.params.product_length_mm,
                "sync_window_mm": self.params.sync_window_mm,
            },
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
        }


# ---------------------------------------------------------------------------
# 飞剪应用主体
# ---------------------------------------------------------------------------
class FlyingShear:
    """
    飞剪应用仿真：
      输送带 = 虚拟主轴（匀速点动，位置持续增长）
      剪切轴 = 从轴，每周期按凸轮表执行
      "待机 → 加速追赶 → 同步(落刀) → 快速返回"
    """

    def __init__(self, params: ShearParams | None = None):
        """控制周期等全部参数统一收敛到 ShearParams，避免 dt 双通道配置"""
        self.p = params if params is not None else ShearParams()

    # ------------------------------------------------------------------
    def _make_master(self) -> VirtualAxis:
        """输送带虚拟主轴：匀速运动，不设上限（软限位放大到 1e9）"""
        m = VirtualAxis(
            AxisConfig(
                name="BELT_MASTER",
                max_vel=self.p.belt_speed_mm_s * 1.2,
                max_acc=5000.0,
                max_jerk=100000.0,
                use_s_curve=True,
                soft_limit_min=-1e9,
                soft_limit_max=1e9,
                following_error_alarm=0,
            ),
            dt=self.p.dt,
        )
        m.enable()
        return m

    def _make_shear_axis(self, peak_travel: float) -> VirtualAxis:
        """剪切从轴：闭环参数可配，行程按凸轮表峰值设置"""
        ax = VirtualAxis(
            AxisConfig(
                name="SHEAR",
                kp_pos=self.p.kp_shear,
                max_vel=max(3000.0, self.p.belt_speed_mm_s * 3.0),
                max_acc=20000.0,
                max_jerk=400000.0,
                use_s_curve=False,   # 从轴直通跟随，规划来自凸轮表
                soft_limit_min=-200.0,
                soft_limit_max=peak_travel + 200.0,
                following_error_alarm=0,  # 飞剪工况滞后天然较大，此处不使能单轴超差
            ),
            dt=self.p.dt,
        )
        ax.enable()
        ax.set_following_stream(True)
        return ax

    # ------------------------------------------------------------------
    def run(self, verbose: bool = False) -> ShearResult:
        """执行完整飞剪仿真并返回结果"""
        p = self.p
        dt = p.dt

        # 1) 生成凸轮表（主表 + 可选的第二定长表用于"换产切换"演示）
        table1, seg1 = build_flying_shear_table(
            p.product_length_mm, p.sync_window_mm,
            p.standby_frac, p.accel_frac, p.decel_frac, p.n_table_points,
            name=f"SHEAR_L{p.product_length_mm:.0f}",
        )
        cam = ElectronicCam()
        cam.add_table(table1)
        if p.switch_to_length_mm:
            table2, _ = build_flying_shear_table(
                p.switch_to_length_mm, min(p.sync_window_mm, p.switch_to_length_mm * 0.2),
                p.standby_frac, p.accel_frac, p.decel_frac, p.n_table_points,
                name=f"SHEAR_L{p.switch_to_length_mm:.0f}",
            )
            cam.add_table(table2)

        # 2) 创建主轴与剪切轴
        master = self._make_master()
        shear = self._make_shear_axis(seg1["peak_blade_travel_mm"])
        master.jog(p.belt_speed_mm_s)  # 输送带匀速开动

        theta_cut = seg1["theta_cut"]           # 落刀相位
        win = p.sync_window_mm / p.product_length_mm
        win_inner = win * 0.4                    # 统计用的内窗半宽（避开加减速边缘）

        total_steps = int((p.sim_cycles + 1.5) * p.product_length_mm / p.belt_speed_mm_s / dt)
        rec_every = max(1, int(0.002 / dt))      # 约 2ms 记录一个点

        t_l, m_l, bc_l, ba_l, bv_l, syn_l = [], [], [], [], [], []
        table_l: list[str] = []
        cut_times: list[float] = []
        prev_phase = 0.0
        switched = False
        switch_time: float | None = None
        cam_offset = 0.0  # 换表定相偏移：换表点被记为新表的相位零点（见下方说明）

        for k in range(total_steps):
            master.step()

            # 凸轮插值 → 从轴指令流 → 闭环。
            # 用 m_eff = 主轴位置 - 换表偏移 查表：未换表时 offset=0 行为不变；
            # 换表后新周期从换表点重新开始（相位归零），两张表在换表点都满足
            # y=0（刀具待机位），因此刀的目标位置连续、无跳变。
            m_eff = master.cmd_pos - cam_offset
            target = cam.evaluate(m_eff)
            shear.set_stream_target(target)
            shear.step()

            # 相位推进与事件检测（处理跨周期回绕）
            L_active = cam.active.period
            phase = math.fmod(m_eff, L_active) / L_active
            crossed_cut = (
                prev_phase < theta_cut <= phase
                if phase >= prev_phase
                else (prev_phase < theta_cut) or (theta_cut <= phase)
            )
            if crossed_cut:
                cut_times.append(k * dt)
            wrapped = phase < prev_phase  # 本周期是否发生回绕（先判回绕再更新缓存）
            prev_phase = phase

            # 换产切表：在周期回绕边界（刀具已回到待机位）执行。
            # 关键：同时把换表点记为新表的相位零点（重新定相/re-indexing）——
            # 若不这样做，新表周期更长会使相位突然"前跳"，被误判为跨过落刀
            # 相位而虚记一刀（评审发现的真实缺陷）。重新定相后相位平滑延续，
            # 刀目标位置连续，落刀序列严格按新定长推进。
            if (
                p.switch_to_length_mm is not None
                and not switched
                and len(cut_times) >= 1
                and wrapped
            ):
                cam.switch_table(f"SHEAR_L{p.switch_to_length_mm:.0f}")
                cam_offset = master.cmd_pos  # 新表从当前主轴位置重新计周期
                switch_time = k * dt
                switched = True

            # 同步内窗标记（速度误差统计口径）
            dist = abs(phase - theta_cut)
            dist = min(dist, 1.0 - dist)  # 环形距离
            in_sync_now = dist <= win_inner

            if k % rec_every == 0:
                t_l.append(k * dt)
                m_l.append(master.cmd_pos)
                bc_l.append(shear.cmd_pos)
                ba_l.append(shear.act_pos)
                bv_l.append(shear.act_vel)
                syn_l.append(in_sync_now)
                table_l.append(cam.active_name)

            if verbose and k % int(0.25 / dt) == 0:
                print(
                    f"t={k*dt:6.3f}s 带={master.cmd_pos:8.1f}mm 刀cmd={shear.cmd_pos:7.2f} "
                    f"act={shear.act_pos:7.2f} v={shear.act_vel:8.1f}"
                )

        result = ShearResult(
            params=p,
            t=np.array(t_l),
            master_pos=np.array(m_l),
            blade_cmd=np.array(bc_l),
            blade_act=np.array(ba_l),
            blade_vel=np.array(bv_l),
            in_sync=np.array(syn_l, dtype=bool),
            cut_times=cut_times,
            table_names=table_l,
            switch_time=switch_time,
        )
        result.finalize_metrics()
        return result


# ---------------------------------------------------------------------------
# 自测试：直接运行本文件
# python cam/electronic_cam.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("电子凸轮 + 飞剪 自测试")
    print("=" * 72)

    # 1) 凸轮表静态检查：同步段"物理斜率"应≈1.0（刀每前进1mm对应带走1mm，即同速）
    tbl, info = build_flying_shear_table(600.0, 120.0)
    th_cut = info["theta_cut"]
    probe_x = np.array([600.0 * (th_cut - 0.02), 600.0 * (th_cut + 0.02)])
    probe_y = [tbl.evaluate(x) for x in probe_x]
    slope = (probe_y[1] - probe_y[0]) / (probe_x[1] - probe_x[0])
    print(f"[凸轮表检查] 同步段物理斜率实测 {slope:.4f}，应为 1.0000（刀带同速）→ "
          f"{'通过' if abs(slope - 1.0) < 0.05 else '失败'}")

    # 2) 单一 speeds 飞剪运行
    fs = FlyingShear(ShearParams(belt_speed_mm_s=700.0, sim_cycles=4))
    res = fs.run()
    m = res.metrics
    print(f"\n[飞剪运行] 带速 {m['belt_speed_mm_s']} mm/s，切割 {m['cuts']} 刀，"
          f"节拍均值 {m['cycle_mean_ms']:.1f}ms（理论 {m['cycle_theory_ms']:.1f}ms）")
    print(f"[核心指标] 同步段速度误差 均值={m['sync_err_mean_mm_s']:.2f}mm/s "
          f"最大={m['sync_err_max_mm_s']:.2f}mm/s")

    # 3) 定长切换演示：第 1 刀后从 600mm 换到 900mm
    #    回归点：换表采用"重新定相"，不得虚记落刀；跨越换表的过渡周期要
    #    单独标记，稳态节拍必须严格等于新定长/带速。
    fs2 = FlyingShear(ShearParams(belt_speed_mm_s=700.0, sim_cycles=4,
                                  switch_to_length_mm=900.0))
    res2 = fs2.run()
    m2 = res2.metrics
    cuts2 = np.array(res2.cut_times)
    gaps = np.diff(cuts2) * 1000.0
    theory2 = m2["cycle_theory_switched_ms"]          # = 900/700*1000 ≈ 1285.7ms
    steady_gaps = gaps[1:]                            # 首段横跨换表，属过渡周期
    print(f"\n[换产演示] 用表 {m2['tables_used']}，"
          f"切割时刻 {[round(t, 3) for t in res2.cut_times]}s")
    print(f"[换产回归] 过渡周期 {m2['transition_cycles']} 个已单独标记；"
          f"各刀间隔 {np.round(gaps, 1)}ms；换产后稳态节拍 "
          f"{np.round(steady_gaps, 1)}ms vs 理论 {theory2:.1f}ms")
    assert m2["transition_cycles"] == 1, "应恰好有 1 个横跨换表的过渡周期"
    assert len(steady_gaps) > 0 and np.all(
        np.abs(steady_gaps - theory2) < 2.0
    ), f"换产后节拍偏离理论值：{steady_gaps}"
    assert abs(m2["cycle_mean_ms"] - theory2) < 2.0, \
        "稳态节拍统计不应被过渡周期污染"
    print("自测试完成 [OK]")
