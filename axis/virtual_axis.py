# -*- coding: utf-8 -*-
"""
虚拟伺服轴库（Virtual Servo Axis）
==================================

设计思想：没有真实驱动器，但控制律与真实伺服系统"同构"：

    上位机指令 ──► 运动规划器(梯形/S曲线, 加加速度限制) ──► 指令位置 p_cmd
                                                            │ (脉冲当量量化)
                                                            ▼
              实际位置 p_act ◄── 一阶闭环模型(位置环增益 Kp) ◄── 位置偏差 e

真实 P 位置环的闭环传函近似为一阶滞后：
        dp_act/dt = Kp * (p_cmd - p_act) - d_load
由此天然产生跟随误差，恒速时稳态跟随误差：
        e_ss = (v + d_load) / Kp      （单位/秒 ÷ 1/秒 = 单位）
两轴的 Kp 与负载可分别配置 —— 这正是龙门双轴同步偏差实验的物理根源。

本文件只依赖标准库，可直接运行自测试：python axis/virtual_axis.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# ---------------------------------------------------------------------------
# 常量定义：报警码 / 运动模式 / 曲线类型
# ---------------------------------------------------------------------------
class AlarmCode:
    """报警码定义（锁存式：触发后保持，直到人工清除且条件消失）"""

    NONE = 0            # 无报警
    SOFT_LIMIT = 1      # 软限位触发
    FOLLOWING_ERROR = 2 # 跟随误差超差
    SYNC_EXCEED = 3     # 龙门同步偏差超限（由上层控制器经 latch_alarm() 注入锁存）


class MotionMode:
    """运动模式"""

    ABSOLUTE = "absolute"   # 绝对运动：走到绝对坐标
    RELATIVE = "relative"   # 相对运动：从当前位置偏移
    JOG = "jog"             # 点动：以指定速度持续运动，直到停止指令


@dataclass
class AxisConfig:
    """虚拟伺服轴参数配置（全部可用中文注释的工程量）"""

    name: str = "AXIS"            # 轴名，用于打印与图表图例
    # ---- 机械 / 接口 ----
    pulse_per_unit: float = 1000.0  # 脉冲当量：每工程单位(mm)对应多少脉冲，如 1000脉冲/mm
    soft_limit_min: float = -10.0   # 软限位下限 (mm)
    soft_limit_max: float = 1500.0  # 软限位上限 (mm)
    # ---- 规划器限幅 ----
    max_vel: float = 800.0          # 最大速度 (mm/s)
    max_acc: float = 4000.0         # 最大加速度 (mm/s^2)
    max_jerk: float = 60000.0       # 最大加加速度 J (mm/s^3)，S 曲线用；梯形曲线视为无穷大
    use_s_curve: bool = True        # True=S 曲线(限制加加速度)；False=纯梯形速度曲线
    in_pos_window: float = 0.005    # 到位判定窗口 (mm)
    # ---- 闭环（同构于真实驱动器）----
    kp_pos: float = 40.0            # 位置环增益 (1/s)：决定跟随误差 e≈v/Kp
    load_disturbance: float = 0.0   # 等效负载扰动 (mm/s)：折算成对速度方程的拖拽项
    vel_feedforward: float = 0.0    # 速度前馈系数 0~1：补偿后 e_ss ≈ (1-Kfv)*(v+d)/Kp
    following_error_alarm: float = 5.0  # 跟随误差报警阈值 (mm)；<=0 表示关闭该保护
    # 软限位受控回退的恢复速度比例（× max_vel）：可配置，
    # 但无论配多大都会被 VirtualAxis.RECOVER_VEL_MAX_RATIO（25%）硬钳制（防误配）
    recover_vel_ratio: float = 0.2


# ---------------------------------------------------------------------------
# 虚拟伺服轴主体
# ---------------------------------------------------------------------------
class VirtualAxis:
    """
    虚拟伺服轴：规划器 + 闭环被控对象 + 保护逻辑，三部分合一。

    典型用法（每仿真步调用一次 step(dt)）::

        ax = VirtualAxis(AxisConfig(name="X1", kp_pos=40))
        ax.enable()
        ax.move_abs(300.0, vel=500)          # 下发绝对定位
        for _ in range(int(3 / dt)):          # 以 1ms 步长推进仿真
            ax.step(dt)
            print(ax.cmd_pos, ax.act_pos, ax.following_error)

    软限位报警（SOFT_LIMIT）的恢复遵循真实工程的"受控回退"惯例：
    报警态输出封锁冻结，但保留唯一运动通道 jog_recover()——仅允许朝
    限位内侧低速点动，退回限位内后锁存自动解除（见 jog_recover 说明）。
    """

    # 恢复点动速度硬上限 = max_vel × 25%（AxisConfig.recover_vel_ratio 可配，
    # 但配置值超过该比例的部分一律无效——低速回退是安全底线，防止误配超速）
    RECOVER_VEL_MAX_RATIO = 0.25
    # 解除软限位锁存的内侧滞回带 (mm)：实际位置须退回限位内并越过该
    # 滞回量才自动解除，避免贴着边界来回抖动反复触发/解除
    RECOVER_RELEASE_HYST_MM = 0.05

    def __init__(self, config: AxisConfig | None = None, dt: float = 0.001):
        self.cfg = config if config is not None else AxisConfig()
        self.dt = dt  # 仿真控制周期 (s)，与总线循环周期同构（如 1ms EtherCAT）

        # ---- 指令层状态（规划器输出）----
        self._cmd_pos: float = 0.0     # 指令位置 (mm)
        self._cmd_vel: float = 0.0     # 指令速度 (mm/s)
        self._cmd_acc: float = 0.0     # 指令加速度 (mm/s^2)

        # ---- 反馈层状态（闭环模拟输出）----
        self._act_pos: float = 0.0     # 实际位置 (mm)
        self._act_vel: float = 0.0     # 实际速度 (mm/s)
        self._loop_error: float = 0.0  # 本控制周期参与运算的位置偏差（跟随误差）

        # ---- 任务状态 ----
        self._mode: str | None = None      # 当前任务类型（MotionMode 之一或 None）
        self._target: float = 0.0          # 点到点目标位置 (mm)
        self._profile_vel: float = 0.0     # 本次任务的规划速度上限 (mm/s)
        self._jog_vel: float = 0.0         # 点动速度 (mm/s，带方向)
        self._following_stream: bool = False  # 是否处于外部目标流跟随态（龙门/凸轮从轴）

        # ---- 使能 / 报警 ----
        self._enabled: bool = False
        self._alarm: int = AlarmCode.NONE  # 锁存报警码

        # ---- 软限位受控回退（恢复）状态 ----
        self._recovering: bool = False  # 是否处于受控回退模式（SOFT_LIMIT 报警态）
        self._recover_side: int = 0     # 越界侧：+1 已越上限 / -1 已越下限 / 0 无
        self._recover_cmd_bound: float = 0.0  # 回退期间指令位置的包络钳位点 (mm)

        # ---- 统计 ----
        self._steps: int = 0               # 累计仿真步数

    # ------------------------------------------------------------------
    # 对外只读属性（供上层采集曲线 / 判断状态）
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def cmd_pos(self) -> float:
        """指令位置 (mm)"""
        return self._cmd_pos

    @property
    def act_pos(self) -> float:
        """实际位置 (mm)"""
        return self._act_pos

    @property
    def cmd_vel(self) -> float:
        """指令速度 (mm/s)"""
        return self._cmd_vel

    @property
    def act_vel(self) -> float:
        """实际速度 (mm/s)"""
        return self._act_vel

    @property
    def following_error(self) -> float:
        """
        跟随误差 (mm)：本控制周期位置环实际参与运算的偏差 e = p_cmd - p_act。
        恒速时稳态值 ≈ (v + 负载) / Kp —— 与真实伺服的采样口径一致
        （若直接用 cmd_pos-act_pos 事后相减，会差出一个周期的指令增量 v*T）。
        """
        return self._loop_error

    @property
    def cmd_pulses(self) -> int:
        """指令位置折算脉冲数（体现脉冲当量）"""
        return int(round(self._cmd_pos * self.cfg.pulse_per_unit))

    @property
    def act_pulses(self) -> int:
        """实际位置折算脉冲数"""
        return int(round(self._act_pos * self.cfg.pulse_per_unit))

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def alarm(self) -> int:
        return self._alarm

    @property
    def has_alarm(self) -> bool:
        return self._alarm != AlarmCode.NONE

    @property
    def in_position(self) -> bool:
        """到位判断：无任务且指令与实际都进入到位窗口"""
        return (
            self._mode is None
            and abs(self.following_error) <= self.cfg.in_pos_window * 20  # 放宽含闭环滞后
        )

    @property
    def is_moving(self) -> bool:
        return self._mode is not None or abs(self._cmd_vel) > 1e-6

    @property
    def recover_vel_cap(self) -> float:
        """
        软限位受控回退的速度上限 (mm/s) = max_vel × recover_vel_ratio。
        比例被钳制在 [5%, 25%] 内：上限 25% 为硬编码安全底线（防误配），
        下限 5% 防止误配成 0 导致恢复通道死锁。
        """
        ratio = min(max(self.cfg.recover_vel_ratio, 0.05), self.RECOVER_VEL_MAX_RATIO)
        return self.cfg.max_vel * ratio

    @property
    def is_recovering(self) -> bool:
        """是否处于软限位受控回退模式"""
        return self._recovering

    # ------------------------------------------------------------------
    # 使能 / 报警管理（与真实伺服操作习惯一致）
    # ------------------------------------------------------------------
    def enable(self) -> "VirtualAxis":
        """使能轴（上电）。有未清除报警时拒绝使能。"""
        if self.has_alarm:
            raise RuntimeError(f"[{self.cfg.name}] 存在报警码 {self._alarm}，请先 clear_alarm()")
        self._enabled = True
        return self

    def disable(self) -> "VirtualAxis":
        """去使能：立即封锁输出（模拟驱动器下电），并退出受控回退状态"""
        self._enabled = False
        self._recovering = False
        self._recover_side = 0
        self._abort_task()
        return self

    def clear_alarm(self) -> bool:
        """
        清除报警。只有当报警条件已经消失（例如已离开软限位区间）才允许清除，
        模拟真实系统的"故障条件不消除不能复位"。

        软限位报警的标准恢复流程是 jog_recover() 受控回退：实际位置退回
        限位内侧时锁存会自动解除；本方法仅在条件已消失时才会复位成功。
        """
        if self._alarm == AlarmCode.SOFT_LIMIT:
            inside = self.cfg.soft_limit_min <= self._act_pos <= self.cfg.soft_limit_max
            if not inside:
                return False  # 仍压在限位上，禁止复位
        self._alarm = AlarmCode.NONE
        self._recovering = False
        self._recover_side = 0
        return True

    def _latch_alarm(self, code: int, reason: str) -> None:
        """锁存报警并急停（模拟驱动器报警停机）"""
        if self._alarm == AlarmCode.NONE:  # 只记录第一个报警
            self._alarm = code
            print(f"[{self.cfg.name}] 触发报警 code={code}：{reason}，已停机锁存")
        self._abort_task()

    def latch_alarm(self, code: int, reason: str) -> "VirtualAxis":
        """
        注入一个外部（控制器级）报警并锁存，如龙门"同步偏差超限"：
        报警后本轴立即封锁输出（实际位置/速度冻结，见 step()），
        且在 clear_alarm() 之前禁止使能与任何运动指令。
        """
        self._latch_alarm(code, reason)
        return self

    def _abort_task(self) -> None:
        """终止当前运动任务，指令速度清零"""
        self._mode = None
        self._cmd_vel = 0.0
        self._cmd_acc = 0.0

    def _check_recover_release(self) -> None:
        """
        受控回退中的解除条件判定（step() 每周期调用）：
        实际位置退回限位内侧滞回带后，自动解除软限位锁存并停止回退点动；
        轴保持使能，闭环会把剩余指令误差收敛掉，随即恢复正常运动。
        """
        hyst = self.RECOVER_RELEASE_HYST_MM
        if self._recover_side > 0:
            back_inside = self._act_pos <= self.cfg.soft_limit_max - hyst
        else:
            back_inside = self._act_pos >= self.cfg.soft_limit_min + hyst
        if back_inside:
            print(
                f"[{self.cfg.name}] 实际位置 {self._act_pos:.4f}mm 已退回软限位"
                f"内侧（滞回 {hyst}mm），锁存自动解除"
            )
            self._alarm = AlarmCode.NONE
            self._recovering = False
            self._recover_side = 0
            self._abort_task()  # 停止回退点动（恢复到位即停，不允许再向外）

    # ------------------------------------------------------------------
    # 参数在线修改（用于增益失配 / 负载突变等对比实验）
    # ------------------------------------------------------------------
    def set_kp(self, kp: float) -> None:
        """在线修改位置环增益 (1/s)"""
        self.cfg.kp_pos = float(kp)

    def set_load_disturbance(self, disturbance: float) -> None:
        """在线修改等效负载扰动 (mm/s)"""
        self.cfg.load_disturbance = float(disturbance)

    def set_s_curve(self, use: bool) -> None:
        """切换梯形/S 曲线规划方式"""
        self.cfg.use_s_curve = bool(use)

    # ------------------------------------------------------------------
    # 运动指令接口
    # ------------------------------------------------------------------
    def _check_ready(self) -> None:
        if not self._enabled:
            raise RuntimeError(f"[{self.cfg.name}] 轴未使能，请先 enable()")
        if self.has_alarm:
            raise RuntimeError(f"[{self.cfg.name}] 存在报警码 {self._alarm}，禁止运动")

    def _validate_target(self, pos: float) -> float:
        """目标位置合法性检查（软限位预检）"""
        if not (self.cfg.soft_limit_min <= pos <= self.cfg.soft_limit_max):
            raise ValueError(
                f"[{self.cfg.name}] 目标位置 {pos:.3f}mm 超出软限位 "
                f"[{self.cfg.soft_limit_min}, {self.cfg.soft_limit_max}]"
            )
        return pos

    def _resolve_profile_vel(self, vel: float | None) -> float:
        """解析规划速度上限：None 用默认最大值，并钳制到轴速度上限内"""
        v = abs(vel) if vel is not None else self.cfg.max_vel
        return min(v, self.cfg.max_vel)

    def move_abs(self, pos: float, vel: float | None = None) -> "VirtualAxis":
        """绝对运动：走到绝对坐标 pos (mm)"""
        self._check_ready()
        self._mode = MotionMode.ABSOLUTE
        self._target = self._validate_target(pos)
        self._profile_vel = self._resolve_profile_vel(vel)
        return self

    def move_rel(self, delta: float, vel: float | None = None) -> "VirtualAxis":
        """相对运动：从当前指令位置偏移 delta (mm)"""
        self._check_ready()
        self._mode = MotionMode.RELATIVE
        self._target = self._validate_target(self._cmd_pos + delta)
        self._profile_vel = self._resolve_profile_vel(vel)
        return self

    def jog(self, vel: float) -> "VirtualAxis":
        """点动：以速度 vel (mm/s，带符号) 持续运动，调用 stop() 后减速停止"""
        self._check_ready()
        self._mode = MotionMode.JOG
        self._jog_vel = max(-self.cfg.max_vel, min(self.cfg.max_vel, vel))
        self._profile_vel = abs(self._jog_vel)
        return self

    def jog_recover(self, vel: float) -> "VirtualAxis":
        """
        软限位受控回退点动 —— SOFT_LIMIT 报警态下唯一放行的运动通道。

        报警锁存后输出封锁、实际位置冻结，而锁存瞬间实际位置必然在限位外，
        若无恢复通道则 clear_alarm() 的复位条件永远无法满足（复位死锁）。
        工业惯例的恢复方式是"受控回退"，本方法的安全约束：

          - 仅适用于 SOFT_LIMIT 报警且轴处于使能状态；其他报警
            （SYNC_EXCEED / FOLLOWING_ERROR）不提供此通道，语义不变；
          - 方向约束：仅允许朝限位"内侧"回退，朝外（继续深入限位）的
            指令直接拒绝；
          - 速度约束：实际回退速度被钳制到 recover_vel_cap
            （= max_vel × 比例，硬上限 25%，见类常量），指令端残余误差
            造成的闭环瞬态同样被该上限封顶（见 step() 的执行器限幅）；
          - 包络约束：回退起点把指令位置钳回限位边界，恢复期间指令位置
            绝不允许越过该点（只能原地或向内，见 _plan_step）；
          - 解除条件：实际位置退回限位内侧滞回带（RECOVER_RELEASE_HYST_MM）
            后自动解除锁存并停止回退点动，轴保持使能、恢复正常运动。

        若实际位置已在限位内（如报警后由外部方式回位），直接复位报警。
        """
        if self._alarm != AlarmCode.SOFT_LIMIT:
            raise RuntimeError(
                f"[{self.cfg.name}] 受控回退仅适用于 SOFT_LIMIT 报警"
                f"（当前报警码 {self._alarm}），语义不变"
            )
        if not self._enabled:
            raise RuntimeError(
                f"[{self.cfg.name}] 轴已去使能，受控回退通道不可用；"
                "请按真实驱动器惯例重建/上电复位（报警未清不能上电）"
            )

        # 越界侧判定（以实际位置为准——软限位本就以实际位置越界为触发口径）
        if self._act_pos > self.cfg.soft_limit_max:
            side = 1    # 已越过上限：只允许向内（负速度）回退
        elif self._act_pos < self.cfg.soft_limit_min:
            side = -1   # 已越过下限：只允许向内（正速度）回退
        else:
            # 实际位置已在限位内：故障条件已消失，直接复位锁存
            self.clear_alarm()
            print(f"[{self.cfg.name}] 实际位置已在软限位内，报警直接复位")
            return self

        if vel * side >= 0.0:
            raise ValueError(
                f"[{self.cfg.name}] 受控回退只允许朝限位内侧方向"
                f"（此处应给{'负' if side > 0 else '正'}速度），拒绝朝外指令"
            )

        cap = self.recover_vel_cap
        v = min(abs(vel), cap)  # 低速上限硬钳制：请求再大也不超过 cap
        self.set_following_stream(False)  # 回退是轴自身点动，先脱离任何目标流
        self._recovering = True
        self._recover_side = side
        # 指令位置钳回限位边界：清除锁存时刻指令端可能越界的残量，
        # 使恢复期间闭环不会被残余指令误差拉向限位外
        if side > 0:
            self._cmd_pos = min(self._cmd_pos, self.cfg.soft_limit_max)
            self._recover_cmd_bound = self.cfg.soft_limit_max
        else:
            self._cmd_pos = max(self._cmd_pos, self.cfg.soft_limit_min)
            self._recover_cmd_bound = self.cfg.soft_limit_min
        self._mode = MotionMode.JOG
        self._jog_vel = -side * v
        self._profile_vel = v
        print(
            f"[{self.cfg.name}] 软限位受控回退：以 {v:.2f}mm/s "
            f"（上限 {cap:.2f}）朝限位内侧点动"
        )
        return self

    def set_stream_target(self, target: float, vel: float | None = None) -> None:
        """
        刷新外部目标流的最新目标点（龙门从轴 / 电子凸轮从轴每个控制周期调用）。
        规划器会在后续周期内朝该目标做带限幅的平滑插补，等价于电子齿轮 1:1 跟随。
        """
        self.set_following_stream(True)
        self._mode = MotionMode.JOG  # 借用"持续运动"语义，速度由位置插补决定
        self._jog_vel = 0.0          # 纯跟随：不使用固定点动速度
        if vel is not None:
            self._profile_vel = min(abs(vel), self.cfg.max_vel)
        self._target = float(target)

    def stop(self) -> "VirtualAxis":
        """停令：按最大加速度减速停车（点动/连续跟随的正常退出方式）"""
        self._mode = None
        return self

    # ------------------------------------------------------------------
    # 核心一步：每个控制周期调用一次
    # ------------------------------------------------------------------
    def step(self, dt: float | None = None) -> None:
        """
        推进一个控制周期的仿真：
          1) 规划器生成新指令位置（梯形/S 曲线、限幅）
          2) 软限位与跟随误差保护检查
          3) 一阶闭环模型更新实际位置（产生跟随误差）
        """
        T = self.dt if dt is None else dt
        self._steps += 1

        # ---------- 1. 运动规划器 ----------
        if self._enabled and (not self.has_alarm or self._recovering):
            self._plan_step(T)
        else:
            # 未使能或报警（非受控回退）：指令冻结，模拟驱动器封锁脉冲。
            # 受控回退（软限位恢复）是唯一例外：报警态下仅放行朝限位
            # 内侧的低速点动，见 jog_recover()。
            self._cmd_vel = 0.0
            self._cmd_acc = 0.0

        # ---------- 2. 保护逻辑 ----------
        if self._alarm == AlarmCode.NONE:
            # 软限位：以实际位置越界为准（模拟撞机）
            if self._act_pos > self.cfg.soft_limit_max or self._act_pos < self.cfg.soft_limit_min:
                self._latch_alarm(AlarmCode.SOFT_LIMIT, "实际位置触及软限位")
            # 跟随误差超差（阈值>0 时启用）
            elif (
                self.cfg.following_error_alarm > 0
                and abs(self.following_error) > self.cfg.following_error_alarm
            ):
                self._latch_alarm(AlarmCode.FOLLOWING_ERROR, "跟随误差超过设定阈值")

        # ---------- 3. 闭环被控对象（一阶模型）----------
        if (not self._enabled or self.has_alarm) and not self._recovering:
            # 未使能或报警锁存（非受控回退）：驱动器封锁输出（模拟下电/抱闸），
            # 被控对象不再受力 —— 实际位置/实际速度冻结，绝不会被闭环拉向
            # 指令位置。（否则"去使能立即封锁输出"就名不符实：轴会继续
            # "伺服"到冻结的指令位置，软限位报警后还会表现为自动退回限位内。）
            self._act_vel = 0.0
            cmd_quantized = round(self._cmd_pos * self.cfg.pulse_per_unit) / self.cfg.pulse_per_unit
            self._loop_error = cmd_quantized - self._act_pos  # 遥测：冻结后的环内偏差
            return
        # 脉冲当量量化：真实系统里指令是离散脉冲，这里把指令量化到 1 个脉冲分辨率
        ppu = self.cfg.pulse_per_unit
        cmd_quantized = round(self._cmd_pos * ppu) / ppu
        err = cmd_quantized - self._act_pos           # 位置偏差 e (mm)
        self._loop_error = err                        # 记录环内误差（对外报告口径）
        # 速度前馈：真实伺服常见配置，抵消恒速下的稳态跟随误差
        ff = self.cfg.vel_feedforward * (self._cmd_vel + self.cfg.load_disturbance)
        self._act_vel = self.cfg.kp_pos * err + ff - self.cfg.load_disturbance
        # 执行机构速度限幅（模拟电机最高转速）
        vmax_plant = self.cfg.max_vel * 1.2
        if self._act_vel > vmax_plant:
            self._act_vel = vmax_plant
        elif self._act_vel < -vmax_plant:
            self._act_vel = -vmax_plant
        # 受控回退的驱动器级限速：实际运动速度被硬性钳制在恢复速度上限内，
        # 即使指令端钳位造成残余误差瞬态，也绝不会超速（低速回退安全底线）
        if self._recovering:
            cap = self.recover_vel_cap
            if self._act_vel > cap:
                self._act_vel = cap
            elif self._act_vel < -cap:
                self._act_vel = -cap
        self._act_pos += self._act_vel * T

        # 受控回退到位检查：实际位置退回限位内侧滞回带 → 自动解除锁存
        if self._recovering:
            self._check_recover_release()

    def _plan_step(self, T: float) -> None:
        """单周期规划：根据任务模式计算本周期指令速度/位置增量"""
        cfg = self.cfg

        if self._mode == MotionMode.JOG:
            if self._following_stream:
                # 外部目标流跟随（电子齿轮/凸轮/龙门从轴）：指令直通，见函数内说明。
                self._follow_stream(T)
                return  # 位置已在直通中刷新，不再走通用积分
            elif self._jog_vel != 0.0:
                # 纯点动：朝点动速度做加速度限幅逼近
                self._ramp_velocity(self._jog_vel, T)
            else:
                self._ramp_velocity(0.0, T)
        elif self._mode in (MotionMode.ABSOLUTE, MotionMode.RELATIVE):
            remaining = self._target - self._cmd_pos
            dist = abs(remaining)
            if dist <= max(abs(self._cmd_vel) * T, cfg.in_pos_window):
                # 本周期即可到达：贴合目标并停车
                self._cmd_pos = self._target
                self._cmd_vel = 0.0
                self._cmd_acc = 0.0
                self._mode = None
                return
            direction = 1.0 if remaining > 0 else -1.0
            # 减速约束：保证剩余距离内能把速度降到 0（v^2 = 2*a*d）
            v_decel = math.sqrt(2.0 * cfg.max_acc * dist)
            v_allow = min(self._profile_vel, v_decel)
            self._ramp_velocity(direction * v_allow, T)
        else:
            # 无任务：指令速度回零（如停令后的减速收尾）
            self._ramp_velocity(0.0, T)

        self._cmd_pos += self._cmd_vel * T

        if self._recovering:
            # 受控回退的安全底线（包络约束）：指令位置绝不越过回退钳位点
            # （限位边界）——恢复过程中指令只能原地或朝限位内侧移动，
            # 任何情况（含规划器异常）下都不允许更深入限位。
            if self._recover_side > 0 and self._cmd_pos > self._recover_cmd_bound:
                self._cmd_pos = self._recover_cmd_bound
                self._cmd_vel = min(self._cmd_vel, 0.0)
            elif self._recover_side < 0 and self._cmd_pos < self._recover_cmd_bound:
                self._cmd_pos = self._recover_cmd_bound
                self._cmd_vel = max(self._cmd_vel, 0.0)
        else:
            # 点动模式下也要防止冲出软限位（规划层面预判）
            nxt = self._cmd_pos + self._cmd_vel * T
            if nxt > cfg.soft_limit_max or nxt < cfg.soft_limit_min:
                self._cmd_vel = 0.0

    def _follow_stream(self, T: float) -> None:
        """
        外部目标流跟随（电子齿轮 1:1 的真实行为）：
          与真实系统同构 —— 主站（CNC/主轴）已完成轨迹插补，从轴的"位置指令"
          就是主站流式下发的目标值本身，从轴不再自行规划（否则会引入额外的
          指令级滞后与噪声，破坏同步分析）。跟随误差完全由伺服闭环一阶模型
          产生：e ≈ v/Kp。仅保留速度钳制以模拟驱动器速度极限（正常平滑
          目标流不会触发；一旦触发即等价于"失步"，由超差保护兜底）。
        """
        prev_pos = self._cmd_pos
        prev_vel = self._cmd_vel
        new_cmd = self._target

        # 速度钳制（驱动器速度极限）
        dv = (new_cmd - prev_pos) / T
        if dv > self.cfg.max_vel:
            new_cmd = prev_pos + self.cfg.max_vel * T
        elif dv < -self.cfg.max_vel:
            new_cmd = prev_pos - self.cfg.max_vel * T

        self._cmd_pos = new_cmd
        self._cmd_vel = (new_cmd - prev_pos) / T           # 遥测：指令速度
        self._cmd_acc = (self._cmd_vel - prev_vel) / T     # 遥测：指令加速度

    def _ramp_velocity(self, v_des: float, T: float) -> None:
        """
        把指令速度向期望值 v_des 逼近：
          - 先做加速度限幅（梯形曲线）
          - 若开启 S 曲线，再做加加速度限幅（加速度本身平滑变化）
        """
        cfg = self.cfg
        a_des = (v_des - self._cmd_vel) / T
        if a_des > cfg.max_acc:
            a_des = cfg.max_acc
        elif a_des < -cfg.max_acc:
            a_des = -cfg.max_acc

        if cfg.use_s_curve:
            # 加加速度限制：每周期加速度变化量不超过 max_jerk * T（S 曲线的本质）
            da_max = cfg.max_jerk * T
            da = a_des - self._cmd_acc
            if da > da_max:
                da = da_max
            elif da < -da_max:
                da = -da_max
            a_new = self._cmd_acc + da
        else:
            a_new = a_des  # 梯形：加速度允许跳变

        v_before = self._cmd_vel
        v_new = v_before + a_new * T
        # 过冲保护：本周期速度跨越了期望值 → 直接贴齐期望速度并归零加速度
        if (v_before - v_des) * (v_new - v_des) <= 0.0 and v_before != v_des:
            self._cmd_vel = v_des
            self._cmd_acc = 0.0
        else:
            self._cmd_vel = v_new
            self._cmd_acc = a_new

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------
    def set_following_stream(self, on: bool) -> None:
        """声明进入"外部目标流跟随"状态（龙门从轴/凸轮从轴调用）"""
        self._following_stream = bool(on)

    def get_state(self) -> dict:
        """快照当前状态（供记录曲线 / Web 接口使用）"""
        return {
            "name": self.cfg.name,
            "t": self._steps * self.dt,
            "cmd_pos": self._cmd_pos,
            "act_pos": self._act_pos,
            "cmd_vel": self._cmd_vel,
            "act_vel": self._act_vel,
            "following_error": self.following_error,
            "enabled": self._enabled,
            "alarm": self._alarm,
        }

    def wait_done(self, timeout: float = 30.0, realtime: bool = False) -> bool:
        """
        便捷阻塞推进：不断 step 直到到位或超时。
        realtime=False 时为纯仿真加速推进（推荐）；True 时按真实时间等待。
        """
        import time

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            self.step()
            if not self.is_moving and self.in_position:
                return True
            if realtime:
                time.sleep(self.dt)
        return False


# ---------------------------------------------------------------------------
# 通用工具：图表数据等距降采样（gantry / cam 的 to_plot_series 共用）
# ---------------------------------------------------------------------------
def downsample_slice(n: int, max_points: int) -> slice:
    """生成长度为 n 的序列的等距降采样切片，压缩到约 max_points 个点"""
    step = max(1, n // max_points)
    return slice(0, n, step)


# ---------------------------------------------------------------------------
# 自测试：直接运行本文件可看到一条完整的 S 曲线定位过程
# python axis/virtual_axis.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Windows 控制台默认 GBK 编码，强制 UTF-8 输出避免中文/符号乱码
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("虚拟伺服轴自测试：S 曲线定位 + 跟随误差观察")
    print("=" * 60)

    dt = 0.001
    ax = VirtualAxis(
        AxisConfig(
            name="TEST",
            kp_pos=40.0,
            load_disturbance=0.0,
            vel_feedforward=0.0,
            following_error_alarm=0,  # 自测试关闭超差保护，便于观察大误差
        ),
        dt=dt,
    )
    ax.enable()
    ax.move_abs(300.0, vel=200.0)

    log = []
    steps = int(2.5 / dt)
    for i in range(steps):
        ax.step()
        if i % 50 == 0:  # 每 50ms 记录一点
            log.append(ax.get_state())

    print(f"{'时间(s)':>8} {'指令位置':>10} {'实际位置':>10} {'跟随误差':>10} {'指令速度':>10}")
    for s in log:
        print(
            f"{s['t']:8.3f} {s['cmd_pos']:10.3f} {s['act_pos']:10.3f} "
            f"{s['following_error']:10.4f} {s['cmd_vel']:10.2f}"
        )

    # 恒速段理论跟随误差 e = v / Kp = 500/40 = 12.5mm 太大会触发报警，
    # 自测试用较低速度演示数量级正确性：
    ax2 = VirtualAxis(AxisConfig(name="FE", kp_pos=40.0, following_error_alarm=0), dt=dt)
    ax2.enable()
    ax2.jog(80.0)  # 80mm/s 恒速
    for _ in range(int(2.0 / dt)):
        ax2.step()
    theory = 80.0 / 40.0
    print("\n[跟随误差验证] 恒速 80mm/s, Kp=40(1/s)")
    print(f"  理论稳态误差 = v/Kp = {theory:.4f} mm")
    print(f"  仿真实测误差 = {ax2.following_error:.4f} mm")
    assert abs(ax2.following_error - theory) < 0.02 * theory, "跟随误差与理论不符！"

    # ---- 软限位受控回退恢复回归（复审报告 05 N-P2-1：报警不可复位死锁）----
    # 场景：点动运行中把软限位收紧到实际位置之内（模拟坐标系偏置后限位
    # 收紧），实际位置已在限位外 → SOFT_LIMIT 锁存、输出封锁冻结。
    print("\n[软限位受控回退验证]")
    ax3 = VirtualAxis(
        AxisConfig(
            name="SL",
            kp_pos=40.0,
            max_vel=800.0,
            soft_limit_min=-10.0,
            soft_limit_max=50.0,
            following_error_alarm=0,  # 本用例只关注软限位保护
        ),
        dt=dt,
    )
    ax3.enable()
    ax3.jog(300.0)
    for _ in range(int(0.15 / dt)):
        ax3.step()
    ax3.cfg.soft_limit_max = round(ax3.act_pos - 5.0, 3)  # 收紧到实际位置内 5mm
    ax3.step()
    assert ax3.alarm == AlarmCode.SOFT_LIMIT, "收紧限位后应触发 SOFT_LIMIT 锁存"
    assert ax3.act_pos > ax3.cfg.soft_limit_max, "锁存时实际位置应在限位外"
    frozen = ax3.act_pos
    for _ in range(int(0.3 / dt)):
        ax3.step()
    assert ax3.act_pos == frozen and ax3.act_vel == 0.0, "报警锁存后实际位置必须冻结"
    assert ax3.clear_alarm() is False, "条件未消失（仍在限位外）时禁止复位"
    for bad_vel in (+50.0, -50.0):
        try:
            ax3.jog(bad_vel)
            raise AssertionError("报警态普通 jog 应被拒绝")
        except RuntimeError:
            pass
    try:
        ax3.jog_recover(+30.0)
        raise AssertionError("朝外（继续深入限位）的受控回退应被拒绝")
    except ValueError:
        pass

    cap = ax3.recover_vel_cap  # 800 × 20% = 160 mm/s
    start_pos = ax3.act_pos
    ax3.jog_recover(-2000.0)  # 请求速度远超上限 → 实际执行必须被钳到 cap
    rec_pos: list[float] = []
    rec_vel: list[float] = []
    released = False
    for _ in range(int(2.0 / dt)):
        ax3.step()
        rec_pos.append(ax3.act_pos)
        rec_vel.append(ax3.act_vel)
        if not ax3.has_alarm:
            released = True
            break
    assert released, "退回限位内后锁存应自动解除"
    peak = max(abs(v) for v in rec_vel)
    assert peak <= cap + 1e-6, f"回退实际速度 {peak:.2f} 必须≤硬上限 {cap:.2f}"
    assert peak > cap * 0.95, "回退速度应确实跑到上限（钳制生效），而非从未加速"
    assert max(rec_pos) <= start_pos + 1e-9, "回退全程不得比锁存点更深入限位"
    assert ax3.act_pos < ax3.cfg.soft_limit_max, "解除时实际位置应已在限位内"
    assert ax3.is_enabled, "锁存解除后轴应保持使能"

    # 解除后恢复正常运动：点动 + 绝对定位均应可用并真实到位
    ax3.jog(-100.0)
    for _ in range(int(0.1 / dt)):
        ax3.step()
    ax3.stop()
    ax3.move_abs(10.0, vel=200.0)
    settled = False
    for _ in range(int(3.0 / dt)):
        ax3.step()
        if not ax3.is_moving and ax3.in_position:
            settled = True
            break
    assert settled and ax3.alarm == AlarmCode.NONE, "解除锁存后应恢复正常定位运动"
    print(
        f"  锁存于 {frozen:.3f}mm（上限 {ax3.cfg.soft_limit_max:.3f}）→ 朝外拒绝 →"
        f" 回退峰值 {peak:.1f}mm/s = 硬上限 {cap:.1f} → 回位自动解除 → 正常定位 [OK]"
    )

    print("自测试通过 [OK]")
