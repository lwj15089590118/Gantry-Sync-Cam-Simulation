# -*- coding: utf-8 -*-
"""
axis 包：虚拟伺服轴库。

对外暴露：
    VirtualAxis  -- 虚拟伺服轴（规划器 + 闭环模拟 + 限位报警）
    AxisConfig   -- 轴参数配置（脉冲当量 / 增益 / 负载扰动 / 运动限幅等）

说明：本库不依赖任何真实驱动器，全部运动学/动力学行为用 Python 数值积分
模拟，但控制律结构与真实伺服系统同构（脉冲当量、梯形/S 曲线规划、
位置环增益产生跟随误差、负载扰动），可用于同步控制算法的教学与验证。
"""

from .virtual_axis import AxisConfig, VirtualAxis, AlarmCode, MotionMode

__all__ = ["VirtualAxis", "AxisConfig", "AlarmCode", "MotionMode"]
