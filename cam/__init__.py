# -*- coding: utf-8 -*-
"""
cam 包：电子凸轮与飞剪应用仿真。

对外暴露：
    CamTable        -- 电子凸轮表（256 点主轴位移→从轴位移，线性插值）
    ElectronicCam   -- 凸轮表管理器（多表存储 / 运行中切换 / 比例缩放）
    FlyingShear     -- 飞剪应用（输送带虚拟主轴 + 剪切轴凸轮跟随 + 指标统计）
    ShearParams     -- 飞剪参数
    ShearResult     -- 飞剪仿真结果（曲线 + 指标）
"""

from .electronic_cam import (
    CamTable,
    ElectronicCam,
    FlyingShear,
    ShearParams,
    ShearResult,
)

__all__ = [
    "CamTable",
    "ElectronicCam",
    "FlyingShear",
    "ShearParams",
    "ShearResult",
]
