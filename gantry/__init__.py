# -*- coding: utf-8 -*-
"""
gantry 包：龙门双轴同步控制仿真。

对外暴露：
    GantryController -- 龙门双轴同步控制器（主从指令 + 交叉耦合同步补偿）
    GantryResult     -- 一次定位过程的完整遥测与统计结果
"""

from .gantry import GantryController, GantryResult

__all__ = ["GantryController", "GantryResult"]
