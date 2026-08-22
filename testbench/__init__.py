# -*- coding: utf-8 -*-
"""
testbench 包：参数扫描批量测试与测试报告生成。

对外暴露：
    main                -- 运行全部实验并生成 docs/测试报告.md
    run_gantry_matrix   -- 龙门矩阵实验：补偿开/关 × 增益失配 5%~30%
    run_shear_sweep     -- 飞剪带速扫描实验
    run_alarm_demo      -- 同步偏差报警联锁演示

说明：这里使用惰性导入（模块级 __getattr__），既支持 `from testbench import ...`
的正常用法，也保证 `python -m testbench.batch_test` 直接运行时不会产生重复导入警告。
"""


def __getattr__(name: str):
    if name == "main":
        from .batch_test import main

        return main
    if name == "run_gantry_matrix":
        from .batch_test import run_gantry_matrix

        return run_gantry_matrix
    if name == "run_shear_sweep":
        from .batch_test import run_shear_sweep

        return run_shear_sweep
    if name == "run_alarm_demo":
        from .batch_test import run_alarm_demo

        return run_alarm_demo
    raise AttributeError(f"testbench 模块没有属性 {name!r}")


__all__ = ["main", "run_gantry_matrix", "run_shear_sweep", "run_alarm_demo"]
