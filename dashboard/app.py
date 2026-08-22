# -*- coding: utf-8 -*-
"""
Web 曲线看板（Flask + ECharts + Canvas）
========================================

功能：
  - 龙门双轴实验：在线改"增益失配度 / 补偿开关"，返回双轴位置曲线与
    同步偏差曲线（含阈值线）；
  - 电子凸轮：凸轮表可视化（256 点），可改定长/同步窗；
  - 飞剪仿真：不同带速下运行，返回速度曲线与指标，前端用 Canvas 播放
    "输送带 + 切刀联动"示意动画；
  - 报告查看：直接输出 docs/测试报告.md 的文本内容。

启动方式：
    python dashboard/app.py
    然后浏览器打开 http://127.0.0.1:5000

免责说明：所有数据均为 Python 虚拟轴仿真（仿真验证值），非真实驱动器。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接脚本运行（python dashboard/app.py）：把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request

from gantry.gantry import GantryController, GantryParams
from cam.electronic_cam import FlyingShear, ShearParams, build_flying_shear_table

app = Flask(__name__)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """主页面：参数面板 + 图表区 + Canvas 动画"""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API：龙门双轴定位实验
# ---------------------------------------------------------------------------
@app.route("/api/gantry", methods=["POST"])
def api_gantry():
    """
    请求体 JSON：
      mismatch_pct : 两轴增益失配度百分比（5~50）
      comp         : 是否开启交叉耦合同步补偿
    返回：降采样后的时间序列 + 统计指标
    """
    try:
        data = request.get_json(force=True)
        mismatch = float(data.get("mismatch_pct", 20)) / 100.0
        comp = bool(data.get("comp", True))
        mismatch = min(max(mismatch, 0.01), 0.60)  # 夹取到合理区间

        prm = GantryParams(mismatch=mismatch, comp_enabled=comp)
        result = GantryController(prm).run_positioning()
        payload = result.to_plot_series()
        payload["params_used"] = {
            "mismatch_pct": round(mismatch * 100, 1),
            "comp": comp,
            "kp1": prm.kp_base,
            "kp2": round(prm.kp_base * (1 - mismatch), 2),
        }
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001 - 看板接口统一兜底
        return jsonify({"error": f"龙门实验失败：{exc}"}), 400


# ---------------------------------------------------------------------------
# API：电子凸轮表可视化
# ---------------------------------------------------------------------------
@app.route("/api/cam_table")
def api_cam_table():
    """
    查询参数：
      length_mm : 定长（主轴每周期位移）
      window_mm : 同步窗口长度
    返回：256 点表数据 + 工艺分段信息（供图表标注待机/同步/返回区）
    """
    try:
        length = float(request.args.get("length_mm", 600))
        window = float(request.args.get("window_mm", 120))
        length = min(max(length, 100.0), 2000.0)
        window = min(max(window, 10.0), length * 0.4)

        table, seg = build_flying_shear_table(length, window)
        return jsonify(
            {
                "name": table.name,
                "xs": [round(v, 3) for v in table.xs],
                "ys": [round(v, 3) for v in table.ys],
                "seg": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in seg.items()},
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"生成凸轮表失败：{exc}"}), 400


# ---------------------------------------------------------------------------
# API：飞剪仿真
# ---------------------------------------------------------------------------
@app.route("/api/shear", methods=["POST"])
def api_shear():
    """
    请求体 JSON：
      belt_speed : 带速 mm/s（300~1600）
      length_mm  : 定长 mm（600~1200，可选）
    返回：速度曲线、同步标记、落刀时刻与指标；前端据此播放 Canvas 动画
    """
    try:
        data = request.get_json(force=True)
        belt = float(data.get("belt_speed", 700))
        length = float(data.get("length_mm", 600))
        belt = min(max(belt, 200.0), 1800.0)
        length = min(max(length, 300.0), 1500.0)

        prm = ShearParams(belt_speed_mm_s=belt, product_length_mm=length,
                          sync_window_mm=min(120.0, length * 0.2), sim_cycles=3)
        result = FlyingShear(prm).run()
        payload = result.to_plot_series()
        payload["seg"] = None
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"飞剪仿真失败：{exc}"}), 400


# ---------------------------------------------------------------------------
# API：测试报告文本
# ---------------------------------------------------------------------------
@app.route("/api/report")
def api_report():
    """返回 docs/测试报告.md 的 Markdown 文本（缺失时给出提示）"""
    report_path = ROOT / "docs" / "测试报告.md"
    if report_path.exists():
        return jsonify({"report": report_path.read_text(encoding="utf-8")})
    return jsonify(
        {
            "report": "尚未找到 docs/测试报告.md。\n\n请先运行：\n"
                      "    python -m testbench.batch_test\n"
                      "生成报告后再刷新本页。"
        }
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("龙门同步 & 电子凸轮·飞剪 仿真看板")
    print("请用浏览器打开 http://127.0.0.1:5000")
    print("=" * 60)
    # debug=False：避免调试器重复加载仿真模块；局域网内不开放
    app.run(host="127.0.0.1", port=5000, debug=False)
