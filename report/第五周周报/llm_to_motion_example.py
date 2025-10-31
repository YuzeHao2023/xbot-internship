#!/usr/bin/env python3
# 简化演示：把 LLM 输出的 "Pythonic plan" 解析为动作调用，并模拟执行。
# 使用方法示例：
#   python llm_to_motion_example.py --plan_file sample_plan.txt
#
# 该脚本不依赖真实机器人，只演示解析、动作分派、以及调用 DMP 回放 stub。

import argparse
import json
import time
from typing import List, Dict, Any

# --- 动作函数库（模拟实现） ---
def move_to_position(target_name: str):
    print(f"[ACTION] move_to_position -> {target_name}")
    time.sleep(0.2)
    return True

def gripper_control(cmd: str):
    print(f"[ACTION] gripper_control -> {cmd}")
    time.sleep(0.1)
    return True

def base_cycle_move(param: str):
    print(f"[ACTION] base_cycle_move -> {param}")
    time.sleep(0.2)
    return True

def close_move(obj: str):
    print(f"[ACTION] close_move -> {obj}")
    time.sleep(0.2)
    return True

def rotate_waist(degree: float):
    print(f"[ACTION] rotate_waist -> {degree} deg")
    time.sleep(0.15)
    return True

# DMP 回放（stub）
def dmp_publish(name: str):
    print(f"[DMP] play stored trajectory -> {name}")
    time.sleep(0.3)
    return True

# --- 简单解析器：把 LLM 输出的 "Pythonic plan" 转换为动作序列 ---
# 期望的 plan 格式示例（LLM 输出）：
# move_to_position(apple_1)
# gripper_control(close)
# dmp_publish(open_oven_handle_ex)
# base_cycle_move(radius_door2axis)
# gripper_control(open)

ACTION_MAP = {
    "move_to_position": move_to_position,
    "gripper_control": gripper_control,
    "base_cycle_move": base_cycle_move,
    "close_move": close_move,
    "rotate_waist": rotate_waist,
    "dmp_publish": dmp_publish
}

def parse_plan_lines(lines: List[str]) -> List[Dict[str, Any]]:
    actions = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # 支持带序号的行 "1. move_to_position(apple_1)"
        if "." in ln[:3]:
            ln = ln.split(".", 1)[1].strip()
        # 简单匹配 func(args)
        if "(" in ln and ln.endswith(")"):
            func = ln[:ln.find("(")].strip()
            argstr = ln[ln.find("(")+1:-1].strip()
            # 支持多个参数用逗号分隔
            args = [a.strip() for a in argstr.split(",")] if argstr else []
            actions.append({"func": func, "args": args, "raw": ln})
        else:
            # 未识别格式 -> 当作注释/跳过
            actions.append({"func": None, "args": [], "raw": ln})
    return actions

def execute_actions(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for a in actions:
        func_name = a["func"]
        if func_name is None:
            results.append({"raw": a["raw"], "ok": False, "reason": "parse_error"})
            continue
        func = ACTION_MAP.get(func_name)
        if func is None:
            results.append({"raw": a["raw"], "ok": False, "reason": "unknown_action"})
            continue
        try:
            # 简单展开参数（全部当作字符串或数字）
            parsed_args = []
            for arg in a["args"]:
                # 尝试转换为数字
                try:
                    if "." in arg:
                        parsed_args.append(float(arg))
                    else:
                        parsed_args.append(int(arg))
                except Exception:
                    parsed_args.append(arg.strip().strip('"').strip("'"))
            ok = func(*parsed_args)
            results.append({"raw": a["raw"], "ok": bool(ok)})
        except Exception as e:
            results.append({"raw": a["raw"], "ok": False, "reason": str(e)})
    return {"summary": {"total": len(actions), "succeeded": sum(1 for r in results if r.get("ok"))}, "details": results}

# --- 主程序：读取 plan 文件并执行 ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_file", required=True, help="LLM 输出的 plan 文本文件（每行一个动作）")
    parser.add_argument("--out_json", default="plan_exec_result.json", help="输出执行结果")
    args = parser.parse_args()

    with open(args.plan_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    actions = parse_plan_lines(lines)
    result = execute_actions(actions)

    with open(args.out_json, "w", encoding="utf-8") as fo:
        json.dump(result, fo, ensure_ascii=False, indent=2)

    print("Execution result saved to", args.out_json)
    print("Summary:", result["summary"])

if __name__ == "__main__":
    main()
