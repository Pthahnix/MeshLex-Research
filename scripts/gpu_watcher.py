"""GPU Watcher — 每分钟检查 GPU 1/2 空闲状态，空闲时立即执行排队任务。

Usage:
    # 注册任务后启动监控
    python scripts/gpu_watcher.py --command "CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH=. python scripts/train_rvq.py ..." --interval 60

    # 指定只等某张卡
    python scripts/gpu_watcher.py --command "..." --target_gpu 2

    # 干跑模式（只监控，不执行）
    python scripts/gpu_watcher.py --dry_run --interval 60

命令中的 {gpu} 会被替换为实际空闲的 GPU ID (1 或 2)。
"""
import argparse
import subprocess
import time
import sys
from datetime import datetime


def get_gpu_processes():
    """解析 nvidia-smi 获取每张卡的进程列表。返回 {gpu_id: [pid, ...]}"""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    # 先获取 GPU index → UUID 映射
    uuid_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    uuid_to_idx = {}
    for line in uuid_result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid_to_idx[parts[1]] = int(parts[0])

    gpu_procs = {0: [], 1: [], 2: []}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            uuid, pid = parts[0], parts[1]
            idx = uuid_to_idx.get(uuid)
            if idx is not None and idx in gpu_procs:
                gpu_procs[idx].append(int(pid))

    return gpu_procs


def get_gpu_memory():
    """获取每张卡的显存使用情况。返回 {gpu_id: (used_mb, total_mb)}"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpu_mem = {}
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            idx, used, total = int(parts[0]), float(parts[1]), float(parts[2])
            gpu_mem[idx] = (used, total)
    return gpu_mem


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU 1/2 and launch task when free")
    parser.add_argument("--command", type=str, default=None,
                        help="Command to execute when GPU is free. Use {gpu} as placeholder for GPU ID.")
    parser.add_argument("--target_gpu", type=int, default=None, choices=[1, 2],
                        help="Only watch a specific GPU (default: watch both 1 and 2)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only monitor and report, don't execute command")
    parser.add_argument("--free_threshold_mb", type=float, default=30000,
                        help="Minimum free VRAM (MB) to consider GPU as 'free' (default: 30000, ~30GB)")
    args = parser.parse_args()

    watch_gpus = [args.target_gpu] if args.target_gpu else [1, 2]

    log(f"GPU Watcher started")
    log(f"  Watching: GPU {watch_gpus}")
    log(f"  Interval: {args.interval}s")
    log(f"  Free threshold: {args.free_threshold_mb:.0f} MB")
    log(f"  Command: {args.command or '(dry run)'}")
    log(f"  Mode: {'DRY RUN' if args.dry_run else 'ARMED'}")
    log("=" * 60)

    check_count = 0
    while True:
        check_count += 1
        gpu_procs = get_gpu_processes()
        gpu_mem = get_gpu_memory()

        # 状态报告
        status_parts = []
        for gid in [0, 1, 2]:
            used, total = gpu_mem.get(gid, (0, 0))
            n_procs = len(gpu_procs.get(gid, []))
            status_parts.append(f"GPU{gid}: {used:.0f}/{total:.0f}MB ({n_procs}p)")
        log(f"Check #{check_count} | {' | '.join(status_parts)}")

        # 检查目标 GPU 是否空闲
        for gid in watch_gpus:
            procs = gpu_procs.get(gid, [])
            used, total = gpu_mem.get(gid, (0, 0))
            free = total - used

            if len(procs) == 0 and free >= args.free_threshold_mb:
                log(f">>> GPU {gid} is FREE! No processes, {free:.0f} MB available <<<")

                if args.dry_run or not args.command:
                    log(f"DRY RUN: Would execute command on GPU {gid}")
                    log("Exiting watcher (GPU found).")
                    sys.exit(0)

                # 替换 {gpu} 占位符并执行
                cmd = args.command.replace("{gpu}", str(gid))
                log(f"Launching: {cmd}")
                log("=" * 60)

                # 用 exec 替换当前进程，让任务直接占据此进程
                # 这样 tmux/nohup 保护的就是实际任务
                import os
                os.execvp("bash", ["bash", "-c", cmd])
                # execvp 不会返回，下面的代码不会执行

            elif len(procs) == 0 and free < args.free_threshold_mb:
                log(f"  GPU {gid}: no processes but only {free:.0f} MB free (zombie VRAM?)")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
