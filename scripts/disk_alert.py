"""Disk usage alert for dataset pipeline.

Checks disk usage and prints a warning if usage >= 80%.

Used as a cron job to alert cc when cleanup is needed.
"""
import shutil


def check_disk_alert(threshold: int = 80) -> bool:
    """Check if disk usage is above threshold.

    Returns True if alert condition met (disk >= threshold%).
    """
    disk = shutil.disk_usage("/")
    usage_pct = disk.used * 100 // disk.total

    if usage_pct >= threshold:
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  DISK SPACE WARNING ⚠️                                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Disk usage: {usage_pct}% ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)                       ║
║   Free space: {disk.free // (1024**3)}GB                                         ║
║                                                                  ║
║   Action required:                                               ║
║   1. Check /home/cc/.objaverse/ for old GLB caches               ║
║   2. Check /home/cc/.cache/huggingface/ for HF model caches      ║
║   3. Run cleanup if pipeline is stuck or completed               ║
║                                                                  ║
║   Quick fix:                                                     ║
║   python3 -c "from pathlib import Path; ...                      ║
║     # Delete old objaverse GLBs (>30min)                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
        return True

    print(f"Disk OK: {usage_pct}% used ({disk.free // (1024**3)}GB free)")
    return False


if __name__ == "__main__":
    check_disk_alert(threshold=80)
