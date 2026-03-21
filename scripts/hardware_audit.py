"""Phase 0 hardware audit — run once, save results."""
import json
import os
import shutil
import torch
from pathlib import Path


def main():
    audit = {}

    # GPU
    audit["gpu_count"] = torch.cuda.device_count()
    audit["gpus"] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        audit["gpus"].append({
            "index": i,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
        })

    # RAM
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    audit["ram_gb"] = round(mem, 1)

    # Disk (root)
    total, used, free = shutil.disk_usage("/")
    audit["disk_root_total_gb"] = round(total / (1024**3), 1)
    audit["disk_root_free_gb"] = round(free / (1024**3), 1)

    # Disk (/data)
    total_d, used_d, free_d = shutil.disk_usage("/data")
    audit["disk_data_total_gb"] = round(total_d / (1024**3), 1)
    audit["disk_data_free_gb"] = round(free_d / (1024**3), 1)

    # PyTorch
    audit["pytorch_version"] = torch.__version__
    audit["cuda_version"] = torch.version.cuda

    # Go/No-Go
    audit["go"] = (
        audit["gpu_count"] >= 1
        and audit["disk_data_free_gb"] >= 100
        and audit["ram_gb"] >= 64
    )

    Path("results/fullscale_eval").mkdir(parents=True, exist_ok=True)
    with open("results/fullscale_eval/hardware_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))
    if not audit["go"]:
        print("\nSTOP: Hardware below minimum thresholds.")
    else:
        print(f"\nHardware GO — {audit['gpu_count']} GPUs, {audit['ram_gb']}GB RAM, "
              f"{audit['disk_data_free_gb']}GB free on /data")


if __name__ == "__main__":
    main()
