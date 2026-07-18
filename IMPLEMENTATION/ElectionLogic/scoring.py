"""
ElectionLogic/scoring.py

Computes a deterministic integer score from the device profile.
No hardware probing or benchmarking — the heavy numbers (CPU TFLOPS and RAM
bandwidth) are declared by the user during first-run setup in identity.py.
Only the *currently available* RAM is read live (it is the one thing that
changes run-to-run and caps the model that actually fits right now).

Score model
───────────
For local LLM serving the leader's fitness is dominated by memory bandwidth
(decode throughput ≈ bandwidth / model_bytes), then compute (prefill), then
how much RAM is free (caps the model size), with a multiplier for a hardware
accelerator.  Each input is normalised against a typical mid-range device so
the weights are meaningful, then combined:

    norm = 0.45 * (ram_bandwidth_gbs / REF_BW_GBS)
         + 0.35 * (cpu_tflops        / REF_TFLOPS)
         + 0.20 * (available_ram_gb  / REF_RAM_GB)

    score = round( norm * accel_mult * 1000 )      # integer, higher = better

Ties on score are broken deterministically by agent_id in election.py, so the
exact integer scale here never causes a split-brain.
"""

import logging

from utils import available_ram_gb

log = logging.getLogger(__name__)

# Reference (≈ a typical mid-range mini-PC) used to normalise each term.
REF_TFLOPS = 1.0     # 1 TFLOPS of CPU compute
REF_BW_GBS = 25.0    # 25 GB/s memory bandwidth
REF_RAM_GB = 8.0     # 8 GB RAM

# Accelerator multiplies the whole score (declared by the user, not detected).
_ACCEL_MULT = {
    "gpu":   1.5,
    "hailo": 1.3,
    "npu":   1.2,
    "none":  1.0,
}


def compute_score(profile: dict) -> int:
    ram_gb   = available_ram_gb()
    tflops   = float(profile.get("cpu_tflops", 0.0) or 0.0)
    bw_gbs   = float(profile.get("ram_bandwidth_gbs", 0.0) or 0.0)
    accel    = profile.get("accelerator", "none")

    norm = (
        0.45 * (bw_gbs / REF_BW_GBS)
        + 0.35 * (tflops / REF_TFLOPS)
        + 0.20 * (ram_gb / REF_RAM_GB)
    )
    accel_mult = _ACCEL_MULT.get(accel, 1.0)
    score = int(round(norm * accel_mult * 1000))

    log.info("tflops=%s bw=%sGB/s avail_ram=%.1fGB accel=%s(x%s) -> score=%s",
             tflops, bw_gbs, ram_gb, accel, accel_mult, score)
    return score


def hardware_info(profile: dict) -> dict:
    """Human-readable hardware summary for capability broadcasts."""
    accel = profile.get("accelerator", "none")
    return {
        "ram_gb":            round(available_ram_gb(), 1),
        "cpu_cores":         profile.get("cpu_cores", 0),
        "cpu_tflops":        profile.get("cpu_tflops", 0.0),
        "ram_bandwidth_gbs": profile.get("ram_bandwidth_gbs", 0.0),
        "accelerator":       accel,
        "has_gpu":           accel == "gpu",
        "has_hailo":         accel == "hailo",
        "has_npu":           accel == "npu",
    }
