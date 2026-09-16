# DS-025 — Gate 2.3 Frozen Thresholds (Part A)

> FROZEN before any held-out scoring. Derived mechanically from the
> Gate 2.2 calibration records only (qwen_degenerate + prose control);
> nothing downstream may re-derive these numbers. The greenlight rule:
> T(L) = (p90_degenerate + p10_prose) / 2, band = [0.75*T, 1.25*T].

## Calibration source

- `docs/gate22/gate22_results.jsonl` (300 records: qwen_degenerate 100, t2s_degenerate 100, prose control 100).
- Calibration records used for the freeze: `qwen_degenerate` (100) and `prose` (100) ONLY.
- Percentiles: NumPy `np.percentile` default linear interpolation, matching the Gate 2.2 report convention.
- Candidate layers: **{2, 12, 26}**.

## Raw percentiles and derived thresholds

| layer | signal | p90_deg | p10_prose | T = (p90_deg+p10_prose)/2 | band_low = 0.75*T | band_high = 1.25*T |
|---|---|---|---|---|---|---|
| 2 | PR | 3.127386 | 18.782207 | 10.954796 | 8.216097 | 13.693495 |
| 2 | SVSE | 0.393664 | 0.947907 | 0.670785 | 0.503089 | 0.838482 |
| 12 | PR | 3.672535 | 18.674590 | 11.173562 | 8.380172 | 13.966953 |
| 12 | SVSE | 0.591445 | 0.950447 | 0.770946 | 0.578210 | 0.963683 |
| 26 | PR | 3.832678 | 17.712637 | 10.772658 | 8.079493 | 13.465822 |
| 26 | SVSE | 0.615888 | 0.944221 | 0.780054 | 0.585041 | 0.975068 |

## Machine-parseable freeze block

```json
{
  "layers": {
    "2": {
      "pr": {
        "p90_deg": 3.127386,
        "p10_prose": 18.782207,
        "T": 10.954796,
        "band_low": 8.216097,
        "band_high": 13.693495
      },
      "svse": {
        "p90_deg": 0.393664,
        "p10_prose": 0.947907,
        "T": 0.670785,
        "band_low": 0.503089,
        "band_high": 0.838482
      }
    },
    "12": {
      "pr": {
        "p90_deg": 3.672535,
        "p10_prose": 18.67459,
        "T": 11.173562,
        "band_low": 8.380172,
        "band_high": 13.966953
      },
      "svse": {
        "p90_deg": 0.591445,
        "p10_prose": 0.950447,
        "T": 0.770946,
        "band_low": 0.57821,
        "band_high": 0.963683
      }
    },
    "26": {
      "pr": {
        "p90_deg": 3.832678,
        "p10_prose": 17.712637,
        "T": 10.772658,
        "band_low": 8.079493,
        "band_high": 13.465822
      },
      "svse": {
        "p90_deg": 0.615888,
        "p10_prose": 0.944221,
        "T": 0.780054,
        "band_low": 0.585041,
        "band_high": 0.975068
      }
    }
  }
}
```

## Notes

- Fusion rule (RFC-004): PR primary. If PR is below `band_low` → fire. If PR is inside the ambiguity band → fire only if SVSE agrees (SVSE below T_SVSE) AND not both-inside-bands (SVSE below its own lower band edge). If PR is above `band_high` → no fire.
- Direction is degenerate-LOWER for both PR and SVSE (Gate 2.2 measured means: PR ~1.95-3.97 degenerate vs ~18.58 prose; SVSE ~0.25-0.44 degenerate vs ~0.96 prose).
- This file is the freeze. `scripts/score_gate23.py` reads this file; it does not re-derive.
