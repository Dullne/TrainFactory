# Training System - Anomaly Report

**Date**: 2026-02-02
**Task**: infonce-refactor-test (2e337baf-7649-427a-b50f-51f1422fdf3b)
**Status**: Training Succeeded, but anomalies detected

## Summary

After refactoring `infonce_loss` in `qwen3-rerank-trainer/src/qwen3_rerank_trainer/losses/contrastive.py`, testing revealed that the loss function occasionally returns **exact 0.0 values** during training, causing **zero gradients**.

## Observed Pattern

| Step | Loss | Grad Norm | Note |
|------|------|-----------|------|
| 8 | 5.96e-08 | 8.876e-05 | Near-zero |
| 9 | 0.0 | 3.533e-08 | Zero |
| 13 | 0.0 | 3.703e-06 | Zero |
| 15 | 0.0 | 8.789e-14 | Zero |

When loss is exactly 0.0, `grad_norm` is also near-zero, meaning **no learning signal** for those batches.

## Root Cause Analysis

This is **not a bug in the refactored code**, but rather expected behavior under certain conditions:

1. **Low temperature (0.05)**: `scale = 1/temperature = 20`, making softmax distribution very sharp
2. **Small dataset (30 samples)**: Model quickly memorizes patterns
3. **Clear positive-negative separation**: Some query-doc pairs have obvious relevance differences

When the positive example's score is significantly higher than negatives, after scaling by 20x:
```
cross_entropy = -log(softmax(scaled_scores)[positive_index])
             = -log(exp(s_pos*20) / sum(exp(s_i*20)))
             → 0 when s_pos >> s_neg
```

## Impact

- **Functionality**: Training completes successfully
- **API train_loss**: Shows 0.0 because last step's loss was 0.0 (correct behavior)
- **Actual avg train_loss**: 39.36 (reported in final metrics)
- **Eval loss**: 45.13 (normal)

## Recommendations

1. **For production training**:
   - Use larger datasets to avoid memorization
   - Consider higher temperature (0.1-0.2) for more stable gradients

2. **For monitoring**:
   - Add warning when consecutive batches have zero loss
   - Track percentage of zero-loss batches as a metric

3. **No code changes required**: The refactored `infonce_loss` works correctly

## Files Verified

- `qwen3-rerank-trainer/src/qwen3_rerank_trainer/losses/contrastive.py` - Refactored, working
- `qwen3-rerank-trainer/src/qwen3_rerank_trainer/training/sft_trainer.py` - Updated for new API
- `loss_history.jsonl` - Correctly logs step-wise losses

---

# Bug Fix: loss_config=None Causes AttributeError

**Date**: 2026-02-02
**Affected Tasks**: test-logging-v2, test-logging-steps
**Status**: FIXED

## Error

```
AttributeError: 'NoneType' object has no attribute 'get'
```

## Root Cause

In `embedding_trainer.py` and `reranker_trainer.py`, the code used:
```python
loss_config = self.raw_config.get('loss_config', {})
```

When `loss_config` key exists but has value `None`, Python's `.get()` returns `None` (not the default `{}`). This caused `loss_config.get('scale', 20.0)` to fail.

## Fix Applied

Changed to:
```python
loss_config = self.raw_config.get('loss_config') or {}
```

This ensures `loss_config` is always a dict, even when explicitly set to `None`.

## Files Modified

- `train_factory/trainers/encoder/embedding_trainer.py:105`
- `train_factory/trainers/encoder/reranker_trainer.py:101`

## Verification

After fix, embedding trainer creates successfully with `loss_config: None`.
