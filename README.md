---
license: apache-2.0
---
wandb: https://wandb.ai/open-assistant/supervised-finetuning/runs/770a0t41 (exported at 4000 steps)

data:

```
reference-data:
  datasets:
    - oasst_export:
      lang: "bg,ca,cs,da,de,en,es,fr,hr,hu,it,nl,pl,pt,ro,ru,sl,sr,sv,uk"
      input_file_path: 2023-03-25_oasst_research_ready_synth_labels.jsonl.gz
      val_split: 0.05
    - alpaca
  sort_by_length: false
  use_custom_sampler: false
```


pythia:
```
reference-pythia-12b:
  dtype: fp16
  log_dir: "pythia_log_12b"
  learning_rate: 6e-6
  model_name: EleutherAI/pythia-12b-deduped
  output_dir: pythia_model_12b
  weight_decay: 0.0
  max_length: 2048
  warmup_steps: 100
  gradient_checkpointing: true
  gradient_accumulation_steps: 2
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  eval_steps: 100
  save_steps: 1000
  num_train_epochs: 8
  save_total_limit: 4
```

zero config:
```
{
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 1e9,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 1e9,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```