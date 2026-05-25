param(
    [int]$Episodes = 100,
    [string]$NetworkMode = "controlled"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "runs" | Out-Null

python main.py --mode benchmark `
    --benchmark-policy both `
    --eval-episodes $Episodes `
    --network-mode $NetworkMode `
    --rerank-weight 0.35 `
    --provider-group-weight 0.0 `
    --log-path "runs/selection_benchmark_conservative.jsonl"

python main.py --mode analyze-log `
    --log-path "runs/selection_benchmark_conservative_adaptive.jsonl" `
    --example-limit 3

python main.py --mode benchmark `
    --benchmark-policy adaptive `
    --eval-episodes $Episodes `
    --network-mode $NetworkMode `
    --rerank-weight 1.0 `
    --provider-group-weight 1.0 `
    --log-path "runs/selection_benchmark_aggressive.jsonl"

python main.py --mode analyze-log `
    --log-path "runs/selection_benchmark_aggressive.jsonl" `
    --example-limit 3
