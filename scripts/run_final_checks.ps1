param(
    [int]$Episodes = 100,
    [int]$EvalEpisodes = 200,
    [string]$NetworkMode = "controlled",
    [string]$Profile = "laptop",
    [switch]$IncludeLLMEvaluate
)

$ErrorActionPreference = "Stop"

$ReportDir = "runs/final_report"
New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "Final Project Checks"
Write-Host "============================================================"

Write-Host ""
Write-Host "[1/5] Healthcheck"
python main.py --mode healthcheck

Write-Host ""
Write-Host "[2/5] Unit tests"
python -m unittest `
    tests.test_query_normalization `
    tests.test_training_plot `
    tests.test_sonar_selector `
    tests.test_adaptive_selector `
    tests.test_log_analysis `
    tests.test_tool_features `
    tests.test_healthcheck

Write-Host ""
Write-Host "[3/5] Benchmark: semantic, SONAR-style, adaptive"
python main.py --mode benchmark `
    --benchmark-policy all `
    --eval-episodes $Episodes `
    --network-mode $NetworkMode `
    --rerank-weight 0.35 `
    --provider-group-weight 0.0 `
    --log-path "$ReportDir/selection_benchmark.jsonl"

Write-Host ""
Write-Host "[4/5] Analyze adaptive benchmark log"
python main.py --mode analyze-log `
    --log-path "$ReportDir/selection_benchmark_adaptive.jsonl" `
    --example-limit 3

if ($IncludeLLMEvaluate) {
    Write-Host ""
    Write-Host "[5/5] Evaluate trained GRPO policy"
    python main.py --mode evaluate `
        --checkpoint checkpoints/best `
        --eval-episodes $EvalEpisodes `
        --network-mode $NetworkMode `
        --profile $Profile
} else {
    Write-Host ""
    Write-Host "[5/5] Skipped LLM evaluate"
    Write-Host "Run with -IncludeLLMEvaluate to evaluate checkpoints/best."
}

Write-Host ""
Write-Host "Final checks complete. Logs are in $ReportDir"
