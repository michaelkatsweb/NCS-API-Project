# Workflow Cleanup Script
Write-Host "🧹 Cleaning up redundant workflow files..." -ForegroundColor Yellow

$workflowsToRemove = @(
    ".github/workflows/ci-cd.yml",
    ".github/workflows/pipeline-test.yml", 
    ".github/workflows/ci-pipeline.yml"
)

foreach ($workflow in $workflowsToRemove) {
    if (Test-Path $workflow) {
        Remove-Item $workflow -Force
        Write-Host "🗑️  Removed: $workflow" -ForegroundColor Red
    }
}

Write-Host "✅ Workflow cleanup complete!" -ForegroundColor Green
Write-Host "The main-ci-cd.yml workflow will handle everything now." -ForegroundColor Cyan