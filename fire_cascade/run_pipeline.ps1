param(
    [Parameter(Mandatory = $true)]
    [string] $Run,
    [ValidateSet('check', 'train', 'probe', 'infer', 'score')]
    [string] $Start = 'check'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$pipelinePython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
$pipelineEntry = Join-Path $PSScriptRoot 'run.py'
$pipelineRun = (Resolve-Path -LiteralPath $Run).Path
$stages = @('check', 'train', 'probe', 'infer', 'score')
$runStage = $false
foreach ($stage in $stages) {
    $runStage = $runStage -or ($stage -eq $Start)
    if (!$runStage) { continue }
    Write-Output "$(Get-Date -Format o) Starting $stage"
    $stageLog = Join-Path $pipelineRun "$stage.log"
    & $pipelinePython -u $pipelineEntry $stage --run $pipelineRun *> $stageLog
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $stage failed with exit code $LASTEXITCODE. See $stageLog"
    }
    Write-Output "$(Get-Date -Format o) Finished $stage"
}
