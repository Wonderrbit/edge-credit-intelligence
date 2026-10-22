param(
    [string]$TaskName = "AkropoliAutopilot",
    [string]$Time = "09:00"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "scripts\run_autopilot.py"

if (-not (Test-Path $python)) {
    throw "Python da venv não encontrado: $python"
}

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`" --mode full" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Executa o autopilot de crédito e cobrança." -Force | Out-Null

Write-Host "Tarefa '$TaskName' registrada para todos os dias às $Time."
