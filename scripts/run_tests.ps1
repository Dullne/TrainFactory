# Use the project's environment even when a global Conda Python is on PATH.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $projectPython -PathType Leaf)) {
    throw 'Project Python is missing. Follow docs/local-tests.md to set up .venv.'
}

Push-Location -LiteralPath $projectRoot
try {
    & $projectPython -I -m pytest @args
    $testExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $testExitCode
