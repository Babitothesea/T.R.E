# Trova la versione di onnxruntime piu' recente che si carica su questa macchina.
# 1.30 crasha con access violation: troppo recente per la VC++ runtime 14.24.
$ErrorActionPreference = "Continue"
foreach ($v in @("1.22.1", "1.21.1", "1.20.1", "1.19.2", "1.18.1", "1.17.3")) {
    Write-Host "--- provando onnxruntime==$v"
    python -m pip install --quiet --disable-pip-version-check "onnxruntime==$v" 2>&1 | Out-Null
    $out = python -c "import onnxruntime as o; print(o.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    COMPATIBILE: onnxruntime $out"
        break
    }
    Write-Host "    non compatibile (exit=$LASTEXITCODE)"
}
