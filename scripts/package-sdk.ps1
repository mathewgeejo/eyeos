param(
    [string]$LibraryPath = (Join-Path $PSScriptRoot '..\target\release\eyeos.dll'),
    [string]$Destination = (Join-Path $PSScriptRoot '..\dist\eye-tracker-sdk'),
    [ValidateSet('debug', 'release')]
    [string]$BuildFlavor = 'release'
)
$ErrorActionPreference = 'Stop'
$workspacePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sdkLibraryPath = [IO.Path]::GetFullPath($LibraryPath)
$sdkPackagePath = [IO.Path]::GetFullPath($Destination)
if (-not (Test-Path -LiteralPath $sdkLibraryPath -PathType Leaf)) {
    throw "Native SDK DLL missing at $sdkLibraryPath. This script packages an existing DLL; it never builds."
}
foreach ($folder in @('native', 'include', 'python', 'node', 'licenses')) {
    New-Item -ItemType Directory -Path (Join-Path $sdkPackagePath $folder) -Force | Out-Null
}
Copy-Item -LiteralPath $sdkLibraryPath -Destination (Join-Path $sdkPackagePath 'native\eyeos.dll')
Copy-Item -LiteralPath (Join-Path $workspacePath 'sdk\include\eye_tracker.h') -Destination (Join-Path $sdkPackagePath 'include')
foreach ($file in @('eye_tracker.py', 'example_overlay.py', 'example_rgb.py', 'pyproject.toml')) {
    Copy-Item -LiteralPath (Join-Path $workspacePath "sdk\python\$file") -Destination (Join-Path $sdkPackagePath 'python')
}
foreach ($file in @('index.mjs', 'index.d.ts', 'example_camera.mjs', 'package.json', 'package-lock.json')) {
    Copy-Item -LiteralPath (Join-Path $workspacePath "sdk\node\$file") -Destination (Join-Path $sdkPackagePath 'node')
}
Copy-Item -LiteralPath (Join-Path $workspacePath 'sdk\README.md') -Destination $sdkPackagePath
Copy-Item -LiteralPath (Join-Path $workspacePath 'LICENSE-MIT') -Destination (Join-Path $sdkPackagePath 'licenses')
Copy-Item -LiteralPath (Join-Path $workspacePath 'assets\runtime\mediapipe\LICENSE') -Destination (Join-Path $sdkPackagePath 'licenses\Apache-2.0.txt')
Copy-Item -LiteralPath (Join-Path $workspacePath 'assets\models\NOTICE.md') -Destination (Join-Path $sdkPackagePath 'licenses\MediaPipe-model-NOTICE.md')
Copy-Item -LiteralPath (Join-Path $workspacePath 'assets\models\openvino\NOTICE.md') -Destination (Join-Path $sdkPackagePath 'licenses\OpenVINO-NOTICE.md')
Copy-Item -LiteralPath (Join-Path $workspacePath 'sdk\REDISTRIBUTION.md') -Destination (Join-Path $sdkPackagePath 'licenses')
$manifest = @{
    abi_version = 1
    platform = 'windows-x86_64'
    build_flavor = $BuildFlavor
    model_id = 'mediapipe-64184e229b26/adas-0002/preprocess-v2'
    native_sha256 = (Get-FileHash -LiteralPath (Join-Path $sdkPackagePath 'native\eyeos.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    accuracy = 'Measured per user; no universal precision guarantee'
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $sdkPackagePath 'manifest.json') -Encoding utf8
Write-Output "SDK packaged at $sdkPackagePath"
