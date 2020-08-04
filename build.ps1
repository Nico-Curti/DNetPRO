#!/usr/bin/env pwsh

$number_of_build_workers=8

if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) {
  $vstype = "Professional"
  if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools") {
  }
  else {
    $vstype = "Enterprise"
    if (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools") {
    }
    else {
      $vstype = "Community"
    }
  }
  Write-Host "Found VS 2019 ${vstype}" -ForegroundColor Yellow
  Push-Location "C:\Program Files (x86)\Microsoft Visual Studio\2019\${vstype}\Common7\Tools"
  cmd /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
    if ($_ -match "=") {
      $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
    }
  }
  Pop-Location
  Write-Host "Visual Studio 2019 ${vstype} Command Prompt variables set.`n" -ForegroundColor Yellow
}
else {
  Write-Host "No Compiler found" -ForegroundColor Red
}


# DEBUG
Remove-Item .\build_win_debug -Force -Recurse -ErrorAction SilentlyContinue
New-Item -Path .\build_win_debug -ItemType directory -Force
Set-Location build_win_debug
cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DOMP=OFF" "-DPYWRAP=OFF" ..
cmake --build . --config Debug --parallel ${number_of_build_workers} --target install
Set-Location ..

# RELEASE
Remove-Item .\build_win_release -Force -Recurse -ErrorAction SilentlyContinue
New-Item -Path .\build_win_release -ItemType directory -Force
Set-Location build_win_release
cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DOMP=OFF" ..
cmake --build . --config Release --parallel ${number_of_build_workers} --target install
Set-Location ..

