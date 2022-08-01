#!/usr/bin/env pwsh

# $args[0] = Release/Debug
# $args[1] = other cmake defines

[CmdletBinding()]
Param
(
  [parameter(mandatory=$true, position=0)][string]$build_type,
  [parameter(mandatory=$false, position=1, ValueFromRemainingArguments=$true)]$other_cmake_flags
)

$number_of_build_workers=(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

function getProgramFiles32bit() {
  $out = ${env:PROGRAMFILES(X86)}
  if ($null -eq $out) {
    $out = ${env:PROGRAMFILES}
  }

  if ($null -eq $out) {
    Write-Host "Could not find [Program Files 32-bit]" -ForegroundColor Yellow
  }

  return $out
}

function getLatestVisualStudioWithDesktopWorkloadPath() {
  $programFiles = getProgramFiles32bit
  $vswhereExe = "$programFiles\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-Not (Test-Path $vswhereExe)) {
    $vswhereExe = "C:\ProgramData\chocolatey\bin\vswhere.exe"
  }
  if (Test-Path $vswhereExe) {
    $output = & $vswhereExe -products * -latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -format xml
    [xml]$asXml = $output
    foreach ($instance in $asXml.instances.instance) {
      $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
    }
    if (-Not ($installationPath)) {
      Write-Host "Warning: no full Visual Studio setup has been found, extending search to include also partial installations" -ForegroundColor Yellow
      $output = & $vswhereExe -products * -latest -format xml
      [xml]$asXml = $output
      foreach ($instance in $asXml.instances.instance) {
        $installationPath = $instance.InstallationPath -replace "\\$" # Remove potential trailing backslash
      }
    }
    if (-Not ($installationPath)) {
      Write-Host "Critical: could not locate any installation of Visual Studio" -ForegroundColor Red
    }
  }
  else {
    Write-Host "Could not locate vswhere at $vswhereExe" -ForegroundColor Yellow
  }
  return $installationPath
}

if ($null -eq (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
  $vsfound = getLatestVisualStudioWithDesktopWorkloadPath
  if ($vsfound) {
    Write-Host "Found VS in ${vsfound}"
    Push-Location "${vsfound}\Common7\Tools"
    cmd.exe /c "VsDevCmd.bat -arch=x64 & set" |
    ForEach-Object {
      if ($_ -match "=") {
        $v = $_.split("="); Set-Item -force -path "ENV:\$($v[0])" -value "$($v[1])"
      }
    }
    Pop-Location
    Write-Host "Visual Studio Command Prompt variables set" -ForegroundColor Yellow
  }
}

Push-Location $PSScriptRoot

If ( $build_type -eq "Debug" -or $build_type -eq "debug" )
{
  # DEBUG
  #Remove-Item .\build_win_debug -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_debug -ItemType directory -Force
  Set-Location build_win_debug
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_BUILD_TYPE=Debug" ${other_cmake_flags} ..
  cmake --build . --config Debug --target install
  Set-Location ..
}
ElseIf ( $build_type -eq "Release" -or $build_type -eq "release" )
{
  # RELEASE
  #Remove-Item .\build_win_release -Force -Recurse -ErrorAction SilentlyContinue
  New-Item -Path .\build_win_release -ItemType directory -Force
  Set-Location build_win_release
  cmake -G "Visual Studio 16 2019" -T "host=x64" -A "x64" "-DCMAKE_BUILD_TYPE=Release" ${other_cmake_flags} ..
  cmake --build . --config Release --target install
  Set-Location ..
}
Else
{
  Write-Host "Unknown build type - Allowed only [Debug, Release]" -ForeGroundColor Red
  exit 1
}

Pop-Location
