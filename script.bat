:: script.bat
@echo off
set filename=%1

echo Running black
black %filename%

echo Running mypy
mypy %filename%

echo Running radon
radon cc %filename%

:: Disabling non-member and long line errors and unable to import
echo Running pylint
pylint %filename%  --disable=E1101,C0301,E0401
