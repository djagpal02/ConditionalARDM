:: script.bat
@echo off
set filename=%1

echo Running black
black %filename%

echo Running mypy
mypy %filename%

echo Running radon
radon cc %filename%

echo Running pylint
pylint %filename%
