@echo off
python -m pip install --upgrade pip
echo Upgrading pip...

echo Installing/upgrading packages from requirements.txt...
pip install --upgrade -r requirements.txt

echo Process complete. Press any key to exit.
pause
