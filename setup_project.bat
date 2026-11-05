@echo off
echo Setting up Driver Drowsiness Detection Environment...

echo Creating conda environment with Python 3.10...
call conda create -n drowsiness_detection python=3.10 -y

echo Activating environment...
call conda activate drowsiness_detection

echo Installing dlib (this may take a few minutes)...
call conda install -c conda-forge dlib -y

echo Installing other dependencies...
call pip install opencv-python pygame playsound scikit-learn

echo Setup complete!
echo To run the project, use the run_detection.bat file
pause