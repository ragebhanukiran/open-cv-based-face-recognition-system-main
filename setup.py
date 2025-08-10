from cx_Freeze import setup, Executable
import sys
import os

# Set TCL/TK library paths for Tkinter on Windows
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

# Determine base for executable (use None for console application)
base = None
if sys.platform == "win32":
    base = None  # Set to "Win32GUI" for GUI apps without console

# Define the executable
executables = [Executable("train.py", base=base, target_name="FaceRecognitionAttendance.exe")]

# List of packages to include
packages = [
    "tkinter",
    "cv2",
    "numpy",
    "PIL",
    "pandas",
    "datetime",
    "time",
    "logging",
    "threading"
]

# Files to include (e.g., Haar cascade file)
include_files = [
    ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_default.xml"),  # Include Haar cascade file
    # Add other necessary files or directories if needed (e.g., "TrainingImage", "StudentDetails")
]

# Build options
options = {
    'build_exe': {
        'packages': packages,
        'include_files': include_files,
        'include_msvcr': True,  # Include Microsoft Visual C++ runtime for Windows
        'optimize': 2,  # Optimize the bytecode
        'excludes': ['idna', 'setuptools'],  # Exclude unnecessary packages
        'silent': True,  # Suppress unnecessary console output
    }
}

# Setup configuration
setup(
    name="FaceRecognitionAttendance",
    version="0.0.1",
    description="Face Recognition Based Attendance System",
    author="R.BhanuKiran", 
    options=options,
    executables=executables
)