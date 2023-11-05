import os
import subprocess
import platform

os_name = platform.system().lower()
zipfile = "assignment_3_submission.zip"
if "windows" in os_name:
    if os.path.exists("./" + zipfile):
        command = ["del", zipfile]
        subprocess.run(command)

    command = [
        "tar",
        "-a",
        "-c",
        "-f",
        zipfile,
        "visualizers",
        "style_modules",
        "gradcam.py",
        "saliency_map.py",
        "style_utils.py",
    ]
    subprocess.run(command)
else:
    command = ["rm", "-f", zipfile]
    subprocess.run(command)

    command = [
        "zip",
        "-r",
        zipfile,
        "visualizers",
        "style_modules",
        "gradcam.py",
        "saliency_map.py",
        "style_utils.py",
    ]
    subprocess.run(command)
