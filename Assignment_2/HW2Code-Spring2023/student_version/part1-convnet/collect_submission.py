import os
import subprocess
import platform

os_name = platform.system().lower()
zipfile = 'assignment_2_part_1_submission.zip'
if "windows" in os_name:
    if os.path.exists('./' + zipfile):
        command = ['del', zipfile]
        subprocess.run(command)

    command = ['tar', '-a', '-c', '-f', zipfile,
        'modules', 'optimizer', 'trainer.py', 'train.py']
    subprocess.run(command)
else:
    command = ['rm', '-f', zipfile]
    subprocess.run(command)

    command = ['zip', '-r', zipfile,
        'modules/', 'optimizer/', 'trainer.py', 'train.py']
    subprocess.run(command)
