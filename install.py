import os
import sys
import subprocess


def _run(cmd, cwd=None):
    subprocess.check_call(cmd, cwd=cwd)


def install():
    here = os.path.dirname(os.path.abspath(__file__))
    req = os.path.join(here, "requirements.txt")
    if os.path.isfile(req):
        _run([sys.executable, "-m", "pip", "install", "-r", req])


if __name__ == "__main__":
    install()
