"""Create the double-click shortcuts (Windows): `Images.lnk`, `Image Screensaver.lnk`,
`Screensaver Settings.lnk` and `Image Movie.lnk`.

They start the app and the screensaver with this folder's .venv\\Scripts\\pythonw.exe (no console window).
Run after `uv sync`:  uv run make_shortcuts.py   (add --desktop to put them on the desktop instead)
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# shortcut → (script, icon): each program has its own colour of the icon (make_icons.py)
SHORTCUTS = {"Images": ("app.pyw", "icon.ico"),
             "Image Screensaver": ("screensaver.pyw", "icon_screensaver.ico"),
             "Screensaver Settings": ("screensaver_settings.py", "icon_settings.ico"),
             "Image Movie": ("moviemaker.py", "icon_movie.ico")}


def main() -> None:
    if os.name != "nt":
        sys.exit("Shortcuts are for Windows; elsewhere run `uv run app.pyw` or `uv run screensaver.pyw`.")
    pythonw = HERE / ".venv" / "Scripts" / "pythonw.exe"
    if not pythonw.exists():
        sys.exit(f"{pythonw} not found: run `uv sync` first.")
    where = Path(os.environ["USERPROFILE"]) / "Desktop" if "--desktop" in sys.argv else HERE
    for name, (script, icon) in SHORTCUTS.items():
        lnk = where / f"{name}.lnk"
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); "
              "$s.TargetPath = $env:TARGET; $s.Arguments = '\"' + $env:SCRIPT + '\"'; "
              "$s.WorkingDirectory = $env:HERE; $s.IconLocation = $env:ICON + ',0'; $s.Save()")
        env = {**os.environ, "LNK": str(lnk), "TARGET": str(pythonw), "SCRIPT": str(HERE / script),
               "HERE": str(HERE), "ICON": str(HERE / icon)}
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], env=env, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        print(f"created {lnk}")


if __name__ == "__main__":
    main()
