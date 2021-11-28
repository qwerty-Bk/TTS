from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path
from git import Repo


def do():
    pt_file = './waveglow_256channels_universal_v5.pt'
    if not Path(pt_file).exists():
        gdd.download_file_from_google_drive(
            file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
            dest_path=pt_file
        )

    git_url = "https://github.com/NVIDIA/waveglow.git"
    repo_dir = "waveglow"
    if not Path("waveglow").exists():
        Repo.clone_from(git_url, repo_dir)
