import os

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


IDS = {
    "Platynereis-H2B-TL": "1jGwaJ62w80GYo5I_Jcb3O_g7y4RKEhjI",
    "Zebrafish-XSPIM-multiview": "https://drive.google.com/drive/folders/175hZRrUNWM2UzY0wzXPFjuFZ5QKUN-tm?usp=drive_link"  # noqa
}

# ROOT = "/mnt/lustre-grete/usr/u12086/data/flamingo"
ROOT = "/mnt/lustre-emmy-hdd/usr/u12086/data/flamingo"


def download_folder(drive, name):
    os.makedirs(ROOT, exist_ok=True)

    destination_folder = os.path.join(ROOT, name)
    folder_id = IDS[name]

    folder_query = f"'{folder_id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': folder_query}).GetList()

    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            folder_name = os.path.join(destination_folder, file['title'])
            os.makedirs(folder_name, exist_ok=True)
            download_folder(file['id'], folder_name)
        else:
            print(f"Downloading {file['title']} to {destination_folder}")
            # breakpoint()
            file.GetContentFile(os.path.join(destination_folder, file['title']))


def get_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")  # Use the saved credentials
    if gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    drive = GoogleDrive(gauth)
    return drive


def main():
    drive = get_drive()

    # download_from_gdrive(name="Zebrafish-XSPIM-multiview")
    download_folder(drive, name="Platynereis-H2B-TL")


if __name__ == "__main__":
    main()
