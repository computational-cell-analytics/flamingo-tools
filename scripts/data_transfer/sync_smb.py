import argparse
import json
import os

from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import Open, CreateDisposition, FileAttributes
# from smbprotocol.open import Open, CreateDisposition, FileAttributes, FileInformationClass
# from smbprotocol.file_info import FileStandardInformation
# from smbprotocol.exceptions import SMBResponseException


def sync_smb_to_local(smb_server, smb_share, smb_username, smb_password, smb_path, local_dir):
    # Establish an SMB connection
    connection = Connection(guid=os.urandom(16), server_name=smb_server, port=445)
    connection.connect()

    session = Session(connection, smb_username, smb_password)
    session.connect()

    tree = TreeConnect(session, f"\\\\{smb_server}\\{smb_share}")
    tree.connect()

    def download_directory(smb_base_path, local_base_path):
        with Open(tree, smb_base_path, create_disposition=CreateDisposition.FILE_OPEN) as smb_dir:
            smb_files = smb_dir.query_directory("*")

            # Ensure the local directory exists
            if not os.path.exists(local_base_path):
                os.makedirs(local_base_path)

            for smb_file in smb_files:
                file_name = smb_file['file_name'].get_value()
                smb_file_path = os.path.join(smb_base_path, file_name).replace("\\", "/")
                local_file_path = os.path.join(local_base_path, file_name)

                # Check if it's a directory or a file
                if smb_file['file_attributes'].has_flag(FileAttributes.FILE_ATTRIBUTE_DIRECTORY):
                    if file_name not in [".", ".."]:  # Skip '.' and '..' directories
                        download_directory(smb_file_path, local_file_path)
                else:
                    # Download the file
                    with Open(tree, smb_file_path, create_disposition=CreateDisposition.FILE_OPEN) as smb_file_open:
                        with open(local_file_path, 'wb') as local_file:
                            data = smb_file_open.read(0, smb_file_open.get_maximum_read_size())
                            while data:
                                local_file.write(data)
                                data = smb_file_open.read(len(data), smb_file_open.get_maximum_read_size())

    # Start downloading the directory structure
    download_directory(smb_path, local_dir)

    # Disconnect the SMB connection
    tree.disconnect()
    session.disconnect()
    connection.disconnect()


def run_sync(args):
    output_root = "/mnt/lustre-emmy-hdd/projects/nim00007/data/moser/lightsheet/volumes"

    smb_server = "wfs-medizin.top.gwdg.de"
    smb_share = "ukon-all$"
    with open("./credentials.json") as f:
        credentials = json.load(f)
        smb_username = credentials["user"]
        smb_password = credentials["password"]

    smb_source_path = args.smb_path
    volume_name = args.volume_name

    local_directory = os.path.join(output_root, volume_name)
    sync_smb_to_local(smb_server, smb_share, smb_username, smb_password, smb_source_path, local_directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smb_path")
    parser.add_argument("volume_name")
    args = parser.parse_args()
    run_sync(args)


if __name__ == "__main__":
    main()
