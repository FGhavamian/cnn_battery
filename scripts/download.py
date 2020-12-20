import os 
from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile


file_id_processed_data = '1gTUa4_IcoDeYic9ifYINAg81q9_p3sQ_'
dest_processed_data = './processed.zip'

file_id_raw_data = '1we-5EDTwoBe1YN-kXkzgCWRdqx-IcuVL'
dest_raw_data = './raw.zip'

file_id_model = '1Y3oCh7GYImmi1J364wFPXHonYKYZElgA'
dest_model = './model.zip'


file_ids = [file_id_processed_data, file_id_raw_data, file_id_model]
dests = [dest_processed_data, dest_raw_data, dest_model]

for file_id, dest in zip(file_ids, dests):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest,
                                        unzip=False)


    with zipfile.ZipFile(dest, 'r') as zip_ref:
        zip_ref.extractall()

    # os.remove(dest)