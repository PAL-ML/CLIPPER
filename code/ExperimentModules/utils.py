import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
from oauth2client.client import GoogleCredentials

# Data processing
import PIL
import base64
import imageio
import pandas as pd
import numpy as np
import json

from PIL import Image
import cv2

# Misc
import logging

logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

def save_npy(filepath, data):
	np.savez_compressed(filepath, data=data)
	print("Data saved to {}".format(filepath))

def load_npy(filepath):
	loaded_data = np.load(filepath)
	print("Data loaded from {}".format(filepath))
	return loaded_data['data']

def get_folder_id(drive, parent_folder_id, folder_name):
    """ 
		Check if destination folder exists and return it's ID
	"""
    folder_exists = False

    # Auto-iterate through all files in the parent folder.
    file_list = GoogleDriveFileList()
    try:
        file_list = drive.ListFile(
			{'q': "'{0}' in parents and trashed=false".format(parent_folder_id)}
		).GetList()
	# Exit if the parent folder doesn't exist
    except googleapiclient.errors.HttpError as err:
		# Parse error message
        message = ast.literal_eval(err.content)['error']['message']
        if message == 'File not found: ':
            print(message + folder_name)
            exit(1)
		# Exit with stacktrace in case of other error
        else:
            raise

	# Find the the destination folder in the parent folder's files
    for file1 in file_list:
        if file1['title'] == folder_name:
            print('title: %s, id: %s' % (file1['title'], file1['id']))
            folder_exists = True
            return file1['id'], folder_exists

    return None, False

def create_expt_dir(drive, parentid, folder_name):
	folderid, folder_exists = get_folder_id(drive, parentid, folder_name)

	if not folder_exists:
		folder_metadata = {'title' : folder_name, 'mimeType' : 'application/vnd.google-apps.folder', 'parents':[{'id':parentid}]}
		folder = drive.CreateFile(folder_metadata)
		folder.Upload()

		#Get folder info and print to screen
		foldertitle = folder['title']
		folderid = folder['id']
		print('title: %s, id: %s' % (foldertitle, folderid))
		return folderid
	else:
		print("Experiment folder already exists. WARNING: Following with this run might overwrite existing results stored.")
		return folderid

def save_to_drive(drive, folderid, filepath):
	#Upload file to folder
	file = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folderid}]})
	file.SetContentFile(filepath)
	file.Upload()
	print("Uploaded {} to https://drive.google.com/drive/u/1/folders/{}".format(filepath, folderid))

def load_all_from_drive_folder(drive, folderid):
	file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folderid)}).GetList()
	for i, file1 in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
		print('Downloading {} from GDrive ({}/{})'.format(file1['title'], i, len(file_list)))
		file1.GetContentFile(file1['title'])

def download_file_by_name(drive, folderid, filename):
	file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folderid)}).GetList()
	for i, file1 in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
		if filename == file1['title']:
			print('Downloading {} from GDrive'.format(file1['title']))
			file1.GetContentFile(file1['title'])

def delete_file_by_name(drive, folderid, filename):
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folderid)}).GetList()
    for i, file1 in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
        if filename == file1['title']:
            print('Deleting {} from GDrive'.format(file1['title']))
            file2 = drive.CreateFile({'id': file1['id']})

            file2.Trash()
            file2.UnTrash()
            file2.Delete()		