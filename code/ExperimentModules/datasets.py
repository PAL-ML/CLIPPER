import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Data processing
import PIL
from PIL import Image
import cv2
import base64
import imageio
import pandas as pd
import numpy as np
import json

import os
import gdown
import zipfile
import tarfile
import wget

from .dataset_base import Dataset

class OmniglotDataset(Dataset):
	def __init__(self):
		self.name = 'omniglot'
	
	def get_class_names(self):
		omniglot_classes_orig = self.info.features['alphabet'].names
		omniglot_classes = [x.replace('_', ' ') for x in omniglot_classes_orig]
		return omniglot_classes

class LfwDataset(Dataset):
	def __init__(self):
		self.name = 'lfw'

	def get_class_names(self):
		unique_labels = set()
		
		if not self.data_generator:
			self.load('lfw', split='train', img_height=224, img_width=224)

		for i, data in enumerate(self.data_generator):
			label = data['label'].decode("utf-8").replace("_", " ")
			unique_labels.add(label)

		unique_labels = list(unique_labels)
		return unique_labels

class ClevrDataset(Dataset):
	def __init__(self):
		self.name = 'clevr'

class CUBDataset(Dataset):
	def __init__(self):
		self.name = 'caltech_birds2011'

class ImagenetADataset(Dataset):
	def __init__(self):
		self.name = 'imagenet_a'

class ImagenetRDataset(Dataset):
	def __init__(self):
		self.name = 'imagenet_r'

class MiniImagenetDataset(Dataset):
	def __init__(self):
		self.name = 'mini_imagenet'
		self.batch_size = 1
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, split='test', img_height=224, img_width=224):
		data_folder = "mini-imagenet"
		compressed_data = "mini-imagenet.zip"
		if not os.path.exists(data_folder):
			os.makedirs(data_folder)
			if not os.path.exists(compressed_data):
				self._download(compressed_data)
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "test", "val"]:
			raise Exception("Invalid split type. Choose between train, test and val...")

		split_folder = os.path.join(data_folder, split)
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			split_folder,
			image_size=(img_height, img_width),
			shuffle=False,
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np
		
	def _download(self, output_file):
		gdown.download("https://drive.google.com/uc?id=1SRCYvMStAFfU68e2EatbLgEdKZ46rpeK", output_file, quiet=False)

	def _uncompress(self, zip_file, dest_folder):
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall(dest_folder)
		
		print("Unzipping test images...")
		test_tar = tarfile.open(os.path.join(dest_folder, "test.tar"))
		test_tar.extractall(dest_folder)
		test_tar.close()

		print("Unzipping train images...")
		train_tar = tarfile.open(os.path.join(dest_folder, "train.tar"))
		train_tar.extractall(dest_folder)
		train_tar.close()

		print("Unzipping val images...")
		val_tar = tarfile.open(os.path.join(dest_folder, "val.tar"))
		val_tar.extractall(dest_folder)
		val_tar.close()

		print("Completed unzipping all files...")

class ImagenetSketchDataset(Dataset):
	def __init__(self):
		self.name = 'imagenet_sketch'
		self.batch_size = 1
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, split='test', img_height=224, img_width=224):
		data_folder = "imagenet-sketch"
		compressed_data = "ImageNet-Sketch.zip"
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download(compressed_data)
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "val"]:
			raise Exception("Invalid split type. Choose between train and val...")
		
		if split == "train":
			subset = "training"
		elif split == "val":
			subset = "validation"
		
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_folder,
			validation_split=0.2,
			subset=subset,
			seed=123,
			shuffle=False,
			image_size=(img_height, img_width),
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np

	def _download(self, output_file):
		gdown.download("https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA", output_file, quiet=False)

	def _uncompress(self, zip_file, dest_folder):
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall()
		os.rename("sketch", dest_folder)

class ImagenetTieredDataset(Dataset):
	def __init__(self):
		self.name = 'imagenet_tiered'
		self.batch_size = 1
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, split='test', img_height=224, img_width=224):
		data_folder = "imagenet-tiered"
		compressed_data = "gdrive/MyDrive/PAL_HILL_2021/Datasets/tiered-imagenet/tiered_imagenet.tar"
		if not os.path.exists(data_folder):
			os.makedirs(data_folder)
			if not os.path.exists(compressed_data):
				raise Exception("Make sure tiered_imagenet is downloaded from drive...")
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "test", "val"]:
			raise Exception("Invalid split type. Choose between train, test and val...")

		split_folder = os.path.join(data_folder, split)
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			split_folder,
			image_size=(img_height, img_width),
			shuffle=False,
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np

	def _uncompress(self, compressed_file, dest_folder):
		tar_file = tarfile.open(compressed_file)
		tar_file.extractall()
		tar_file.close()

		os.rename("tiered_imagenet", dest_folder)

class UCF101Dataset(Dataset):
	def __init__(self):
		self.name = 'ucf_101'
		self.batch_size = 1
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, split="val", img_height=224, img_width=224):
		data_folder = "UCF-101"
		compressed_data = "UCF_Images.zip"
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download(compressed_data)
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "val"]:
			raise Exception("Invalid split type. Choose between train and val...")
		
		if split == "train":
			subset = "training"
		elif split == "val":
			subset = "validation"
		
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_folder,
			validation_split=0.2,
			subset=subset,
			seed=123,
			shuffle=False,
			image_size=(img_height, img_width),
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np

	def _download(self, output_file):
		gdown.download("https://drive.google.com/uc?id=1NwERNU-266cCK_uofbEXraArleazADRS", output_file, quiet=False)

	def _uncompress(self, zip_file, dest_folder):
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall()
		os.rename("UCF_Images", dest_folder)

class ISRDataset(Dataset):
	def __init__(self):
		self.name = 'indoor_scene_recognition'
		self.batch_size = 1
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, split="val", img_height=224, img_width=224):
		data_folder = "indoor-scene-recognition"
		compressed_data = "indoorCVPR_09.tar"
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download()
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "val"]:
			raise Exception("Invalid split type. Choose between train and val...")
		
		if split == "train":
			subset = "training"
		elif split == "val":
			subset = "validation"
		
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_folder,
			validation_split=0.2,
			subset=subset,
			seed=123,
			shuffle=False,
			image_size=(img_height, img_width),
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np

	def _download(self):
		wget.download("http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar")

	def _uncompress(self, compressed_data, dest_folder):
		tar_file = tarfile.open(os.path.join(compressed_data))
		tar_file.extractall()
		tar_file.close()
		os.rename("Images", dest_folder)

class VisDA19Dataset(Dataset):
	def __init__(self):
		self.name = 'visda19'
		self.batch_size = 1
		self.domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

	def get_class_names(self):
		return self.ds.class_names

	def load(self, domain="clipart", split='val', img_height=224, img_width=224):
		if domain not in self.domains:
			raise Exception("Invalid domain. Please pick among {}".format(self.domains))
		
		data_folder = "visda19-" + domain
		compressed_data = "{}.zip".format(domain)
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download(domain)
			self._uncompress(compressed_data, data_folder, domain)
		
		if split not in ["train", "val"]:
			raise Exception("Invalid split type. Choose between train and val...")
		
		if split == "train":
			subset = "training"
		elif split == "val":
			subset = "validation"
		
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_folder,
			validation_split=0.2,
			subset=subset,
			seed=123,
			shuffle=False,
			image_size=(img_height, img_width),
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds

	def _download(self, domain):
		wget.download("http://csr.bu.edu/ftp/visda/2019/multi-source/{}.zip".format(domain))

	def _uncompress(self, zip_file, dest_folder, domain):
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall()
		os.rename(domain, dest_folder)

class ImageMatchingDataset(Dataset):
	def __init__(self):
		self.name = 'image-matching'
		self.batch_size = 1
		self.locations = []
	
	def get_class_names(self):
		return self.ds.class_names

	def load(self, location="brandenburg_gate", split='test', img_height=224, img_width=224):
		if domain not in self.locations:
			raise Exception("Invalid location. Please pick a location from the following: {}".format(self.domains))
		
		data_folder = "ImageMatching-" + location
		compressed_data = "{}.tar.gz".format(location)
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download(domain)
			self._uncompress(compressed_data, data_folder)
		
		if split not in ["train", "test"]:
			raise Exception("Invalid split type. Choose between train and test...")
		
		if split == "train":
			subset = "training"
		elif split == "test":
			subset = "validation"
		
		self.img_height = img_height
		self.img_width = img_width

		self.ds = tf.keras.preprocessing.image_dataset_from_directory(
			data_folder,
			validation_split=0.2,
			subset=subset,
			seed=123,
			shuffle=False,
			image_size=(img_height, img_width),
			batch_size=self.batch_size)
		
		ds_np = tfds.as_numpy(self.ds)
		return ds_np

	def _download(self, domain):
		wget.download("http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/{}.zip".format(domain))

	def _uncompress(self, zip_file, dest_folder):
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall()
		os.rename(domain, dest_folder)


class SUN397Dataset(Dataset):
	def __init__(self):
		self.name = 'sun397/standard-part1-120k'

class Cifar10Dataset(Dataset):
	def __init__(self):
		self.name = 'cifar10'

class COCO2017Dataset(Dataset):
	def __init__(self):
		self.name = 'coco/2017'
		self.base_dir = '/content/drive/MyDrive/PAL_HILL_2021/Datasets/Coco2017Train/'
		
	def get_class_names(self):
		raise Exception("Not implemented...")

	def load(self, split=None, img_height=224, img_width=224):
		# Throwaway argument split added for compatibility
		print("Loading labels...")
		try:
			with open(self.base_dir + 'labels.json', "r") as jsonfile:
  				labels = json.load(jsonfile)
		except FileNotFoundError:
			print("Please check if you have the following folder in your drive")
			print(self.base_dir + 'labels.json')

		print("Loading images...")
		IMG_PATH = '/content/drive/MyDrive/PAL_HILL_2021/Datasets/Coco2017Train/img_list.npz'
		img_list = np.load(IMG_LIST_PATH)['data']
		images = {}
		for name in img_list:
			images[name] = np.array(Image.open(self.base_dir + 'train2017/' + name))

		dict_keys = list(labels.keys())

		for i in dict_keys:
		  name = i.zfill(12)+".jpg"
		  if name not in img_list:
		    del labels[i]

		return images, labels


##########################
####### New Stuff ########
##########################

class IMaterialistDataset(Dataset):
	def __init__(self):
		self.name = 'fgcv_imaterialist'

	def get_class_names(self):
		data_folder = "IMaterialistFashion"
		label_data_filename = "label_descriptions.json"

		with open(os.path.join(data_folder, label_data_filename)) as f:
			labels_data_json = json.load(f)

		categories_names = []
		attributes_names = []

		for item in labels_data_json['categories']:
		    categories_names.append(item['name'])

		for item in labels_data_json['attributes']:
		    attributes_names.append(item['name'])

		return categories_names, attributes_names

	def load(self, split="train", img_height=224, img_width=224):
		data_folder = "IMaterialistFashion"

		# Runs into gdown error for large files sometimes
		# compressed_data = "imaterialist_fashion_fgvc.zip"
		# if not os.path.exists(data_folder):
		# 	if not os.path.exists(compressed_data):
		# 		self._download(compressed_data)
		# 	self._uncompress(compressed_data, data_folder)

		# to fix for gdown error
		compressed_data = "gdrive/MyDrive/PAL_HILL_2021/Datasets/imaterialist-fashion/imaterialist_fashion_fgvc.zip"
		if not os.path.exists(data_folder):
			# os.makedirs(data_folder)
			if not os.path.exists(compressed_data):
				raise Exception("Make sure imaterialist_fashion_fgvc is downloaded from drive...")
			self._uncompress(compressed_data, data_folder)

		if split == 'test':
			raise NotImplementedError("IMaterialist test labels not available")
		if split not in ["train"]:
			raise Exception("Invalid split type. Choose between train...")

		df_filename = os.path.join(data_folder, split+'.csv')
		data_df = pd.read_csv(df_filename)

		image_dir = os.path.join(data_folder, split)
		attributes_ids_col = data_df['AttributesIds'].copy().fillna('').values
		class_id_col = data_df['ClassId'].copy().values # no null values
		image_id_col = data_df['ImageId'].copy().values
		image_fn_col = data_df['ImageId'].copy().apply(lambda x: os.path.join(image_dir, x+'.jpg')).values
		encoded_pixels_col = data_df['EncodedPixels'].copy().values
		height_col = data_df['Height'].copy().values
		width_col = data_df['Width'].copy().values

		ds = tf.data.Dataset.from_tensor_slices(
		    (
		        image_id_col, 
		        image_fn_col,
		        encoded_pixels_col, 
		        height_col, 
		        width_col, 
		        class_id_col, 
		        attributes_ids_col
		    )
		)

		ds_np = ds.map(
			lambda image_id, image_fn, encoded_pixels, height, width, class_id, attributes_ids: tf.py_function(
				func=self._parse_function, 
				inp=[image_id, image_fn, encoded_pixels, height, width, class_id, attributes_ids], 
				Tout=[tf.string, tf.uint8, tf.int64, tf.int64, tf.int64]
				),
			num_parallel_calls=tf.data.experimental.AUTOTUNE
			).as_numpy_iterator()


		return ds_np

	def _download(self, output_file):
		# modified to fgcv zip file
		gdown.download("https://drive.google.com/uc?id=103-eKcrpXf-do6SMH1-zkIs9niWVgGl9", output_file, quiet=False)

	def _uncompress(self, zip_file, dest_folder):
		# modified for fgcv file
		if not os.path.isdir(dest_folder):
		    os.makedirs(dest_folder)
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		    zip_ref.extractall(dest_folder)

	def _process_encoded_pixels(self, row):
		def get_pixel_loc(value):
		    x = value%height
		    y = value//height

		    return x, y

		def process_encoded_pixels_string(encoded_pixels):
		    mask_pixels = []
		    ep_list = [int(ep_item) for ep_item in encoded_pixels.split(' ')]

		    idx = 0
		    while idx < len(ep_list):
		        pixel = ep_list[idx]
		        num_pixels = ep_list[idx+1]

		        for np in range(num_pixels):
		            mask_pixels.append(pixel+np)
		        
		        idx += 2

		    return mask_pixels

		def get_mask(mask_pixels, height, width):
		    mask = np.zeros((height, width))
		    for mp in mask_pixels:
		        x, y = get_pixel_loc(mp)
		        mask[x, y] = 1
		    
		    return mask

		encoded_pixels = row[0]# .numpy().decode('utf=8')
		height = int(row[1])
		width = int(row[2])
		mask_pixels = process_encoded_pixels_string(encoded_pixels)
		mask = get_mask(mask_pixels, height, width)

		return mask

	def _parse_function(
		self,
		image_id, 
		image_fn, 
		encoded_pixels, 
		height, 
		width, 
		class_id, 
		attributes_ids
	):
		image_string = tf.io.read_file(image_fn)
		# Decode it into a dense vector
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)
		# Resize it to fixed shape
		# image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
		# Normalize it from [0, 255] to [0.0, 1.0]
		# image_normalized = image_resized / 255.0

		ep = encoded_pixels.numpy().decode('utf-8')
		h = height.numpy()
		w = width.numpy()
		mask = self._process_encoded_pixels((ep, h, w))

		attributes_arr = attributes_ids.numpy().decode('utf-8').split(',')
		attributes = np.array(
			[int(val) for val in attributes_ids.numpy().decode('utf-8').split(',') if val != '']
			).astype(np.int64)

		return image_id, image_decoded, mask, class_id, attributes

class YaleFaces(Dataset):
	def __init__(self):
		self.name = 'yale_faces'
		self.class_names = None

	def get_class_names(self):
		return self.class_names

	def load(self, split="train", img_width=None, img_height=None):
		# Added dummy arguments img_width and img_height for compatibility
		data_folder = "yalefaces"
		compressed_data = "yalefaces.zip"
		if not os.path.exists(data_folder):
			if not os.path.exists(compressed_data):
				self._download()
			self._uncompress(compressed_data)

		if split not in ["train"]:
			# split isn't being used as only train data is available
			raise Exception("Invalid split type. Only train available...")

		files_to_ignore = ['Readme.txt', 'subject01.glasses.gif']
		files_to_rename = {'subject01.gif': 'subject01.centerlight'}

		images = []
		labels = []
		self.class_names = (set(), set())
		# data will always be loaded in alphabetical order of the filenames
		data_filenames = sorted(os.listdir(data_folder))
		for fn in data_filenames:
		    if fn in files_to_ignore:
		        continue

		    image = cv2.cvtColor(
		        np.array(Image.open(os.path.join(data_folder, fn))), 
		        cv2.COLOR_GRAY2RGB
		        )
		    if fn in files_to_rename:
		        fn = files_to_rename[fn]
		    label = fn.split('.')

		    images.append(image)
		    labels.append(label)

		    self.class_names[0].add(label[0])
		    self.class_names[1].add(label[1])

		self.class_names = (list(self.class_names[0]), list(self.class_names[1]))

		ds = tf.data.Dataset.from_tensor_slices((images, labels)).as_numpy_iterator()
		return ds

	def _download(self):
		data_link = "http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip"
		wget.download(data_link)

	def _uncompress(self, data_zipfile):
		with zipfile.ZipFile(data_zipfile, 'r') as zip_ref:
			zip_ref.extractall()


class UTKFaces(Dataset):
	def __init__(self):
	    self.name = 'utk_faces'
	    self.class_names = None

	def get_class_names(self):
	    return self.class_names

	def load(self, split="train", img_width=None, img_height=None):
		# Added dummy arguments img_width and img_height for compatibility
		data_folder = "UTKFace"
		compressed_data = "utk_faces.tar.gz"
		if not os.path.exists(data_folder):
		    if not os.path.exists(compressed_data):
		        self._download(compressed_data)
		    self._uncompress(compressed_data)

		if split not in ["train"]:
		    # split isn't being used as only train data is available
		    raise Exception("Invalid split type. Only train available...")

		image_filenames = []
		labels = []

		def age_map(age):
		    if isinstance(age, str):
		        age = int(age)

		    if age < 11:
		        return 'Child'
		    elif age < 18:
		        return 'Teen'
		    elif age < 40:
		    	return 'Adult'
		    elif age < 65:
		        return 'Older Adult'
		    elif age < 90:
		    	return 'Senior'
		    else:
		        return 'Older Senior'

		gender_map = {'0': "Male", '1': "Female"}
		race_map = {'0':"White", '1':"Black", '2':"Asian", '3':"Indian", '4':"Others"}
		unknown_val = 'Unknown'

		age_class_names = ['Child', 'Teen', 'Adult', 'Older Adult', 'Senior', 'Older Senior']
		gender_class_names = list(gender_map.values())
		race_class_names = list(race_map.values())

		self.class_names = (
			age_class_names, gender_class_names, race_class_names
			)

		for fn in sorted(os.listdir(data_folder)):
		    if not fn.endswith('jpg'):
		        continue

		    _labels = fn.split('_')
		    try:
		        labels.append(
		            [
		            age_map(_labels[0]), 
		            gender_map[_labels[1]], 
		            race_map[_labels[2]]
		            ]
		            )
		        image_filenames.append(os.path.join(data_folder, fn))
		    except:
		        # Errors observed only on images with no label for race
		        labels.append(
		            [age_map(_labels[0]), gender_map[_labels[1]], unknown_val]
		            )
		        image_filenames.append(os.path.join(data_folder, fn))

		ds = tf.data.Dataset.from_tensor_slices((image_filenames, labels))
		ds = ds.map(
		    self._parse_function, 
		    num_parallel_calls=tf.data.experimental.AUTOTUNE
		    ).as_numpy_iterator()

		return ds

	def _download(self, compressed_data):
	    data_link = "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"
	    gdown.download(data_link, compressed_data, quiet=False)

	def _uncompress(self, data_zipfile):
	    tar = tarfile.open(data_zipfile, "r:gz")
	    tar.extractall()
	    tar.close()

	def _parse_function(self, image_filename, label):
		image_string = tf.io.read_file(image_filename)
		# Decode it into a dense vector
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)
		# Resize it to fixed shape
		# image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
		# Normalize it from [0, 255] to [0.0, 1.0]
		# image_normalized = image_resized / 255.0

		return image_decoded, label

class CelebAFaces(Dataset):
	def __init__(self):
		self.name = 'celeba_faces'
		self.class_names = None

	def get_class_names(self):
	    return self.class_names

	def load(self, split="train", img_width=None, img_height=None):
		# Added dummy arguments img_width and img_height for compatibility
		data_folder = "img_align_celeba"
		compressed_data = "img_align_celeba.zip"
		labels_data = "list_attr_celeba.txt"
		splits_data = "list_eval_partition.txt"
		if not os.path.exists(data_folder):
		    if not os.path.exists(compressed_data):
		        self._download(compressed_data, labels_data, splits_data)
		    self._uncompress(compressed_data)

		if split not in ['train', 'val', 'test']:
			raise Exception("Invalid split type. Choose between train, val, test...")

		split_val = {'train':0, 'val':1, 'test':2}

		eval_partition = pd.read_csv(
			splits_data, 
			sep=' ', 
			header=None)
		eval_partition.columns = ['filename', 'value']
		split_fn_list = list(
			eval_partition.loc[eval_partition.loc[:, 'value'] == split_val[split], 'filename']
			)

		attr_labels = pd.read_csv(labels_data, sep='\s+', header=1)
		attr_labels.replace(-1, 0, inplace=True)
		self.class_names = list(attr_labels.columns)
		split_attr_labels = attr_labels.loc[split_fn_list].values
		split_fn_list = [os.path.join(data_folder, fn) for fn in split_fn_list]

		ds = tf.data.Dataset.from_tensor_slices((split_fn_list, split_attr_labels))

		ds = ds.map(
			self._parse_function, 
			num_parallel_calls=tf.data.experimental.AUTOTUNE
			).as_numpy_iterator()

		return ds

	def _download(self, compressed_data, labels_data, splits_data):
		# time.sleep(100)
		gdown.download("https://drive.google.com/uc?id=1UGlMQ1AGCKUx1y0MvXvJTTaBFiTvbIq-", compressed_data, quiet=False)
		# time.sleep(100)
		gdown.download("https://drive.google.com/uc?id=1g_W8UiX8KU7d9e2pxhYizJP9eGcJkqog", labels_data, quiet=False)
		# time.sleep(100)
		gdown.download("https://drive.google.com/uc?id=192UAAHs-LK8FviyVeC7loTy4p4N3cGSB", splits_data, quiet=False)

		# If gdown gives error will have to switch this to ImagenetTiered-like structure

	def _uncompress(self, compressed_data):
		with zipfile.ZipFile(compressed_data, 'r') as zip_ref:
			zip_ref.extractall()

	def _parse_function(self, image_filename, label):
		image_string = tf.io.read_file(image_filename)
		# Decode it into a dense vector
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)
		# Resize it to fixed shape
		# image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
		# Normalize it from [0, 255] to [0.0, 1.0]
		# image_normalized = image_resized / 255.0

		return image_decoded, label
