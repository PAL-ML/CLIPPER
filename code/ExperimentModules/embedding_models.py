# ML Libraries
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tensorflow import keras

# Data processing
import PIL
import base64
import imageio
import pandas as pd
import numpy as np
import json

from PIL import Image
import cv2

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# Models
import clip

# Misc
import progressbar
import logging
from abc import ABC, abstractmethod
import time
import urllib.request
import os

# Utils
from .utils import save_npy, save_to_drive

logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

class BaseEmbeddingWrapper(ABC):
	def __init__(self):
		self.embedding_model = None

	@abstractmethod
	def load_model(self):
		pass

	@abstractmethod
	def preprocess_data(self, data):
		pass

	@abstractmethod
	def embed_images(self, images):
		pass

	def save_embeddings_to_drive(
		self, 
		embeddings, 
		embeddings_filepath, 
		drive, 
		folderid
	):
		save_npy(embeddings_filepath, np.array(embeddings))
		save_to_drive(drive, folderid, embeddings_filepath)

class InceptionV3EmbeddingWrapper(BaseEmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'InceptionV3'
		self.model_description = "IncpetionV3 loaded with pretrained imagenet weights from keras.applications."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		self.embedding_model = keras.applications.InceptionV3(
			include_top=include_top, 
			weights='imagenet', 
			input_shape=input_shape,
			pooling='avg' # 'max'
			)

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

	def preprocess_data(self, data, verbose=False):
		start_time = time.time()

		preprocessed_data = keras.applications.inception_v3.preprocess_input(
			data
			)

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Data processed in {} seconds".format(total_time))

		return preprocessed_data

	def embed_images(self, images, batch_size=None, verbose=False):
		ndim = len(images.shape)

		if ndim == 3:
			images = tf.expand_dims(images, 0)

		start_time = time.time()

		if batch_size is None:
			embeddings = self.embedding_model.predict(images)
		else:
			num_images = images.shape[0]
			num_iter = int(np.ceil(num_images / batch_size))
			_embeddings = []

			if verbose:
				print("Embedding images...")
				widgets = [
					' [', 
					progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
					'] ', 
					progressbar.Bar('*'),' (', 
					progressbar.ETA(), ') ', 
					]
				pbar = progressbar.ProgressBar(
					max_value=num_iter, widgets=widgets
					).start()

			for i in range(num_iter):
				if (i+1)*batch_size > num_images:
					image_batch = images[i*batch_size:]
				else:
					image_batch = images[i*batch_size:(i+1)*batch_size]

				embeddings_batch = self.embedding_model(image_batch)
				_embeddings.append(embeddings_batch.numpy().reshape(embeddings_batch.shape[0], -1))
				del embeddings_batch

				if verbose:
					pbar.update(i+1)

			embeddings = np.concatenate(_embeddings)
			del _embeddings       

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Images embedded in {} seconds".format(total_time))

		return embeddings

class Resnet50EmbeddingWrapper(BaseEmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'Resnet50'
		self.model_description = "Resnet50 loaded with pretrained imagenet weights from torchvision.models."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		resnet50_model = models.resnet.resnet50(pretrained=True)

		# freeze all layers but the last fc
		for name, param in resnet50_model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False

		if include_top:
			self.embedding_model = resnet50_model
		else:
			self.embedding_model = nn.Sequential(*list(resnet50_model.children())[:-1])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

	def preprocess_data(self, data, batch_size=50, verbose=False):

		if not isinstance(data, np.ndarray):
			data = data.numpy()
		if np.max(data) <= 1:
			data = data * 255
		data = data.astype(np.uint8)

		ndim = len(data.shape)
		if ndim == 3:
			data = np.expand_dims(data, 0)

		IMG_SIZE = 224
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                                 std=[0.229, 0.224, 0.225])
		transform_images = transforms.Compose([
		    transforms.Resize(IMG_SIZE),
		    transforms.ToTensor(),
		    normalize,
			])

		start_time = time.time()

		if batch_size is not None:
			num_images = data.shape[0]
			num_iter = int(np.ceil(num_images/batch_size))

			_transformed_images = []

			if verbose:
				print("Preprocessing data...")
				widgets = [
					' [', 
					progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
					'] ', 
					progressbar.Bar('*'),' (', 
					progressbar.ETA(), ') ', 
					]
				pbar = progressbar.ProgressBar(
					max_value=num_iter, widgets=widgets
					).start()

			for i in range(num_iter):
				if (i+1)*batch_size > num_images:
					data_batch = data[i*batch_size:]
				else:
					data_batch = data[i*batch_size:(i+1)*batch_size]

				transformed_images_batch = torch.stack(
					[transform_images(Image.fromarray(im)) for im in data_batch]
					)
				_transformed_images.append(transformed_images_batch)
				del transformed_images_batch

				if verbose:
					pbar.update(i+1)

			transformed_images = torch.cat(_transformed_images)
			del _transformed_images
		else: 
			# transforms need input to be PIL Images
			transformed_images = torch.stack([transform_images(Image.fromarray(im)) for im in data])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Data processed in {} seconds".format(total_time))

		return transformed_images

	def embed_images(self, images, batch_size=50, verbose=False):
		ndim = len(images.shape)
		if ndim == 3:
			images = torch.unsqueeze(images, 0)

		num_images = images.shape[0]
		num_iter = int(np.ceil(num_images/batch_size))

		start_time = time.time()

		_embeddings = []

		if verbose:
			print("Embedding images...")
			widgets = [
				' [', 
				progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
				'] ', 
				progressbar.Bar('*'),' (', 
				progressbar.ETA(), ') ', 
				]
			pbar = progressbar.ProgressBar(
				max_value=num_iter, widgets=widgets
				).start()

		for i in range(num_iter):
			if (i+1)*batch_size > num_images:
				image_batch = images[i*batch_size:]
			else:
				image_batch = images[i*batch_size:(i+1)*batch_size]

			embeddings_batch = self.embedding_model(image_batch)
			_embeddings.append(embeddings_batch.numpy().reshape(embeddings_batch.shape[0], -1))

			if verbose:
				pbar.update(i+1)

		embeddings = np.concatenate(_embeddings)
		del _embeddings

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Images embedded in {} seconds".format(total_time))

		return embeddings

class MoCoResnet50EmbeddingWrapper(Resnet50EmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'MoCoResnet50'
		self.model_description = "Resnet50 from torchvision.models loaded with MoCo weights."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		if not os.path.isdir('/content/weights'):
			os.makedirs('/content/weights')

		checkpoint_url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
		checkpoint_filepath = '/content/weights/moco_v2_800ep_pretrain.pth.tar'
		urllib.request.urlretrieve(checkpoint_url, checkpoint_filepath)

		resnet50_model = models.resnet.resnet50(pretrained=True)

		# freeze all layers but the last fc
		for name, param in resnet50_model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False

		checkpoint = torch.load(checkpoint_filepath)
		# rename moco pre-trained keys
		state_dict = checkpoint['state_dict']
		for k in list(state_dict.keys()):
			# retain only encoder_q up to before the embedding layer
			if (k.startswith('module.encoder_q') and 
				not k.startswith('module.encoder_q.fc')
			):
				# remove prefix
				state_dict[k[len("module.encoder_q."):]] = state_dict[k]
			# delete renamed or unused k
			del state_dict[k]

		msg = resnet50_model.load_state_dict(state_dict, strict=False)

		if include_top:
			self.embedding_model = resnet50_model
		else:
			self.embedding_model = nn.Sequential(*list(resnet50_model.children())[:-1])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

class PCLResnet50EmbeddingWrapper(Resnet50EmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'MoCoResnet50'
		self.model_description = "Resnet50 from torchvision.models loaded with MoCo weights."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		if not os.path.isdir('/content/weights'):
			os.makedirs('/content/weights')

		checkpoint_url = 'https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar'
		checkpoint_filepath = '/content/weights/PCL_v2_epoch200.pth.tar'
		urllib.request.urlretrieve(checkpoint_url, checkpoint_filepath)

		resnet50_model = models.resnet.resnet50(pretrained=True)

		# freeze all layers but the last fc
		for name, param in resnet50_model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False

		checkpoint = torch.load(checkpoint_filepath)
		# PCL weights are saved in the same format as MoCo
		# rename moco pre-trained keys
		state_dict = checkpoint['state_dict']
		for k in list(state_dict.keys()):
			# retain only encoder_q up to before the embedding layer
			if (k.startswith('module.encoder_q') and 
				not k.startswith('module.encoder_q.fc')
			):
				# remove prefix
				state_dict[k[len("module.encoder_q."):]] = state_dict[k]
			# delete renamed or unused k
			del state_dict[k]

		msg = resnet50_model.load_state_dict(state_dict, strict=False)

		if include_top:
			self.embedding_model = resnet50_model
		else:
			self.embedding_model = nn.Sequential(*list(resnet50_model.children())[:-1])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

class SwAVResnet50EmbeddingWrapper(Resnet50EmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'SwAVResnet50'
		self.model_description = "SwAV Resnet50 model from feacebook repository."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		swav_model = torch.hub.load(
			'facebookresearch/swav', 'resnet50', pretrained=True
			)

		# freeze all layers but the last fc
		for name, param in swav_model.named_parameters():
			if name not in ['fc.weight', 'fc.bias']:
				param.requires_grad = False

		if include_top:
			self.embedding_model = swav_model
		else:
			self.embedding_model = nn.Sequential(*list(swav_model.children())[:-1])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))


class SimCLREmbeddingWrapper(BaseEmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'SimCLR'
		self.model_description = "SimCLR loaded with pretrained weights from saved tf2 saved_model checkpoint."

		self.FLAGS_color_jitter_strength = 0.3
		self.CROP_PROPORTION = 0.875
		self.include_top = False

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		saved_simclr_model_path = 'gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/'
		self.embedding_model = tf.saved_model.load(saved_simclr_model_path)

		end_time = time.time()
		total_time = end_time - start_time

		self.include_top = include_top

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

	def preprocess_data(self, data, verbose=False):
		def _preprocess(x):
			x = self.preprocess_image(
				x, 224, 224, is_training=False, color_distort=False
				)
			return x

		ndim = len(data.shape)
		if ndim == 3:
			data = tf.expand_dims(data, 0)

		start_time = time.time()

		preprocessed_data = tf.stack([_preprocess(im) for im in data])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Data processed in {} seconds".format(total_time))

		return preprocessed_data

	def embed_images(self, images, batch_size=50, embedding_layer='final_avg_pool', verbose=False):
		# if include_top is True(set in load_model) ignore the embedding layer 
		# and fetch the last layer embeddings
		if self.include_top:
			embedding_layer = 'logits_sup'

		ndim = len(images.shape)

		if ndim == 3:
			images = tf.expand_dims(images, 0)

		num_images = images.shape[0]
		num_iter = int(np.ceil(num_images/batch_size))

		_embeddings = []
		start_time = time.time()

		if verbose:
			print("Embedding images...")
			widgets = [
				' [', 
				progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
				'] ', 
				progressbar.Bar('*'),' (', 
				progressbar.ETA(), ') ', 
				]
			pbar = progressbar.ProgressBar(
				max_value=num_iter, widgets=widgets
				).start()

		for i in range(num_iter):
			if (i+1)*batch_size > num_images:
				image_batch = images[i*batch_size:]
			else:
				image_batch = images[i*batch_size:(i+1)*batch_size]

			embeddings_batch = self.embedding_model(
				image_batch, trainable=False
				)[embedding_layer]
			_embeddings.append(
				embeddings_batch.numpy().reshape(embeddings_batch.shape[0], -1)
				)
			del embeddings_batch

			if verbose:
				pbar.update(i+1)

		embeddings = np.concatenate(_embeddings)
		del _embeddings

		end_time = time.time()
		total_time = end_time - start_time

		if ndim == 3:
			embeddings = embeddings[0]

		if verbose:
			print("Images embedded in {} seconds".format(total_time))

		return embeddings

	# Functions copied from  https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb
	def random_apply(self, func, p, x):
		"""Randomly apply function func to x with probability p."""
		return tf.cond(
		    tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
		            tf.cast(p, tf.float32)),
		    lambda: func(x),
		    lambda: x
		    )

	def random_brightness(self, image, max_delta, impl='simclrv2'):
		"""A multiplicative vs additive change of brightness."""
		if impl == 'simclrv2':
			factor = tf.random_uniform(
				[], tf.maximum(1.0 - max_delta, 0), 1.0 + max_delta
				)
			image = image * factor
		elif impl == 'simclrv1':
			image = self.random_brightness(image, max_delta=max_delta)
		else:
			raise ValueError('Unknown impl {} for random brightness.'.format(impl))
		return image

	def to_grayscale(self, image, keep_channels=True):
		image = tf.image.rgb_to_grayscale(image)
		if keep_channels:
			image = tf.tile(image, [1, 1, 3])
		return image

	def color_jitter(self, 
		image,
		strength,
		random_order=True
	):
		"""Distorts the color of the image.
		Args:
		  image: The input image tensor.
		  strength: the floating number for the strength of the color augmentation.
		  random_order: A bool, specifying whether to randomize the jittering order.
		Returns:
		  The distorted image tensor.
		"""
		brightness = 0.8 * strength
		contrast = 0.8 * strength
		saturation = 0.8 * strength
		hue = 0.2 * strength
		if random_order:
			return self.color_jitter_rand(image, brightness, contrast, saturation, hue)
		else:
			return self.color_jitter_nonrand(image, brightness, contrast, saturation, hue)

	def color_jitter_nonrand(self, 
		image, 
		brightness=0, 
		contrast=0, 
		saturation=0, 
		hue=0
	):
		"""Distorts the color of the image (jittering order is fixed).
		Args:
		  image: The input image tensor.
		  brightness: A float, specifying the brightness for color jitter.
		  contrast: A float, specifying the contrast for color jitter.
		  saturation: A float, specifying the saturation for color jitter.
		  hue: A float, specifying the hue for color jitter.
		Returns:
		  The distorted image tensor.
		"""
		with tf.name_scope('distort_color'):
			def apply_transform(i, x, brightness, contrast, saturation, hue):
				"""Apply the i-th transformation."""
				if brightness != 0 and i == 0:
					x = self.random_brightness(x, max_delta=brightness)
				elif contrast != 0 and i == 1:
					x = tf.image.random_contrast(
						x, lower=1-contrast, upper=1+contrast
						)
				elif saturation != 0 and i == 2:
					x = tf.image.random_saturation(
						x, lower=1-saturation, upper=1+saturation
						)
				elif hue != 0:
					x = tf.image.random_hue(x, max_delta=hue)
				return x

			for i in range(4):
				image = apply_transform(i, image, brightness, contrast, saturation, hue)
				image = tf.clip_by_value(image, 0., 1.)
			return image

	def color_jitter_rand(self, image, brightness=0, contrast=0, saturation=0, hue=0):
		"""Distorts the color of the image (jittering order is random).
		Args:
		  image: The input image tensor.
		  brightness: A float, specifying the brightness for color jitter.
		  contrast: A float, specifying the contrast for color jitter.
		  saturation: A float, specifying the saturation for color jitter.
		  hue: A float, specifying the hue for color jitter.
		Returns:
		  The distorted image tensor.
		"""
		with tf.name_scope('distort_color'):
			def apply_transform(i, x):
				"""Apply the i-th transformation."""
				def brightness_foo():
					if brightness == 0:
						return x
					else:
						return self.random_brightness(x, max_delta=brightness)
				def contrast_foo():
					if contrast == 0:
						return x
					else:
						return tf.image.random_contrast(
							x, lower=1-contrast, upper=1+contrast
							)
				def saturation_foo():
					if saturation == 0:
						return x
					else:
						return tf.image.random_saturation(
						    x, lower=1-saturation, upper=1+saturation
						    )
				def hue_foo():
					if hue == 0:
						return x
					else:
						return tf.image.random_hue(x, max_delta=hue)
				x = tf.cond(tf.less(i, 2),
			            lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
			            lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo)
			            )
				return x

			perm = tf.random_shuffle(tf.range(4))
			for i in range(4):
				image = apply_transform(perm[i], image)
				image = tf.clip_by_value(image, 0., 1.)
			return image

	def _compute_crop_shape(
		self, 
		image_height, 
		image_width, 
		aspect_ratio, 
		crop_proportion
	):
		"""Compute aspect ratio-preserving shape for central crop.
		The resulting shape retains `crop_proportion` along one side and a proportion
		less than or equal to `crop_proportion` along the other side.
		Args:
		  image_height: Height of image to be cropped.
		  image_width: Width of image to be cropped.
		  aspect_ratio: Desired aspect ratio (width / height) of output.
		  crop_proportion: Proportion of image to retain along the less-cropped side.
		Returns:
		  crop_height: Height of image after cropping.
		  crop_width: Width of image after cropping.
		"""
		image_width_float = tf.cast(image_width, tf.float32)
		image_height_float = tf.cast(image_height, tf.float32)

		def _requested_aspect_ratio_wider_than_image():
			crop_height = tf.cast(tf.math.rint(
				crop_proportion / aspect_ratio * image_width_float), tf.int32
			)
			crop_width = tf.cast(tf.math.rint(
				crop_proportion * image_width_float), tf.int32
			)
			return crop_height, crop_width

		def _image_wider_than_requested_aspect_ratio():
			crop_height = tf.cast(
				tf.math.rint(crop_proportion * image_height_float), tf.int32
				)
			crop_width = tf.cast(tf.math.rint(
				crop_proportion * aspect_ratio *
				image_height_float), tf.int32
			)
			return crop_height, crop_width

		return tf.cond(
		    aspect_ratio > image_width_float / image_height_float,
		    _requested_aspect_ratio_wider_than_image,
		    _image_wider_than_requested_aspect_ratio
		    )

	def center_crop(self, image, height, width, crop_proportion):
		"""Crops to center of image and rescales to desired size.
		Args:
		  image: Image Tensor to crop.
		  height: Height of image to be cropped.
		  width: Width of image to be cropped.
		  crop_proportion: Proportion of image to retain along the less-cropped side.
		Returns:
		  A `height` x `width` x channels Tensor holding a central crop of `image`.
		"""
		shape = tf.shape(image)
		image_height = shape[0]
		image_width = shape[1]
		crop_height, crop_width = self._compute_crop_shape(
		    image_height, image_width, height / width, crop_proportion
		    )
		offset_height = ((image_height - crop_height) + 1) // 2
		offset_width = ((image_width - crop_width) + 1) // 2
		image = tf.image.crop_to_bounding_box(
		    image, offset_height, offset_width, crop_height, crop_width
		    )

		image = tf.compat.v1.image.resize_bicubic([image], [height, width])[0]

		return image

	def distorted_bounding_box_crop(self, 
		image,
		bbox,
		min_object_covered=0.1,
		aspect_ratio_range=(0.75, 1.33),
		area_range=(0.05, 1.0),
		max_attempts=100,
		scope=None
	):
		"""Generates cropped_image using one of the bboxes randomly distorted.
		See `tf.image.sample_distorted_bounding_box` for more documentation.
		Args:
		  image: `Tensor` of image data.
		  bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
		      where each coordinate is [0, 1) and the coordinates are arranged
		      as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
		      image.
		  min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
		      area of the image must contain at least this fraction of any bounding
		      box supplied.
		  aspect_ratio_range: An optional list of `float`s. The cropped area of the
		      image must have an aspect ratio = width / height within this range.
		  area_range: An optional list of `float`s. The cropped area of the image
		      must contain a fraction of the supplied image within in this range.
		  max_attempts: An optional `int`. Number of attempts at generating a cropped
		      region of the image of the specified constraints. After `max_attempts`
		      failures, return the entire image.
		  scope: Optional `str` for name scope.
		Returns:
		  (cropped image `Tensor`, distorted bbox `Tensor`).
		"""
		with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
			shape = tf.shape(image)
			sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
				shape,
				bounding_boxes=bbox,
				min_object_covered=min_object_covered,
				aspect_ratio_range=aspect_ratio_range,
				area_range=area_range,
				max_attempts=max_attempts,
				use_image_if_no_bounding_boxes=True
				)
			bbox_begin, bbox_size, _ = sample_distorted_bounding_box

			# Crop the image to the specified bounding box.
			offset_y, offset_x, _ = tf.unstack(bbox_begin)
			target_height, target_width, _ = tf.unstack(bbox_size)
			image = tf.image.crop_to_bounding_box(
				image, offset_y, offset_x, target_height, target_width
				)

			return image

	def crop_and_resize(self, image, height, width):
		"""Make a random crop and resize it to height `height` and width `width`.
		Args:
		  image: Tensor representing the image.
		  height: Desired image height.
		  width: Desired image width.
		Returns:
		  A `height` x `width` x channels Tensor holding a random crop of `image`.
		"""
		bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
		aspect_ratio = width / height
		image = self.distorted_bounding_box_crop(
		    image,
		    bbox,
		    min_object_covered=0.1,
		    aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
		    area_range=(0.08, 1.0),
		    max_attempts=100,
		    scope=None
		    )
		return tf.compat.v1.image.resize_bicubic([image], [height, width])[0]

	# Not being used
	def gaussian_blur(self, image, kernel_size, sigma, padding='SAME'):
		"""Blurs the given image with separable convolution.
		Args:
		  image: Tensor of shape [height, width, channels] and dtype float to blur.
		  kernel_size: Integer Tensor for the size of the blur kernel. This is should
		    be an odd number. If it is an even number, the actual kernel size will be
		    size + 1.
		  sigma: Sigma value for gaussian operator.
		  padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
		Returns:
		  A Tensor representing the blurred image.
		"""
		radius = tf.to_int32(kernel_size / 2)
		kernel_size = radius * 2 + 1
		x = tf.to_float(tf.range(-radius, radius + 1))
		blur_filter = tf.exp(
		    -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.to_float(sigma), 2.0)))
		blur_filter /= tf.reduce_sum(blur_filter)
		# One vertical and one horizontal filter.
		blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
		blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
		num_channels = tf.shape(image)[-1]
		blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
		blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
		expand_batch_dim = image.shape.ndims == 3
		if expand_batch_dim:
			# Tensorflow requires batched input to convolutions, which we can fake with
			# an extra dimension.
			image = tf.expand_dims(image, axis=0)
		blurred = tf.nn.depthwise_conv2d(
		    image, blur_h, strides=[1, 1, 1, 1], padding=padding)
		blurred = tf.nn.depthwise_conv2d(
		    blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
		if expand_batch_dim:
			blurred = tf.squeeze(blurred, axis=0)
		return blurred

	def random_crop_with_resize(self, image, height, width, p=1.0):
		"""Randomly crop and resize an image.
		Args:
		  image: `Tensor` representing an image of arbitrary size.
		  height: Height of output image.
		  width: Width of output image.
		  p: Probability of applying this transformation.
		Returns:
		  A preprocessed image `Tensor`.
		"""
		def _transform(image):  # pylint: disable=missing-docstring
			image = self.crop_and_resize(image, height, width)
			return image
		return self.random_apply(_transform, p=p, x=image)

	def random_color_jitter(self, image, p=1.0):
		def _transform(image):
			color_jitter_t = functools.partial(
				self.color_jitter, strength=self.FLAGS_color_jitter_strength
				)
			image = self.random_apply(color_jitter_t, p=0.8, x=image)
			return self.random_apply(self.to_grayscale, p=0.2, x=image)
		return self.random_apply(_transform, p=p, x=image)

	# Not being used
	def random_blur(self, image, height, width, p=1.0):
		"""Randomly blur an image.
		Args:
		  image: `Tensor` representing an image of arbitrary size.
		  height: Height of output image.
		  width: Width of output image.
		  p: probability of applying this transformation.
		Returns:
		  A preprocessed image `Tensor`.
		"""
		del width
		def _transform(image):
			sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
			return self.gaussian_blur(
				image, kernel_size=height//10, sigma=sigma, padding='SAME'
				)
		return self.random_apply(_transform, p=p, x=image)


	def batch_random_blur(
		self, 
		images_list, 
		height, 
		width, 
		blur_probability=0.5
	):
		"""Apply efficient batch data transformations.
		Args:
		  images_list: a list of image tensors.
		  height: the height of image.
		  width: the width of image.
		  blur_probability: the probaility to apply the blur operator.
		Returns:
		  Preprocessed feature list.
		"""
		def generate_selector(p, bsz):
			shape = [bsz, 1, 1, 1]
			selector = tf.cast(
				tf.less(tf.random_uniform(shape, 0, 1, dtype=tf.float32), p),
				tf.float32
				)
			return selector

		new_images_list = []
		for images in images_list:
			images_new = self.random_blur(images, height, width, p=1.)
			selector = generate_selector(blur_probability, tf.shape(images)[0])
			images = images_new * selector + images * (1 - selector)
			images = tf.clip_by_value(images, 0., 1.)
			new_images_list.append(images)

		return new_images_list

	def preprocess_for_train(self, image, height, width,
	                      color_distort=True, crop=True, flip=True):
		"""Preprocesses the given image for training.
		Args:
		  image: `Tensor` representing an image of arbitrary size.
		  height: Height of output image.
		  width: Width of output image.
		  color_distort: Whether to apply the color distortion.
		  crop: Whether to crop the image.
		  flip: Whether or not to flip left and right of an image.
		Returns:
		  A preprocessed image `Tensor`.
		"""
		if crop:
			image = self.random_crop_with_resize(image, height, width)
		if flip:
			image = tf.image.random_flip_left_right(image)
		if color_distort:
			image = self.random_color_jitter(image)
		image = tf.reshape(image, [height, width, 3])
		image = tf.clip_by_value(image, 0., 1.)
		return image

	def preprocess_for_eval(self, image, height, width, crop=True):
		"""Preprocesses the given image for evaluation.
		Args:
		  image: `Tensor` representing an image of arbitrary size.
		  height: Height of output image.
		  width: Width of output image.
		  crop: Whether or not to (center) crop the test images.
		Returns:
		  A preprocessed image `Tensor`.
		"""
		if crop:
			image = self.center_crop(image, height, width, crop_proportion=self.CROP_PROPORTION)
		image = tf.reshape(image, [height, width, 3])
		image = tf.clip_by_value(image, 0., 1.)
		return image

	def preprocess_image(self, image, height, width, is_training=False,
	                  color_distort=True, test_crop=True):
		"""Preprocesses the given image.
		Args:
		  image: `Tensor` representing an image of arbitrary size.
		  height: Height of output image.
		  width: Width of output image.
		  is_training: `bool` for whether the preprocessing is for training.
		  color_distort: whether to apply the color distortion.
		  test_crop: whether or not to extract a central crop of the images
		      (as for standard ImageNet evaluation) during the evaluation.
		Returns:
		  A preprocessed image `Tensor` of range [0, 1].
		"""
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		if is_training:
			return self.preprocess_for_train(image, height, width, color_distort)
		else:
			return self.preprocess_for_eval(image, height, width, test_crop)

class VGG16EmbeddingWrapper(BaseEmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'VGG16'
		self.model_description = "VGG16 loaded with pretrained imagenet weights from keras.applications."

	def load_model(self, input_shape=None, include_top=False, verbose=False):
		start_time = time.time()

		vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', input_shape=input_shape)


		if include_top:
			self.embedding_model = vgg16_model
		else:
			self.embedding_model = keras.Model(
			    inputs = vgg16_model.inputs, 
			    outputs = vgg16_model.layers[-2].output
			    )

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

	def preprocess_data(self, data, verbose=False):
		IMG_SIZE = (224, 224)
		start_time = time.time()

		preprocessed_data = keras.applications.vgg16.preprocess_input(data)
		preprocessed_data = tf.image.resize(preprocessed_data, IMG_SIZE)

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Data processed in {} seconds".format(total_time))

		return preprocessed_data

	def embed_images(self, images, batch_size=None, verbose=False):
		ndim = len(images.shape)

		if ndim == 3:
			images = tf.expand_dims(images, 0)

		start_time = time.time()

		if batch_size is None:
			embeddings = self.embedding_model.predict(images)
		else:
			num_images = images.shape[0]
			num_iter = int(np.ceil(num_images / batch_size))
			_embeddings = []

			if verbose:
				print("Embedding images...")
				widgets = [
					' [',
					progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
					'] ', 
			        progressbar.Bar('*'),' (', 
			        progressbar.ETA(), ') ', 
			        ]
				pbar = progressbar.ProgressBar(
					max_value=num_iter, widgets=widgets
					).start()

			for i in range(num_iter):
				if (i+1)*batch_size > num_images:
					image_batch = images[i*batch_size:]
				else:
					image_batch = images[i*batch_size:(i+1)*batch_size]

				embeddings_batch = self.embedding_model(image_batch)
				_embeddings.append(embeddings_batch.numpy().reshape(embeddings_batch.shape[0], -1))
				del embeddings_batch

				if verbose:
					pbar.update(i+1)

			embeddings = np.concatenate(_embeddings)
			del _embeddings

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Images embedded in {} seconds".format(total_time))

		return embeddings

class CLIPEmbeddingWrapper(BaseEmbeddingWrapper):
	def __init__(self):
		super().__init__()
		self.model_name = 'CLIP'
		self.model_description = "CLIP running with PyTorch, loaded from Sri's forked repository."

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.preprocess = None

	def load_model(self, encoder_model_name="ViT-B/32", verbose=False):
		# TODO: Should encoder_module_name be a constant?

		start_time = time.time()

		self.embedding_model, self.preprocess = clip.load(encoder_model_name, device=self.device)

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			# logging.log(logging.INFO, "Model loaded in {} seconds".format(total_time))
			print("Model loaded in {} seconds".format(total_time))

	def preprocess_data(self, data, verbose=False):
		if not isinstance(data, np.ndarray):
			data = data.numpy()
		if np.max(data) <= 1:
			data = data * 255
		data = data.astype(np.uint8)

		ndim = len(data.shape)
		if ndim == 3:
			data = np.expand_dims(data, 0)

		start_time = time.time()

		# transforms need input to be PIL Images
		preprocessed_data = torch.stack([self.preprocess(Image.fromarray(im, 'RGB')) for im in data])

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Data processed in {} seconds".format(total_time))

		return preprocessed_data

	def embed_images(self, images, batch_size=50, verbose=False):
		ndim = len(images.shape)
		if ndim == 3:
			images = torch.unsqueeze(images, 0)

		num_images = images.shape[0]
		num_iter = int(np.ceil(num_images/batch_size))

		start_time = time.time()

		_embeddings = []

		if verbose:
			print("Embedding images...")
			widgets = [
				' [', 
				progressbar.Timer(format= 'elapsed time: %(elapsed)s'), 
				'] ', 
				progressbar.Bar('*'),' (', 
				progressbar.ETA(), ') ', 
				]
			pbar = progressbar.ProgressBar(
				max_value=num_iter, widgets=widgets
				).start()

		for i in range(num_iter):
			if (i+1)*batch_size > num_images:
				image_batch = images[i*batch_size:]
			else:
				image_batch = images[i*batch_size:(i+1)*batch_size]

			embeddings_batch = self.embedding_model.encode_image(
				image_batch.to(self.device)
				).detach().cpu().numpy()# .ravel()
			_embeddings.append(embeddings_batch)
			del embeddings_batch

			if verbose:
				pbar.update(i+1)

		embeddings = np.concatenate(_embeddings)
		del _embeddings

		end_time = time.time()
		total_time = end_time - start_time

		if verbose:
			print("Images embedded in {} seconds".format(total_time))

		return embeddings