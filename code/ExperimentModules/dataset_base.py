import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers

class Dataset:
    def __init__(self):
        self.name = None
        self.img_height = None
        self.img_width = None
        self.data_generator = None
    
    def get_class_names(self):
	    return self.info.features["label"].names
    
    def get_info(self):
        return self.info

    def apply_transform(self, func, x):
        x['image'] = func(x['image'])
        return x

    def preprocess(self, ds, img_height, img_width):
        transform = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(img_height, img_width),
            #layers.experimental.preprocessing.Rescaling(1./255)
        ])
        resized_ds = ds.map(lambda x: self.apply_transform(transform, x))
        return resized_ds

    def load(self, split='test', img_height=224, img_width=224):
        ds, self.info = tfds.load(self.name, split=split, shuffle_files=False, with_info=True)
        ds = self.preprocess(ds, img_height, img_width)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds_np = tfds.as_numpy(ds)
        self.data_generator = ds_np
        return ds_np 

    def _download(self):
	    pass

    def _uncompress(self):
	    pass