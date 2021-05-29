from . import datasets

class DatasetManager:
    
    def __init__(self):
        self.available_datasets = {
            'omniglot': datasets.OmniglotDataset(), 
            'lfw': datasets.LfwDataset(), 
            'clevr': datasets.ClevrDataset(), 
            'caltech_birds2011': datasets.CUBDataset(), 
            'imagenet_a': datasets.ImagenetADataset(), 
            'imagenet_r': datasets.ImagenetRDataset(),
            'mini_imagenet': datasets.MiniImagenetDataset(),
            'imagenet_sketch': datasets.ImagenetSketchDataset(),
            'imagenet_tiered': datasets.ImagenetTieredDataset(),
            'ucf_101': datasets.UCF101Dataset(),
            'indoor_scene_recognition': datasets.ISRDataset(),
            'fgcv_imaterialist': datasets.IMaterialistDataset(),
            'visda19': datasets.VisDA19Dataset(),
            # 'image_matching': datasets.ImageMatchingDataset(),
            'sun397': datasets.SUN397Dataset(),
            'cifar10': datasets.Cifar10Dataset(),
            'coco/2017': datasets.COCO2017Dataset(),
            'yale_faces': datasets.YaleFaces(),
            'utk_faces': datasets.UTKFaces(),
            'celeba_faces': datasets.CelebAFaces()
        }

        self.dataset = None

    def get_supported_datasets(self):
	    return self.available_datasets.keys()
    
    def get_class_names(self):
        
        if not self.dataset:
            raise Exception("Make sure load_dataset is run before get_class_names is run...")

        return self.dataset.get_class_names()
    
    def load_dataset(self, dataset_name, split='test', domain='clipart', img_width=224, img_height=224):
        if dataset_name in self.available_datasets.keys():

            self.dataset = self.available_datasets[dataset_name]

            if dataset_name == "visda19":
                data = self.dataset.load(split=split, domain=domain, img_width=img_width, img_height=img_height)
            else:
                data = self.dataset.load(split=split, img_width=img_width, img_height=img_height)
            return data
        else:
            print("Dataset not implemented")
            print("Please choose from the following:")
            print(self.get_supported_datasets())