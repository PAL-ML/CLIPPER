from .dataset_manager import DatasetManager

dm = DatasetManager()
data_generator = dm.load_dataset('ucf_101', split="val")

for image, label in data_generator:
    print(image.shape)
    print(label.shape)
    break

