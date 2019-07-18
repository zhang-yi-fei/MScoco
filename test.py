from model import *
from pycocotools.coco import COCO
# import fastText

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator_path = 'generator'
discriminator_path = 'discriminator'

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
# text_model = fastText.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption)

epoch = 8

# try the generator
data = random.choice(dataset.dataset)
plot_heatmap(get_heatmap2(data.get('keypoint')), skeleton, image_folder + data.get('image').get('file_name'),
             random.choice(data.get('caption')).get('caption'))
plot_fake(generator_path + '_' + str(epoch), device, skeleton)

# try the discriminator
heatmap = get_heatmap2(data.get('keypoint'))
discriminate_fake(heatmap, generator_path + '_' + str(epoch), discriminator_path + '_' + str(epoch), device, skeleton)
