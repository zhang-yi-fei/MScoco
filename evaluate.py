from model import *
from pycocotools.coco import COCO
import fasttext

# import fastText

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption, True)

epoch = 2000

# try the GAN
gan = GAN2(generator_path + '_' + f'{epoch:05d}', discriminator_path + '_' + f'{epoch:05d}', text_model, device)
