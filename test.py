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

# get single-person image ids
image_ids = get_one_person_image_ids(coco_keypoint)

# try a caption
image_id, caption = get_one_random_image_id_with_caption(coco_caption, image_ids)
plot_heatmap(get_heatmap2(coco_keypoint, image_id), skeleton, image_folder + coco_keypoint.loadImgs(image_id)[0]['file_name'], caption)
plot_fake(generator_path + '', device, skeleton)

# try the discriminator
heatmap = get_heatmap2(coco_keypoint, image_id)
discriminate_fake(heatmap, generator_path + '', discriminator_path + '', device, skeleton)
