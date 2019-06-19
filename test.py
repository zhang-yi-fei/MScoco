from main import *

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fastText.load_model(text_model_path)

# get single-person image ids
image_ids = get_one_person_image_ids(coco_keypoint)

# try a caption
image_id, caption = get_one_random_image_id_with_caption(coco_caption, image_ids)
plot_caption(caption, text_model, generator_path, device, skeleton)

# try the discriminator
heatmap = get_heatmap(coco_keypoint, image_id)
discriminate(normalized_random_view_tensor(heatmap), caption, text_model, generator_path, discriminator_path, device,
             skeleton)
