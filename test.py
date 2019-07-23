from model import *
from pycocotools.coco import COCO

# import fastText

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

epoch = 6

# try the GAN
gan = GAN(generator_path + '_' + str(epoch), discriminator_path + '_' + str(epoch), device)
data = random.choice(dataset.dataset)
heatmap = get_heatmap(data.get('keypoint'))
file_name = data.get('image').get('file_name')
caption = random.choice(data.get('caption')).get('caption')

# plot a real heatmap
plot_heatmap(heatmap, skeleton, image_folder + file_name, caption)
plt.title('(real) score = ' + str(gan.discriminate(heatmap)))

# plot some fake ones
for i in range(24):
    fake = gan.generate()
    plot_heatmap(fake, skeleton)
    plt.title('(fake) score = ' + str(gan.discriminate(fake)))
