from model import *
from pycocotools.coco import COCO
import fastText

# import fastText

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fastText.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption, True)

epoch = 1500

# try the GAN
gan = GAN2(generator_path + '_' + f'{epoch:05d}', discriminator_path + '_' + f'{epoch:05d}', text_model, device)
data = random.choice(dataset.dataset)
heatmap = get_heatmap(data.get('keypoint'), augment=False)
file_name = data.get('image').get('file_name')
caption = random.choice(data.get('caption')).get('caption')

# plot a real heatmap
plt.figure()
plot_heatmap(heatmap, skeleton, image_folder + file_name, caption)
plt.title('(real) score = ' + str(gan.discriminate(heatmap, caption)))
plt.show()

# plot some fake ones
plt.figure()
for i in range(25):
    fake = gan.generate(caption)
    plt.subplot(5, 5, i + 1)
    plot_heatmap(fake, skeleton)
    plt.title('(fake) score = ' + str(gan.discriminate(fake, caption)))
    plt.show()
