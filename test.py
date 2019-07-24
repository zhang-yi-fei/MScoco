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

epoch = 16

# try the GAN
gan = GAN(generator_path + '_' + str(epoch), discriminator_path + '_' + str(epoch), device)
data = random.choice(dataset.dataset)
heatmap = get_heatmap(data.get('keypoint'))
file_name = data.get('image').get('file_name')
caption = random.choice(data.get('caption')).get('caption')

# plot a real heatmap
plot_heatmap(heatmap, skeleton, image_folder + file_name, caption)
plt.title('(real) score = ' + str(1 / (1 + np.exp(-gan.discriminate(heatmap)))))

# plot some fake ones
for i in range(24):
    fake = gan.generate()
    plot_heatmap(fake, skeleton)
    plt.title('(fake) score = ' + str(1 / (1 + np.exp(-gan.discriminate(fake)))))

# pick one fake
noise = torch.randn(noise_size, dtype=torch.float32)
fake = gan.generate(noise)

# find the nearest neighbor in the training dataset
nearest = dataset.dataset[0]
distance = float('inf')
for data in dataset.dataset:
    if np.linalg.norm(fake - get_heatmap(data.get('keypoint'))) < distance:
        distance = np.linalg.norm(fake - get_heatmap(data.get('keypoint')))
        nearest = data

# plot the fake and the nearest neighbor
plot_heatmap(fake, skeleton)
plt.title('fake score = ' + str(1 / (1 + np.exp(-gan.discriminate(fake)))))
plot_heatmap(get_heatmap(nearest.get('keypoint')), skeleton)
plt.title('nearest neighbor score = ' + str(1 / (1 + np.exp(-gan.discriminate(get_heatmap(nearest.get('keypoint')))))))
