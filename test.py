from model import *
from pycocotools.coco import COCO
import fasttext

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

epoch = 70

# load the GAN
net_g = Generator2()
net_d = Discriminator2()
net_g.load_state_dict(torch.load(generator_path + '_' + f'{epoch:05d}'))
net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{epoch:05d}'))
net_g.to(device)
net_d.to(device)
net_g.eval()
net_d.eval()

# pick a random real sample
data = random.choice(dataset.dataset)
heatmap = dataset.get_heatmap(data, augment=False)
heatmap_tensor = torch.tensor(heatmap * 2 - 1, dtype=torch.float32, device=device).unsqueeze_(0)
file_name = data.get('image').get('file_name')
caption = random.choice(data.get('caption')).get('caption')
caption_vector = torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
    -1).unsqueeze_(-1).unsqueeze_(0)

# plot a real heatmap
plt.figure()
plot_heatmap(heatmap, skeleton, image_folder + file_name, caption)
plt.title('(real) score = ' + str(net_d(heatmap_tensor, caption_vector).item()))
plt.show()

# plot some fake ones
plt.figure()
for i in range(25):
    noise = get_noise_tensor(1).to(device)
    fake_tensor = net_g(noise, caption_vector)
    fake = np.array(fake_tensor.squeeze().tolist()) * 0.5 + 0.5
    plt.subplot(5, 5, i + 1)
    plot_heatmap(fake, skeleton)
    plt.title('(fake) score = ' + str(net_d(fake_tensor, caption_vector).item()))
    plt.show()
