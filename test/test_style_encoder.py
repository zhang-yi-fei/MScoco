from model import *
from pycocotools.coco import COCO
import fasttext

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read captions and keypoints from files
coco_caption = COCO(caption_path_val)
coco_keypoint = COCO(keypoint_path_val)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption, True)

gan_epoch = 2000
se_epoch = 5000

# load the GAN and the style encoder
net_g = Generator2()
net_s = Encoder(noise_size, False)
net_g.load_state_dict(torch.load(generator_path + '_' + f'{gan_epoch:05d}'))
net_s.load_state_dict(torch.load(style_encoder_path + '_' + f'{se_epoch:05d}'))
net_g.to(device)
net_s.to(device)
net_g.eval()
net_s.eval()

same_image_distance = []
random_image_distance = []
baseline_image_distance = []
different_image_distance = []

with torch.no_grad():
    for i in range(10000):
        # compare input noise and calculated style encoding of a generated image
        data = random.choice(dataset.dataset)
        caption = random.choice(data.get('caption')).get('caption')
        vector = torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        heatmap = net_g(noise, vector).detach()
        style = np.array(net_s(heatmap).squeeze().tolist())
        noise = np.array(noise.squeeze().tolist())
        same_image_distance.append(sum((style - noise) ** 2) ** 0.5)

        # compare random noise and calculated style encoding of a generated image
        data = random.choice(dataset.dataset)
        caption = random.choice(data.get('caption')).get('caption')
        vector = torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        heatmap = net_g(noise, vector).detach()
        style = np.array(net_s(heatmap).squeeze().tolist())
        noise = np.array(get_noise_tensor(1).squeeze().tolist())
        random_image_distance.append(sum((style - noise) ** 2) ** 0.5)

        # compare two random noises
        noise_1 = np.array(get_noise_tensor(1).squeeze().tolist())
        noise_2 = np.array(get_noise_tensor(1).squeeze().tolist())
        baseline_image_distance.append(sum((noise_1 - noise_2) ** 2) ** 0.5)

        # compare calculated style encodings of two different generated image
        data = random.sample(dataset.dataset, 2)
        caption_1 = random.choice(data[0].get('caption')).get('caption')
        vector_1 = torch.tensor(get_caption_vector(text_model, caption_1), dtype=torch.float32,
                                device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        heatmap_1 = net_g(noise, vector_1).detach()
        style_1 = np.array(net_s(heatmap_1).squeeze().tolist())
        caption_2 = random.choice(data[1].get('caption')).get('caption')
        vector_2 = torch.tensor(get_caption_vector(text_model, caption_2), dtype=torch.float32,
                                device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        heatmap_2 = net_g(noise, vector_2).detach()
        style_2 = np.array(net_s(heatmap_2).squeeze().tolist())
        different_image_distance.append(sum((style_1 - style_2) ** 2) ** 0.5)

# plot
plt.figure()
plt.hist(same_image_distance, 100, alpha=0.5)
plt.hist(random_image_distance, 100, alpha=0.5)
plt.legend(['same', 'random'])
plt.title('vector distances')
plt.xlabel('distance')
plt.ylabel('frequency')
plt.show()

plt.figure()
plt.hist(same_image_distance, 100, alpha=0.5)
plt.hist(baseline_image_distance, 100, alpha=0.5)
plt.legend(['same', 'baseline'])
plt.title('vector distances')
plt.xlabel('distance')
plt.ylabel('frequency')
plt.show()

plt.figure()
plt.hist(same_image_distance, 100, alpha=0.5)
plt.hist(different_image_distance, 100, alpha=0.5)
plt.legend(['same', 'different'])
plt.title('vector distances')
plt.xlabel('distance')
plt.ylabel('frequency')
plt.show()
