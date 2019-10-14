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

gan_epoch = 1000
se_epoch = 100

# load the GAN and the style encoder
net_g = Generator2()
net_d = Discriminator2()
net_s = StyleEncoder()
net_g.load_state_dict(torch.load(generator_path + '_' + f'{gan_epoch:05d}'))
net_d.load_state_dict(torch.load(discriminator_path + '_' + f'{gan_epoch:05d}'))
net_s.load_state_dict(torch.load(style_encoder_path + '_' + f'{se_epoch:05d}'))
net_g.to(device)
net_d.to(device)
net_s.to(device)
net_g.eval()
net_d.eval()
net_s.eval()

# style transfer

# some captions
new_caption = [[]] * 8
caption_vector = [[]] * 8
new_heatmap = [[]] * 8
for i in range(5):
    new_caption[i] = random.choice(random.choice(dataset.dataset).get('caption')).get('caption')
new_caption[5] = 'The man is standing'
new_caption[6] = 'The woman is walking'
new_caption[7] = 'The boy is playing computer games'
for i in range(8):
    caption_vector[i] = torch.tensor(get_caption_vector(text_model, new_caption[i]), dtype=torch.float32,
                                     device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(0)

for j in range(5):
    # get a style encoding
    style_data = random.choice(dataset.dataset)
    style_caption = random.choice(style_data.get('caption')).get('caption')
    style_heatmap = dataset.get_heatmap(style_data, False)
    style_heatmap_tensor = torch.tensor(style_heatmap * 2 - 1, dtype=torch.float32, device=device).unsqueeze_(0)
    style_vector = net_s(style_heatmap_tensor)

    # transfer the style to the new captions
    for i in range(8):
        new_heatmap[i] = np.array(net_g(style_vector, caption_vector[i]).squeeze().tolist()) * 0.5 + 0.5

    # plot the heatmaps
    plt.figure()

    # style heatmap
    plt.subplot(3, 3, 1)
    plot_heatmap(style_heatmap, skeleton)
    plt.title(style_caption[0:30] + '\n' + style_caption[30:])
    plt.xticks([])
    plt.yticks([])

    # style transfered heatmaps
    for i in range(8):
        plt.subplot(3, 3, i + 2)
        plot_heatmap(new_heatmap[i], skeleton)
        plt.title(new_caption[i][0:30] + '\n' + new_caption[i][30:])
        plt.xticks([])
        plt.yticks([])
