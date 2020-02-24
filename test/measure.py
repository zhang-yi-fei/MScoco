from model import *
from pycocotools.coco import COCO
import fasttext

generator_path = 'models/trained/model_generator'
discriminator_path = 'models/trained/model_uncon_discriminator'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read captions and keypoints from files
coco_caption = COCO(caption_path)
coco_keypoint = COCO(keypoint_path)
coco_caption_val = COCO(caption_path_val)
coco_keypoint_val = COCO(keypoint_path_val)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption, True)
dataset_val = HeatmapDataset(coco_keypoint_val, coco_caption_val, True)

# load the generator
net_g = Generator2()
net_g.load_state_dict(torch.load(generator_path))
net_g.to(device)
net_g.eval()

# load the discriminator
net_d = Discriminator()
net_d.load_state_dict(torch.load(discriminator_path))
net_d.to(device)
net_d.eval()

with torch.no_grad():
    real_feature = []
    fake_feature = []
    test_feature = []

    # classifier two-sample tests measure

    # two list of heatmaps: one real, one fake
    for data in dataset.dataset:
        real_feature.append(
            net_d.feature(torch.tensor(dataset.get_heatmap(data, False) * 2 - 1, dtype=torch.float32, device=device)))
        caption = random.choice(data.get('caption')).get('caption')
        vector = torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        fake_feature.append(net_d.feature(net_g(noise, vector)))

    # one-nearest-neighbor classification accuracy
    print('Classifier Two-sample Tests')
    print('accuracy: ' + str(one_nearest_neighbor(real_feature, fake_feature) * 100) + '%')

    # image retrieval performance measure

    distance_real = []
    distance_fake = []

    # one test list of heatmaps
    for data in dataset_val.dataset:
        test_feature.append(net_d.feature(
            torch.tensor(dataset_val.get_heatmap(data, False) * 2 - 1, dtype=torch.float32, device=device)))

    for i in range(len(real_feature)):
        distance_real.append(nearest_neighbor(real_feature[i], test_feature))
        distance_fake.append(nearest_neighbor(fake_feature[i], test_feature))

    print('Image Retrieval Performance')
    print('mean nearest neighbor distance (real):' + str(np.mean(distance_real)))
    print('mean nearest neighbor distance (fake):' + str(np.mean(distance_fake)))

    # plot
    plt.figure()
    plt.hist(distance_real, np.arange(1, 150, 1), alpha=0.5)
    plt.hist(distance_fake, np.arange(1, 150, 1), alpha=0.5)
    plt.legend(['real', 'fake'])
    plt.title('nearest neighbor distances')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    plt.show()
