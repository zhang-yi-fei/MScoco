from model import *
from pycocotools.coco import COCO
import fasttext

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

gan_epoch = 2000
ce_epoch = 2000

# load the GAN and the content encoder
net_g = Generator2()
net_c = Encoder(sentence_vector_size, True)
net_g.load_state_dict(torch.load(generator_path + '_' + f'{gan_epoch:05d}'))
net_c.load_state_dict(torch.load(content_encoder_path + '_' + f'{ce_epoch:05d}'))
net_g.to(device)
net_c.to(device)
net_g.eval()
net_c.eval()

with torch.no_grad():
    real_max_index = []
    fake_max_index = []
    test_max_index = []

    # classifier two-sample tests measure

    # two list of heatmaps: one real, one fake
    for data in dataset.dataset:
        real_max_index.append(heatmap_to_max_index(dataset.get_heatmap(data, False)))
        caption = random.choice(data.get('caption')).get('caption')
        vector = torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        fake_max_index.append(heatmap_to_max_index(np.array(net_g(noise, vector).squeeze().tolist()) * 0.5 + 0.5))

    # one-nearest-neighbor classification accuracy
    print('Classifier Two-sample Tests')
    print('accuracy: ' + str(one_nearest_neighbor(real_max_index, fake_max_index) * 100) + '%')

    # image retrieval performance measure

    distance_real = []
    distance_fake = []

    # one test list of heatmaps
    for data in dataset_val.dataset:
        test_max_index.append(heatmap_to_max_index(dataset_val.get_heatmap(data, False)))

    for i in range(len(real_max_index)):
        distance_real.append(nearest_neighbor(real_max_index[i], test_max_index))
        distance_fake.append(nearest_neighbor(fake_max_index[i], test_max_index))

    print('Image Retrieval Performance')
    print('mean nearest neighbor distance (real):' + str(np.mean(distance_real)))
    print('mean nearest neighbor distance (fake):' + str(np.mean(distance_fake)))

    # plot
    plt.figure()
    plt.hist(distance_real, 100, alpha=0.5)
    plt.hist(distance_fake, 100, alpha=0.5)
    plt.legend(['real', 'fake'])
    plt.title('nearest neighbor distances')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    plt.show()

    # classification performance measure

    error_real = []
    error_fake = []
    criterion = nn.MSELoss()

    # one test list of heatmaps
    for data in dataset_val.dataset:
        real_heatmap = torch.tensor(dataset_val.get_heatmap(data, False) * 2 - 1, dtype=torch.float32,
                                    device=device).unsqueeze_(0)
        real_caption = random.choice(data.get('caption')).get('caption')
        real_vector = torch.tensor(get_caption_vector(text_model, real_caption), dtype=torch.float32,
                                   device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(0)
        fake_caption = random.choice(data.get('caption')).get('caption')
        fake_vector = torch.tensor(get_caption_vector(text_model, fake_caption), dtype=torch.float32,
                                   device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(0)
        noise = get_noise_tensor(1).to(device)
        fake_heatmap = net_g(noise, fake_vector).detach()
        real_content = net_c(real_heatmap).detach()
        fake_content = net_c(fake_heatmap).detach()
        error_real.append(criterion(real_vector, real_content).item())
        error_fake.append(criterion(fake_vector, fake_content).item())

    print('Classification Performance')
    print('mean error (real):' + str(np.mean(error_real)))
    print('mean error (fake):' + str(np.mean(error_fake)))

    # plot
    plt.figure()
    plt.hist(error_real, 100, alpha=0.5)
    plt.hist(error_fake, 100, alpha=0.5)
    plt.legend(['real', 'fake'])
    plt.title('sentence/content vector errors')
    plt.xlabel('error')
    plt.ylabel('frequency')
    plt.show()
