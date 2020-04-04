from model import *
from pycocotools.coco import COCO
import fasttext

generator_path = 'trained/model_generator'

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

# load the generator
net_g = Generator2()
net_g.load_state_dict(torch.load(generator_path))
net_g.to(device)
net_g.eval()

with torch.no_grad():
    real_max_index = []
    gt_vector = []
    average_vector_distance = []

    nn_distance_10 = []
    gt_distance_10 = []
    average_distance_10 = []
    nn_vector_distance_10 = []

    # list of real heatmaps
    for data in dataset.dataset:
        real_max_index.append(heatmap_to_max_index(dataset.get_heatmap(data, False)))
        caption = random.choice(data.get('caption')).get('caption')
        vector = get_caption_vector(text_model, caption)
        gt_vector.append(vector)

    for i in range(len(gt_vector)):
        average_vector_distance.append(mean_vector_distance(gt_vector[i], gt_vector[0:i] + gt_vector[i + 1:]))

    average_vector_distance = np.array(average_vector_distance)

    for k in range(10):
        fake_max_index = []
        nn_distance = []
        gt_distance = []
        average_distance = []
        nn_vector_distance = []
        average_vector_distance = []
        for vector in gt_vector:
            vector = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(
                0)
            noise = get_noise_tensor(1).to(device)
            fake_max_index.append(heatmap_to_max_index(np.array(net_g(noise, vector).squeeze().tolist()) * 0.5 + 0.5))

        # calculate distances
        for i in range(len(fake_max_index)):
            nn_distance.append(nearest_neighbor(fake_max_index[i], real_max_index))
            gt_distance.append(heatmap_distance(fake_max_index[i], real_max_index[i]))
            average_distance.append(mean_distance(fake_max_index[i], real_max_index))

            nn_index = nearest_neighbor_index(fake_max_index[i], real_max_index)
            nn_vector_distance.append(np.sqrt(np.sum((gt_vector[i] - gt_vector[nn_index]) ** 2)))
            average_vector_distance.append(mean_vector_distance(gt_vector[i], gt_vector[0:i] + gt_vector[i + 1:]))

        nn_distance = np.array(nn_distance)
        gt_distance = np.array(gt_distance)
        average_distance = np.array(average_distance)
        nn_vector_distance = np.array(nn_vector_distance)
        average_vector_distance = np.array(average_vector_distance)

        nn_distance_10.append(nn_distance)
        gt_distance_10.append(gt_distance)
        average_distance_10.append(average_distance)
        nn_vector_distance_10.append(nn_vector_distance)

    nn_distance_10 = np.array(nn_distance_10)
    gt_distance_10 = np.array(gt_distance_10)
    average_distance_10 = np.array(average_distance_10)
    nn_vector_distance_10 = np.array(nn_vector_distance_10)

    nn_distance_10 = nn_distance_10.mean(axis=0)
    gt_distance_10 = gt_distance_10.mean(axis=0)
    average_distance_10 = average_distance_10.mean(axis=0)
    nn_vector_distance_10 = nn_vector_distance_10.mean(axis=0)

    print('Nearest neighbor is ground truth:' + str(np.sum(nn_distance == gt_distance) / len(fake_max_index)))
    print('Mean nearest neighbor distance:' + str(np.mean(nn_distance_10)))
    print('Mean ground truth distance:' + str(np.mean(gt_distance_10)))
    print('Mean average distance:' + str(np.mean(average_distance_10)))
    print('Mean nearest neighbor vector distance:' + str(np.mean(nn_vector_distance_10)))
    print('Mean average vector distance:' + str(np.mean(average_vector_distance)))

    # plot
    plt.figure()
    plt.hist(nn_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(gt_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(average_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.legend(['nearest neighbor', 'ground truth', 'mean'])
    plt.title('distances to fake poses')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    plt.show()

    plt.figure()
    plt.hist(nn_vector_distance_10, np.arange(0, 20, 0.4), alpha=0.5)
    plt.hist(average_vector_distance, np.arange(0, 20, 0.4), alpha=0.5)
    plt.legend(['nearest neighbor', 'mean'])
    plt.title('caption vector distances')
    plt.xlabel('vector distance')
    plt.ylabel('frequency')
    plt.show()
