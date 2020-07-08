from model import *
from pycocotools.coco import COCO
import fasttext

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# read captions and keypoints from files
coco_caption = COCO(caption_path_val)
coco_keypoint = COCO(keypoint_path_val)
coco_caption_train = COCO(caption_path)
coco_keypoint_train = COCO(keypoint_path)

# keypoint connections (skeleton) from annotation file
skeleton = np.array(coco_keypoint.loadCats(coco_keypoint.getCatIds())[0]['skeleton']) - 1

# load text encoding model
text_model = fasttext.load_model(text_model_path)

# get single-person image dataset
dataset = HeatmapDataset(coco_keypoint, coco_caption, True, for_regression=True)
dataset_train = HeatmapDataset(coco_keypoint_train, coco_caption_train, True, for_regression=True)

real_max_index = []
real_max_index_train = []
gt_vector = []
train_vector = []
text_nn_index = []
average_vector_distance = []

# list of real heatmaps
for data in dataset_train.dataset:
    keypoint = data.get('keypoint')
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    x, y, v = get_coordinates(x0, y0, w, h, keypoint)
    real_max_index_train.append(coordinates_to_max_index(x, y, v))
    for i in range(5):
        caption = data.get('caption')[i].get('caption')
        vector = get_caption_vector(text_model, caption)
        train_vector.append(vector)
for data in dataset.dataset:
    keypoint = data.get('keypoint')
    x0, y0, w, h = tuple(keypoint.get('bbox'))
    x, y, v = get_coordinates(x0, y0, w, h, keypoint)
    real_max_index.append(coordinates_to_max_index(x, y, v))
    caption = random.choice(data.get('caption')).get('caption')
    vector = get_caption_vector(text_model, caption)
    text_nn_index.append(round(vector_nearest_neighbor_index(vector, train_vector) / 5 - 0.4))
    gt_vector.append(vector)

for i in range(len(gt_vector)):
    average_vector_distance.append(mean_vector_distance(gt_vector[i], gt_vector[0:i] + gt_vector[i + 1:]))

average_vector_distance = np.array(average_vector_distance)

generator_path = 'trained/model_regression_generator'

nn_distance_10 = []
nn_distance_train_10 = []
text_nn_distance_10 = []
gt_distance_10 = []
average_distance_10 = []
average_distance_train_10 = []
nn_vector_distance_10 = []

# load the generator
net_g = Generator_R()
net_g.load_state_dict(torch.load(generator_path))
net_g.to(device)
net_g.eval()

with torch.no_grad():
    for k in range(10):
        fake_max_index = []
        nn_distance = []
        nn_distance_train = []
        text_nn_distance = []
        gt_distance = []
        average_distance = []
        average_distance_train = []
        nn_vector_distance = []
        for vector in gt_vector:
            vector = torch.tensor(vector, dtype=torch.float32, device=device).unsqueeze_(0)
            noise = get_noise_tensor(1).to(device).squeeze_().unsqueeze_(0)
            x, y, v = result_to_coordinates(np.array(net_g(noise, vector).squeeze().tolist()))
            fake_max_index.append(coordinates_to_max_index(x, y, v))

        # calculate distances
        for i in range(len(fake_max_index)):
            nn_distance.append(nearest_neighbor(fake_max_index[i], real_max_index))
            nn_distance_train.append(nearest_neighbor(fake_max_index[i], real_max_index_train))
            text_nn_distance.append(heatmap_distance(fake_max_index[i], real_max_index_train[text_nn_index[i]]))
            gt_distance.append(heatmap_distance(fake_max_index[i], real_max_index[i]))
            average_distance.append(mean_distance(fake_max_index[i], real_max_index))
            average_distance_train.append(mean_distance(fake_max_index[i], real_max_index_train))

            nn_index = nearest_neighbor_index(fake_max_index[i], real_max_index)
            nn_vector_distance.append(np.sqrt(np.sum((gt_vector[i] - gt_vector[nn_index]) ** 2)))

        nn_distance = np.array(nn_distance)
        nn_distance_train = np.array(nn_distance_train)
        text_nn_distance = np.array(text_nn_distance)
        gt_distance = np.array(gt_distance)
        average_distance = np.array(average_distance)
        average_distance_train = np.array(average_distance_train)
        nn_vector_distance = np.array(nn_vector_distance)

        nn_distance_10.append(nn_distance)
        nn_distance_train_10.append(nn_distance_train)
        text_nn_distance_10.append(text_nn_distance)
        gt_distance_10.append(gt_distance)
        average_distance_10.append(average_distance)
        average_distance_train_10.append(average_distance_train)
        nn_vector_distance_10.append(nn_vector_distance)

    nn_distance_10 = np.array(nn_distance_10)
    nn_distance_train_10 = np.array(nn_distance_train_10)
    text_nn_distance_10 = np.array(text_nn_distance_10)
    gt_distance_10 = np.array(gt_distance_10)
    average_distance_10 = np.array(average_distance_10)
    average_distance_train_10 = np.array(average_distance_train_10)
    nn_vector_distance_10 = np.array(nn_vector_distance_10)

    nn_distance_10 = nn_distance_10.mean(axis=0)
    nn_distance_train_10 = nn_distance_train_10.mean(axis=0)
    text_nn_distance_10 = text_nn_distance_10.mean(axis=0)
    gt_distance_10 = gt_distance_10.mean(axis=0)
    average_distance_10 = average_distance_10.mean(axis=0)
    average_distance_train_10 = average_distance_train_10.mean(axis=0)
    nn_vector_distance_10 = nn_vector_distance_10.mean(axis=0)

    print('Regression')
    print('Nearest neighbor is ground truth:' + str(np.sum(nn_distance == gt_distance) / len(fake_max_index)))
    print('Mean nearest neighbor distance:' + str(np.mean(nn_distance_10)))
    print('Mean nearest neighbor distance (in training set):' + str(np.mean(nn_distance_train_10)))
    print('Mean ground truth distance:' + str(np.mean(gt_distance_10)))
    print('Mean text nn distance (in training set):' + str(np.mean(text_nn_distance_10)))
    print('Mean average distance:' + str(np.mean(average_distance_10)))
    print('Mean average distance (in training set):' + str(np.mean(average_distance_train_10)))
    print('Mean nearest neighbor vector distance:' + str(np.mean(nn_vector_distance_10)))
    print('Mean average vector distance:' + str(np.mean(average_vector_distance)))

    # plot
    plt.figure()
    plt.hist(nn_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(nn_distance_train_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(gt_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(text_nn_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(average_distance_10, np.arange(0, 700, 10), alpha=0.5)
    plt.hist(average_distance_train_10, np.arange(0, 700, 10), alpha=0.5)
    plt.legend(['NN in validation set', 'NN in training set', 'ground truth', 'text NN in training set',
                'mean in training set', 'mean in validation set'])
    plt.title('distances to fake poses')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    plt.show()
    plt.savefig('conditional_measure_pose' + '_regression')

    plt.figure()
    plt.hist(nn_vector_distance_10, np.arange(0, 20, 0.4), alpha=0.5)
    plt.hist(average_vector_distance, np.arange(0, 20, 0.4), alpha=0.5)
    plt.legend(['nearest neighbor', 'mean'])
    plt.title('caption vector distances')
    plt.xlabel('vector distance')
    plt.ylabel('frequency')
    plt.show()
    plt.savefig('conditional_measure_text' + '_regression')
