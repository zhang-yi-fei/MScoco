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

ce_epoch = 5000

# load the content encoder
net_c = Encoder(sentence_vector_size, True)
net_c.load_state_dict(torch.load(content_encoder_path + '_' + f'{ce_epoch:05d}'))
net_c.to(device)
net_c.eval()

same_image_distance = []
random_image_distance = []
baseline_image_distance = []
different_image_distance = []

with torch.no_grad():
    for i in range(10000):
        # compare caption's encoding and calculated content encoding of an image
        data = random.choice(dataset.dataset)
        caption = random.choice(data.get('caption')).get('caption')
        vector = get_caption_vector(text_model, caption)
        heatmap = torch.tensor(dataset.get_heatmap(data, False) * 2 - 1, dtype=torch.float32, device=device).unsqueeze_(
            0)
        content = np.array(net_c(heatmap).squeeze().tolist())
        same_image_distance.append(sum((vector - content) ** 2) ** 0.5)

        # compare caption's encoding of an image and calculated content encoding of a different image
        data = random.sample(dataset.dataset, 2)
        caption = random.choice(data[0].get('caption')).get('caption')
        vector = get_caption_vector(text_model, caption)
        heatmap = torch.tensor(dataset.get_heatmap(data[1], False) * 2 - 1, dtype=torch.float32,
                               device=device).unsqueeze_(0)
        content = np.array(net_c(heatmap).squeeze().tolist())
        random_image_distance.append(sum((vector - content) ** 2) ** 0.5)

        # compare captions' encodings of two different images
        data = random.sample(dataset.dataset, 2)
        caption_1 = random.choice(data[0].get('caption')).get('caption')
        vector_1 = get_caption_vector(text_model, caption_1)
        caption_2 = random.choice(data[1].get('caption')).get('caption')
        vector_2 = get_caption_vector(text_model, caption_2)
        baseline_image_distance.append(sum((vector_1 - vector_2) ** 2) ** 0.5)

        # compare calculated content encodings of two different image
        data = random.sample(dataset.dataset, 2)
        heatmap_1 = torch.tensor(dataset.get_heatmap(data[0], False) * 2 - 1, dtype=torch.float32,
                                 device=device).unsqueeze_(0)
        content_1 = np.array(net_c(heatmap_1).squeeze().tolist())
        heatmap_2 = torch.tensor(dataset.get_heatmap(data[1], False) * 2 - 1, dtype=torch.float32,
                                 device=device).unsqueeze_(0)
        content_2 = np.array(net_c(heatmap_2).squeeze().tolist())
        different_image_distance.append(sum((content_1 - content_2) ** 2) ** 0.5)

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
