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

# load the GAN
net_g = Generator2()
net_g.load_state_dict(torch.load(generator_path))
net_g.to(device)
net_g.eval()

with torch.no_grad():
    # sentence interpolation

    # some captions
    first_caption = ['The man is standing on the beach', 'The woman is sitting behind a table',
                     'The boy has a tennis racket in his hands']
    second_caption = ['The man is holding a surfboard', 'The woman is eating a pizza',
                      'The boy is going to serve the ball']
    first_vector = [
        torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0) for caption in first_caption]
    second_vector = [
        torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
            -1).unsqueeze_(-1).unsqueeze_(0) for caption in second_caption]

    # interpolate them and plot
    a = np.linspace(0, 1, 5)
    plt.figure()
    for i in range(3):
        # noise is kept the same
        noise = get_noise_tensor(1).to(device)
        for j in range(5):
            # interpolate
            interpolated_vector = (1 - a[j]) * first_vector[i] + a[j] * second_vector[i]
            heatmap = np.array(net_g(noise, interpolated_vector).squeeze().tolist()) * 0.5 + 0.5
            plt.subplot(3, 5, i * 5 + j + 1)
            plot_heatmap(heatmap, skeleton, only_skeleton=True)
            plt.xticks([])
            plt.yticks([])
            if j == 0:
                plt.title(first_caption[i])
            elif j == 4:
                plt.title(second_caption[i])
            else:
                plt.title('')

    # some captions
    caption = []
    data = random.sample(dataset.dataset, 3)
    for i in range(3):
        caption.append(random.choice(data[i].get('caption')).get('caption'))
    caption_vector = [
        torch.tensor(get_caption_vector(text_model, c), dtype=torch.float32, device=device).unsqueeze_(-1).unsqueeze_(
            -1).unsqueeze_(0) for c in caption]

    # some noises
    first_noise = []
    second_noise = []
    for i in range(3):
        first_noise.append(get_noise_tensor(1).to(device))
        second_noise.append(get_noise_tensor(1).to(device))

    # interpolate noises and plot
    plt.figure()
    for i in range(3):
        for j in range(5):
            # interpolate
            interpolated_noise = (1 - a[j]) * first_noise[i] + a[j] * second_noise[i]
            heatmap = np.array(net_g(interpolated_noise, caption_vector[i]).squeeze().tolist()) * 0.5 + 0.5
            plt.subplot(3, 5, i * 5 + j + 1)
            plot_heatmap(heatmap, skeleton, only_skeleton=True)
            plt.xticks([])
            plt.yticks([])
            if j == 2:
                plt.title(caption[i])
            else:
                plt.title('')
