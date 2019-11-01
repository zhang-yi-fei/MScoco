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
se_epoch = 2000

# load the GAN and the style encoder
net_g = Generator2()
net_s = Encoder(noise_size, False)
net_g.load_state_dict(torch.load(generator_path + '_' + f'{gan_epoch:05d}'))
net_s.load_state_dict(torch.load(style_encoder_path + '_' + f'{se_epoch:05d}'))
net_g.to(device)
net_s.to(device)
net_g.eval()
net_s.eval()

with torch.no_grad():
    # style transfer

    # some captions
    new_caption = []
    caption_vector = []
    new_heatmap = []
    data = random.sample(dataset.dataset, 6)
    for i in range(6):
        new_caption.append(random.choice(data[i].get('caption')).get('caption'))
    new_caption.append('The man is standing')
    new_caption.append('The woman is walking')
    new_caption.append('The boy is playing computer games')
    for caption in new_caption:
        caption_vector.append(
            torch.tensor(get_caption_vector(text_model, caption), dtype=torch.float32, device=device).unsqueeze_(
                -1).unsqueeze_(-1).unsqueeze_(0))

    plt.figure()
    plt.subplots_adjust(top=0.975, bottom=0.0, left=0.11, right=0.9, hspace=0.0, wspace=0.0)
    for i in range(6):
        new_heatmap.clear()

        # get a style encoding
        style_data = random.choice(dataset.dataset)
        style_heatmap = dataset.get_heatmap(style_data, False)
        style_heatmap_tensor = torch.tensor(style_heatmap * 2 - 1, dtype=torch.float32, device=device).unsqueeze_(0)
        style_vector = net_s(style_heatmap_tensor)

        # transfer the style to the new captions
        for vector in caption_vector:
            new_heatmap.append(np.array(net_g(style_vector, vector).squeeze().tolist()) * 0.5 + 0.5)

        # style heatmap
        plt.subplot(10, 10, i + 5)
        plot_heatmap(style_heatmap, skeleton)
        plt.xticks([])
        plt.yticks([])
        if i == 2:
            plt.title('style images')
        else:
            plt.title('')

        # style transferred heatmaps
        for j in range(9):
            plt.subplot(10, 10, j * 10 + 15 + i)
            plot_heatmap(new_heatmap[j], skeleton)
            plt.xticks([])
            plt.yticks([])
            plt.title('')
            if i == 0:
                plt.ylabel(new_caption[j], rotation='horizontal', ha='right')

    # sentence interpolation

    # some captions
    first_caption = ['The man is standing', 'The woman is sitting', 'The boy is sleeping in bed']
    second_caption = ['The man is running', 'The woman is standing', 'The boy is sitting on a chair']
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
            plot_heatmap(heatmap, skeleton)
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
            plot_heatmap(heatmap, skeleton)
            plt.xticks([])
            plt.yticks([])
            if j == 2:
                plt.title(caption[i])
            else:
                plt.title('')
