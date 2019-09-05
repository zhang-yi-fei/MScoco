from pycocotools.coco import COCO
import fastText
import random
import matplotlib.pyplot as plt

caption_path = '/media/data/yzhang2/coco/train/coco/annotations/captions_train2017.json'
text_model_path = '/media/data/yzhang2/wiki.en/wiki.en.bin'

coco = COCO(caption_path)
model = fastText.load_model('/media/data/yzhang2/wiki.en/wiki.en.bin')

# get all images' ids
image_ids = coco.getImgIds()

same_image_distance = []
different_image_distance = []

# random sample from captions
for i in range(10000):
    # select and compare two captions from the same image
    image_id = random.choice(image_ids)
    caption_ids = coco.getAnnIds(image_id)
    captions = coco.loadAnns(random.sample(caption_ids, 2))
    caption_0 = captions[0].get('caption').replace('\n', '')
    caption_1 = captions[1].get('caption').replace('\n', '')
    vector_0 = model.get_sentence_vector(caption_0)
    vector_1 = model.get_sentence_vector(caption_1)
    same_image_distance.append(sum((vector_0 - vector_1) ** 2) ** 0.5)

    # select and compare two captions from two different images
    image_id_sample = random.sample(image_ids, 2)
    caption_id_0 = random.choice(coco.getAnnIds(image_id_sample[0]))
    caption_id_1 = random.choice(coco.getAnnIds(image_id_sample[1]))
    captions = coco.loadAnns([caption_id_0, caption_id_1])
    caption_0 = captions[0].get('caption').replace('\n', '')
    caption_1 = captions[1].get('caption').replace('\n', '')
    vector_0 = model.get_sentence_vector(caption_0)
    vector_1 = model.get_sentence_vector(caption_1)
    different_image_distance.append(sum((vector_0 - vector_1) ** 2) ** 0.5)

# plot
plt.figure()
plt.hist(same_image_distance, 100, alpha=0.5)
plt.hist(different_image_distance, 100, alpha=0.5)
plt.legend(['same', 'different'])
plt.title('vector distances')
plt.xlabel('distance')
plt.ylabel('frequency')
plt.show()
