from pycocotools.coco import COCO
import torch.utils.data
import fastText

dataDir = '/media/data/yzhang2/coco/val/coco'
dataType = 'val2017'
annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

modelFile = '/media/data/yzhang2/wiki.en/wiki.en.bin'

coco = COCO(annFile)
anns = coco.loadAnns(coco.getAnnIds())

annLoader = torch.utils.data.DataLoader(anns, batch_size=1, shuffle=False)

model = fastText.load_model(modelFile)

for i, ann in enumerate(annLoader, 0):
    caption = ann['caption'][0].replace('\n', '')
    print(model.get_sentence_vector(caption))
