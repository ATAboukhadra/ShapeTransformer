import torch
import torchvision
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datapipes.utils.collation_functions import collate_sequences_as_dicts
from utils import get_keypoints

batch_size = 1
num_workers = 1
sliding_window_size = 9
num_seqs = 4

root = '/ds-av/public_datasets/arctic/td/sequential_resized_egocentric/'
objects_root = 'dataset/arctic_objects'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mode = 'allocentric' if 'allocentric' in root else 'all'
split = 'train'
pipeline, num_samples, decoder, factory = create_pipe(root, objects_root, split, mode, 'cpu', sliding_window_size, num_seqs)
loader = torch.utils.data.DataLoader(pipeline, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_sequences_as_dicts)

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_keypoints=21, num_classes=24).to(device)
model.load_state_dict(torch.load('/data/checkpoints/resnet50_egocentric/rcnn.pth'))
model.eval()

for i, (_, data) in tqdm(enumerate(loader), total=num_samples // batch_size):
    # if (split != 'test' and data['obj_pose'][0][0][0] < 0.5) or i % 10 != 0:
        # continue
    if i < 1000:
        continue
    print(data['key'])
    plt.figure(figsize=(15, 15))
    images = [sample[0].to(device) for sample in data['rgb']]
    outputs = model(images)
    if outputs[0]['keypoints'].shape[0] < 2:
        continue
    
    keypoints, _, _ = get_keypoints(outputs, 0)

    img = data['rgb'][0][0].cpu().numpy().transpose(1, 2, 0)
    img = np.ascontiguousarray(img * 255, np.uint8)

    plt.imshow(img)
    colors = ['peachpuff', 'lightblue', 'lightgreen', 'gold']
    for i in range(4):
        if keypoints[i] is None:
            continue

        plt.scatter(keypoints[i].detach().cpu().numpy()[:, 0], keypoints[i].detach().cpu().numpy()[:, 1], s=1, c=colors[i])

    plt.axis('off')
    plt.show()    