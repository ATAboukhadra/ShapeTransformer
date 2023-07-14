import torchvision
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from dataset.arctic_pipeline import create_pipe, temporal_batching
from tqdm import tqdm

root = '/ds-av/public_datasets/arctic/td/p1_sequential_nocropped/'
objects_root = 'dataset/arctic_objects'
batch_size = 4
num_workers = 4
sliding_window_size = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_keypoints=30, num_classes=22).to(device)

train_pipeline, num_samples, decoder, factory = create_pipe(root, objects_root, 'train', 'cpu', sliding_window_size)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=0, collate_fn=temporal_batching)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for i, data in tqdm(enumerate(trainloader), total=num_samples // batch_size):
    images = [sample[0].to(device) for sample in data['rgb']]
    keys = ['boxes', 'labels', 'keypoints']
    targets = [{k: data[k][i][0].to(device) for k in keys} for i in range(len(images))]
    losses = model(images, targets)

    loss = sum(loss for loss in losses.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
        print(i+1, [(k, round(losses[k].item(), 2)) for k in losses.keys()])