import torchvision
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from dataset.arctic_pipeline import create_pipe
from tqdm import tqdm
from utils import AverageMeter
import os
from datapipes.utils.collation_functions import collate_sequences_as_dicts

root = '/ds-av/public_datasets/arctic/td/sequential_resized/'
objects_root = 'dataset/arctic_objects'
output_folder = '/checkpoints/arctic_obj_rcnn/'
if not os.path.exists(output_folder): os.mkdir(output_folder)

batch_size = 4
num_workers = 4
sliding_window_size = 1
epochs = 5
num_seqs = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(num_keypoints=30, num_classes=22).to(device)

train_pipeline, num_samples, decoder, factory = create_pipe(root, objects_root, 'train', 'cpu', sliding_window_size, num_seqs)
trainloader = torch.utils.data.DataLoader(train_pipeline, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_sequences_as_dicts)
val_pipeline, val_count, _, _ = create_pipe(root, objects_root, 'val', torch.device('cpu'), sliding_window_size, num_seqs, factory=factory, arctic_decoder=decoder)
valloader = torch.utils.data.DataLoader(val_pipeline, batch_size=batch_size, num_workers=num_workers, pin_memory=False, collate_fn=collate_sequences_as_dicts)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for e in range(epochs):

    losses_counters = {'loss_classifier': AverageMeter(), 'loss_box_reg': AverageMeter(), 'loss_keypoint': AverageMeter(), 'loss_objectness': AverageMeter(), 'loss_rpn_box_reg': AverageMeter()}
    for i, (_, data) in tqdm(enumerate(trainloader), total=num_samples // batch_size):
        images = [sample[0].to(device) for sample in data['rgb']]
        keys = ['boxes', 'labels', 'keypoints']
        targets = [{k: data[k][i][0].to(device) for k in keys} for i in range(len(images))]
        losses = model(images, targets)

        loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for k in losses.keys():
            losses_counters[k].update(losses[k].item(), len(images))

        if (i+1) % 1000 == 0:
            print(i+1, [(k, round(losses_counters[k].avg, 2)) for k in losses.keys()])
            torch.save(model.state_dict(), f'{output_folder}keypointrcnn_resnet50_fpn_{e}.pth')
            for k in losses.keys():
                losses_counters[k].reset()
    
    torch.save(model.state_dict(), f'{output_folder}keypointrcnn_resnet50_fpn_{e}.pth')

    losses_counters = {'loss_classifier': AverageMeter(), 'loss_box_reg': AverageMeter(), 'loss_keypoint': AverageMeter(), 'loss_objectness': AverageMeter(), 'loss_rpn_box_reg': AverageMeter()}
    with torch.no_grad():
        for i, data in tqdm(enumerate(valloader), total=val_count // batch_size):
            images = [sample[0].to(device) for sample in data['rgb']]
            keys = ['boxes', 'labels', 'keypoints']
            targets = [{k: data[k][i][0].to(device) for k in keys} for i in range(len(images))]
            losses = model(images, targets)
            for k in losses.keys():
                losses_counters[k].update(losses[k].item(), len(images))

        print('Validation', [(k, round(losses_counters[k].avg, 2)) for k in losses.keys()])


            

