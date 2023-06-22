from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py'
checkpoint_file = 'deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth'

# config_file = 'deeplabv3plus_r101-d8_4xb4-80k_pascal-context-59-480x480.py'
# checkpoint_file = 'deeplabv3plus_r101-d8_480x480_80k_pascal_context_59_20210416_111127-7ca0331d.pth'

# config_file = 'deeplabv3plus_r50-d8_4xb4-80k_pascal-context-480x480.py'
# checkpoint_file = 'deeplabv3plus_r101-d8_480x480_80k_pascal_context_20200911_155322-145d3ee8.pth'

# config_file = 'segformer_mit-b5_8xb2-160k_ade20k-640x640.py'
# checkpoint_file = 'segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'



# config_file = 'upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k.py'


# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
# img = '/home2/HO3D_v3/train/ShSu13/rgb/0002.jpg'
# img = '/home2/HO3D_v3/train/SS2/rgb/0885.jpg'
img = '/home2/HO3D_v3/train/ABF10/rgb/0882.jpg'
# img = '/home2/HO3D_v3/train/GPMF11/rgb/0022.jpg'
# img = 'arctic.png'

result = inference_model(model, img)
print(result.pred_sem_seg.data.shape)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
# show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    show_result_pyplot(model, result, wait_time=1)