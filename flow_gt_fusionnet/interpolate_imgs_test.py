import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
import torch.nn.functional as F

half_tensor = True

from model.RIFE import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
if half_tensor:
    from model.RIFE_half import Model
model = Model()
model.load_model('train_log', -1)

print("Loaded model")
model.eval()
model.device()


def to_tensor(img):
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().cuda() / 255.
    if half_tensor:
        return x.half()
    return x


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def load_flow(flofile, tenImg):
    _, _, imgh, imgw = tenImg.shape
    flo = readFlow(flofile)  # h, w, c
    flo_h, flo_w, _ = flo.shape
    if half_tensor:
        flo = torch.from_numpy(flo.transpose(2, 0, 1)).unsqueeze(0).half().cuda()
    else:
        flo = torch.from_numpy(flo.transpose(2, 0, 1)).unsqueeze(0).float().cuda()

    # scale flow
    flo = F.interpolate(flo, size=(imgh, imgw), mode='bilinear', align_corners=False)
    flo[:, 1:] = flo[:, 1:] * (imgh / flo_h)
    flo[:, 0:1] = flo[:, 0:1] * (imgw / flo_w)

    return flo


def load_image(imfile):
    img = cv2.imread(imfile)
    H, W, _ = img.shape
    ori_h = H
    ori_w = W
    while H % 64 != 0:
        H += 1
    while W % 64 != 0:
        W += 1
    img = cv2.resize(img, (W, H))
    img = to_tensor(img)
    return [ori_h, ori_w, img]


imgs = [os.path.join('flow_dataset/images', f) for f in os.listdir('flow_dataset/images')]
flo_forward = [os.path.join('flow_dataset/flow_forwards', f) for f in os.listdir('flow_dataset/flow_forwards')]
flo_backward = [os.path.join('flow_dataset/flow_backwards', f) for f in os.listdir('flow_dataset/flow_backwards')]

pbar = tqdm(total=len(imgs))
i0, flo01, flo10 = imgs[0], flo_forward[0], flo_backward[0]
cnt = 0
for i in range(1, len(imgs)):
    i1 = imgs[i]
    ori_h, ori_w, I0 = load_image(i0)
    _, _, I1 = load_image(i1)
    flow01, flow10 = load_flow(flo01, I0), load_flow(flo10, I0)
    output = model.inference(I0, I1, model.reuse(I0, I1, flow01=flow01, flow10=flow10), timestep=0.5)
    I0 = I0.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.
    output = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.
    cv2.imwrite(f'results/{cnt}.png', cv2.resize(I0, (ori_w, ori_h)))
    cv2.imwrite(f'results/{cnt + 1}.png', cv2.resize(output, (ori_w, ori_h)))
    cnt += 2
    i0 = i1
    pbar.update(1)
    if i != len(imgs) - 1:
        flo01, flo10 = flo_forward[i], flo_backward[i]

# last frame
I1 = I1.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.
cv2.imwrite(f'results/{cnt}.png', cv2.resize(I1, (ori_w, ori_h)))
pbar.update(1)
