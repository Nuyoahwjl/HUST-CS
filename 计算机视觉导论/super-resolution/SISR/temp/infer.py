import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.edsr import EDSR
from utils.img import imresize_bicubic

MODELS = {'srcnn': SRCNN, 'fsrcnn': FSRCNN, 'espcn': ESPCN, 'edsr': EDSR}

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True, choices=MODELS.keys())
    p.add_argument('--scale', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = {
        'srcnn': SRCNN(),
        'fsrcnn': FSRCNN(scale=args.scale),
        'espcn': ESPCN(scale=args.scale),
        'edsr': EDSR(scale=args.scale)
    }[args.model]
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    model = model.to(device).eval()

    os.makedirs(args.output, exist_ok=True)
    imgs = (
        [args.input]
        if os.path.isfile(args.input)
        else [os.path.join(args.input, f) for f in os.listdir(args.input)]
    )
    to_tensor, to_img = T.ToTensor(), T.ToPILImage()

    for pth in imgs:
        if not pth.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        img = Image.open(pth).convert('RGB')
        lr = imresize_bicubic(img, args.scale, down=True)
        x = (
            to_tensor(imresize_bicubic(lr, args.scale, down=False))
            .unsqueeze(0)
            .to(device)
            if isinstance(model, SRCNN)
            else to_tensor(lr).unsqueeze(0).to(device)
        )
        sr = model(x)
        to_img(sr.squeeze(0).clamp(0, 1).cpu()).save(
            os.path.join(
                args.output,
                os.path.basename(pth).rsplit('.', 1)[0] + f'_x{args.scale}.png'
            )
        )

if __name__ == '__main__':
    main()


# ## infer：
# ```bash
# python infer.py \
#   --ckpt output/edsr_x2/best.pt \
#   --input path/to/your_image_or_folder \
#   --output output/edsr_x2/infer \
#   --model edsr \
#   --scale 2
# ```
