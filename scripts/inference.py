import os
import sys
import time
from argparse import Namespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from models.psp import pSp
from options.test_options import TestOptions
from utils.common import log_input_image, tensor2im


def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert (
            len(test_opts.resize_factors.split(",")) == 1
        ), "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(
            test_opts.exp_dir,
            "inference_results",
            "downsampling_{}".format(test_opts.resize_factors),
        )
        out_path_coupled = os.path.join(
            test_opts.exp_dir,
            "inference_coupled",
            "downsampling_{}".format(test_opts.resize_factors),
        )
    else:
        out_path_results = os.path.join(test_opts.exp_dir, "inference_results")
        out_path_coupled = os.path.join(test_opts.exp_dir, "inference_coupled")

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    opts.update(vars(test_opts))
    if "learn_in_w" not in opts:
        opts["learn_in_w"] = False
    if "output_size" not in opts:
        opts["output_size"] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    # net.cuda()

    print("Loading dataset for {}".format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args["transforms"](opts).get_transforms()
    dataset = InferenceDataset(
        root=opts.data_path, transform=transforms_dict["transform_inference"], opts=opts
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=int(opts.test_workers),
        drop_last=True,
    )

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cpu().float()
            print(input_cuda.shape)
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (
                    (256, 256)
                    if opts.resize_outputs
                    else (opts.output_size, opts.output_size)
                )
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = cv2.imread(im_path)
                    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                    res = np.concatenate(
                        [
                            cv2.resize(
                                source, resize_amount, interpolation=cv2.INTER_LINEAR
                            ),
                            cv2.resize(
                                input_im, resize_amount, interpolation=cv2.INTER_NEAREST
                            ),
                            cv2.resize(
                                result, resize_amount, interpolation=cv2.INTER_LINEAR
                            ),
                        ],
                        axis=1,
                    )
                else:
                    # otherwise, save the original and output
                    res = np.concatenate(
                        [
                            cv2.resize(
                                input_im, resize_amount, interpolation=cv2.INTER_NEAREST
                            ),
                            cv2.resize(
                                result, resize_amount, interpolation=cv2.INTER_LINEAR
                            ),
                        ],
                        axis=1,
                    )
                cv2.imwrite(
                    os.path.join(out_path_coupled, os.path.basename(im_path)),
                    cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
                )

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            cv2.imwrite(im_save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, "stats.txt")
    result_str = "Runtime {:.4f}+-{:.4f}".format(
        np.mean(global_time), np.std(global_time)
    )
    print(result_str)

    with open(stats_path, "w") as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype("float32")
            _, latent_to_inject = net(
                torch.from_numpy(vec_to_inject).to("cpu"),
                input_code=True,
                return_latents=True,
            )
            # get output image with injected style vector
            res = net(
                input_image.unsqueeze(0).to("cpu").float(),
                latent_mask=latent_mask,
                inject_latent=latent_to_inject,
                alpha=opts.mix_alpha,
                resize=opts.resize_outputs,
            )
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == "__main__":
    run()
