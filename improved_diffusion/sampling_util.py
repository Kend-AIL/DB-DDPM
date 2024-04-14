import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as tvu
from PIL import Image
from kornia.enhance import denormalize
from sklearn.metrics import f1_score, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import dist_util

# from datasets.monu import MonuDataset

from .Evaluation import ssim, mse, nmse, psnr


def convert2img(x):
    img_x=np.abs(x[:,0]+x[:,1]*1j)
    min_vals = np.min(img_x, axis=(1, 2), keepdims=True)
    max_vals = np.max(img_x, axis=(1, 2), keepdims=True)
    scaled_images = (img_x - min_vals) / (max_vals - min_vals)
    return scaled_images


def Kline_filter(model_path,kspace,threshold):
    model=torch.load(model_path)
    predictions=model(kspace)
    binary_predictions = (predictions > threshold).int().unsqueeze(-1)  

    mask=binary_predictions.repeat(1, 1, 128)
    return mask
def sampling_without_kspace(batch_size, diffusion_model, ddp_model, output_folder, dataset, logger, clip_denoised, vote_num=4):
    ddp_model.eval()
    batch_size = batch_size
    major_vote_number = vote_num
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    loader_iter = iter(loader)
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = len(loader)

   

    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]

    with torch.no_grad():
        for round_index in range(n_rounds):
            print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
            data_ = next(loader_iter)
            gt_mask = data_["GT"]
            condition_on = {"conditioned_image": data_["input"]}
            name = data_["ipath"]
            print(gt_mask.max(),gt_mask.min())
            print(data_["input"].max(),data_["input"].min())
            folders = [n.split("/")[0] for n in name]
            imgs = [n.split("/")[1] for n in name]
            condition_on = condition_on["conditioned_image"]
            former_frame_for_feature_extraction = condition_on.to(dist_util.dev())

            
            model_kwargs = {
                "conditioned_image": torch.cat([former_frame_for_feature_extraction] * major_vote_number)}

            x = diffusion_model.p_sample_loop(
                ddp_model,
                (major_vote_number*batch_size, gt_mask.shape[1], former_frame_for_feature_extraction.shape[2],
                    former_frame_for_feature_extraction.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )

            # x = (x + 1.0) / 2.0

            # if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
            #     x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            # x = torch.clamp(x, 0.0, 1.0)

            # major vote result
            # x = x.mean(dim=0, keepdim=True)
            x_list = torch.split(x, batch_size)
            tt = torch.stack(x_list)
            x = tt.mean(dim=0)
            # for tt in x_list:
            #     tt = tt.mean(dim=0, keepdim=True)
            # x = torch.cat(x_list)
            gt_img=convert2img(gt_mask.detach().cpu().numpy())
            in_img=convert2img(condition_on.detach().cpu().numpy())
            out_img=convert2img(x.detach().cpu().numpy())
            print(gt_img.shape)
            for i in range(x.shape[0]):
                # save as outer training ids
                # current_output = x[i][0] + 1
                # current_output[current_output == current_output.max()] = 0
                # out_img = Image.fromarray(x[i][0].detach().cpu().numpy().astype('uint8'))
                # out_img.putpalette(cityspallete)
                # out_img.save(
                #     os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                sub_folder = os.path.join(output_folder, folders[i])
                os.makedirs(sub_folder, exist_ok=True)
                gt_img_save = Image.fromarray((gt_img[i])*255)
                gt_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_gt.png"))
                in_img_save = Image.fromarray((in_img[i])*255)
                in_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_input.png"))
                out_img_save = Image.fromarray((out_img[i])*255)
                out_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_output.png"))

           
            psnr_ = psnr(gt_img, out_img)
            mse_ = mse(gt_img, out_img)
            nmse_ = nmse(gt_img, out_img)
            ssim_ = ssim(gt_img, out_img)
            psnr_list.append(psnr_)
            mse_list.append(mse_)
            nmse_list.append(nmse_)
            ssim_list.append(ssim_)
                # f1, miou, wcov, fbound = calculate_metrics(out_im[0], gt_im[0])
                # f1_score_list.append(f1)
                # miou_list.append(miou)
                # wcov_list.append(wcov)
                # fbound_list.append(fbound)

            logger.info(
                    f"{imgs} psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")

    my_length = len(psnr_list)
    length_of_data = torch.tensor(len(psnr_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    ssim_list = [i.item() for i in ssim_list]
    nmse_list = [i.item() for i in nmse_list]

    psnr_tensor = torch.tensor(psnr_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    ssim_tensor = torch.tensor(ssim_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    mse_tensor = torch.tensor(mse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    nmse_tensor = torch.tensor(nmse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    gathered_psnr = [torch.ones_like(psnr_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_ssim = [torch.ones_like(ssim_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_mse = [torch.ones_like(mse_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_nmse = [torch.ones_like(nmse_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_psnr, psnr_tensor)
    dist.all_gather(gathered_ssim, ssim_tensor)
    dist.all_gather(gathered_mse, mse_tensor)
    dist.all_gather(gathered_nmse, nmse_tensor)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    gathered_psnr = torch.cat(gathered_psnr)
    # gathered_psnr = gathered_psnr[gathered_psnr != -1]
    logger.info(f"mean psnr {gathered_psnr.mean()}")

    gathered_ssim = torch.cat(gathered_ssim)
    # gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean ssim {gathered_ssim.mean()}")
    gathered_mse = torch.cat(gathered_mse)
    # gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean mse {gathered_mse.mean()}")
    gathered_nmse = torch.cat(gathered_nmse)
    # gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean nmse {gathered_nmse.mean()}")

    dist.barrier()
    return gathered_psnr.mean().item(), gathered_ssim.mean().item(), gathered_nmse.mean().item()


def sampling_with_kspace(batch_size, diffusion_model, ddp_model,k_deteck_model,output_folder, dataset, logger, clip_denoised, vote_num=4):
    ddp_model.eval()
    batch_size = batch_size
    major_vote_number = vote_num
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    loader_iter = iter(loader)
    os.makedirs(output_folder, exist_ok=True)
    n_rounds = len(loader)

   

    ssim_list, mse_list, nmse_list, psnr_list = [],[],[],[]

    with torch.no_grad():
        for round_index in range(n_rounds):
            print(f"Current Round: {round_index+1} / Total Round: {n_rounds}")
            data_ = next(loader_iter)
            gt_mask = data_["GT"].to(torch.float32)
            condition_on = {"conditioned_image": data_["input"].to(torch.float32)}
            name = data_["ipath"]
            kspace=data_["condtioned_kspace"].to(torch.float32)
            mask=Kline_filter(k_deteck_model,data_["full_ksapce"],0.9)
            folders = [n.split("/")[0] for n in name]
            imgs = [n.split("/")[1] for n in name]
            condition_on = condition_on["conditioned_image"]
            noise_img = condition_on.to(dist_util.dev())
            mask=mask.to(dist_util.dev())
            kspace=kspace.to(dist_util.dev())

            

            
            model_kwargs = {
                "conditioned_image": torch.cat([noise_img] * major_vote_number)}
            kspace_for_vote=torch.cat([kspace] * major_vote_number)
            mask_for_vote=torch.cat([mask] * major_vote_number)
            x = diffusion_model.p_sample_loop_condition(
                ddp_model,
                (major_vote_number*batch_size, 2, noise_img.shape[2],
                    noise_img.shape[3]),
                progress=True,
                kspace=kspace_for_vote,
                mask=mask_for_vote,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )

            # x = (x + 1.0) / 2.0

            # if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
            #     x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            # x = torch.clamp(x, 0.0, 1.0)

            # major vote result
            # x = x.mean(dim=0, keepdim=True)
            x_list = torch.split(x, batch_size)
            tt = torch.stack(x_list)
            x = tt.mean(dim=0)
            # for tt in x_list:
            #     tt = tt.mean(dim=0, keepdim=True)
            # x = torch.cat(x_list)
            gt_img=convert2img(gt_mask.detach().cpu().numpy())
            in_img=convert2img(condition_on.detach().cpu().numpy())
            out_img=convert2img(x.detach().cpu().numpy())
            for i in range(x.shape[0]):
                # save as outer training ids
                # current_output = x[i][0] + 1
                # current_output[current_output == current_output.max()] = 0
                # out_img = Image.fromarray(x[i][0].detach().cpu().numpy().astype('uint8'))
                # out_img.putpalette(cityspallete)
                # out_img.save(
                #     os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                sub_folder = os.path.join(output_folder, folders[i])
                os.makedirs(sub_folder, exist_ok=True)
                gt_img_save = Image.fromarray((gt_img[i])*255)
                gt_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_gt.png"))
                in_img_save = Image.fromarray((in_img[i])*255)
                in_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_input.png"))
                out_img_save = Image.fromarray((out_img[i])*255)
                out_img_save.convert("L").save(
                    os.path.join(output_folder, folders[i],f"{imgs[i]}_output.png"))

           
            psnr_ = psnr(gt_img, out_img)
            mse_ = mse(gt_img, out_img)
            nmse_ = nmse(gt_img, out_img)
            ssim_ = ssim(gt_img, out_img)
            psnr_list.append(psnr_)
            mse_list.append(mse_)
            nmse_list.append(nmse_)
            ssim_list.append(ssim_)
                # f1, miou, wcov, fbound = calculate_metrics(out_im[0], gt_im[0])
                # f1_score_list.append(f1)
                # miou_list.append(miou)
                # wcov_list.append(wcov)
                # fbound_list.append(fbound)

            logger.info(
                    f"{imgs} psnr {psnr_list[-1]}, ssim {ssim_list[-1]}, mse {mse_list[-1]}, nmse {nmse_list[-1]}")

    my_length = len(psnr_list)
    length_of_data = torch.tensor(len(psnr_list), device=dist_util.dev())
    gathered_length_of_data = [torch.tensor(1, device=dist_util.dev()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_length_of_data, length_of_data)
    max_len = torch.max(torch.stack(gathered_length_of_data))

    ssim_list = [i.item() for i in ssim_list]
    nmse_list = [i.item() for i in nmse_list]

    psnr_tensor = torch.tensor(psnr_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    ssim_tensor = torch.tensor(ssim_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    mse_tensor = torch.tensor(mse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    nmse_tensor = torch.tensor(nmse_list + [torch.tensor(-1)] * (max_len - my_length), device=dist_util.dev())
    gathered_psnr = [torch.ones_like(psnr_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_ssim = [torch.ones_like(ssim_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_mse = [torch.ones_like(mse_tensor) * -1 for _ in range(dist.get_world_size())]
    gathered_nmse = [torch.ones_like(nmse_tensor) * -1 for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_psnr, psnr_tensor)
    dist.all_gather(gathered_ssim, ssim_tensor)
    dist.all_gather(gathered_mse, mse_tensor)
    dist.all_gather(gathered_nmse, nmse_tensor)

    # if dist.get_rank() == 0:
    logger.info("measure total avg")
    gathered_psnr = torch.cat(gathered_psnr)
    # gathered_psnr = gathered_psnr[gathered_psnr != -1]
    logger.info(f"mean psnr {gathered_psnr.mean()}")

    gathered_ssim = torch.cat(gathered_ssim)
    # gathered_f1 = gathered_f1[gathered_f1 != -1]
    logger.info(f"mean ssim {gathered_ssim.mean()}")
    gathered_mse = torch.cat(gathered_mse)
    # gathered_wcov = gathered_wcov[gathered_wcov != -1]
    logger.info(f"mean mse {gathered_mse.mean()}")
    gathered_nmse = torch.cat(gathered_nmse)
    # gathered_boundf = gathered_boundf[gathered_boundf != -1]
    logger.info(f"mean nmse {gathered_nmse.mean()}")

    dist.barrier()
    return gathered_psnr.mean().item(), gathered_ssim.mean().item(), gathered_nmse.mean().item()

def sample_one_withou_ksapce(inputs, diffusion_model, ddp_model, output_folder, logger, clip_denoised, vote_num=4):
           
           
            
            condition_on = torch.from_numpy(inputs).unsqueeze(0).to(torch.float32)
            noise_img = condition_on.to(dist_util.dev())
            
            major_vote_number = vote_num
            

            
            model_kwargs = {
                "conditioned_image": torch.cat([noise_img] * major_vote_number)}
            
            x = diffusion_model.p_sample_loop(
                ddp_model,
                (major_vote_number, 2, noise_img.shape[2],
                    noise_img.shape[3]),
                progress=True,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs
            )

            # x = (x + 1.0) / 2.0

            # if x.shape[2] != gt_mask.shape[2] or x.shape[3] != gt_mask.shape[3]:
            #     x = F.interpolate(x, gt_mask.shape[2:], mode='bilinear')

            # x = torch.clamp(x, 0.0, 1.0)

            # major vote result
            # x = x.mean(dim=0, keepdim=True)
            
            
            x_out = x.mean(dim=0).unsqueeze(0)
            # for tt in x_list:
            #     tt = tt.mean(dim=0, keepdim=True)
            # x = torch.cat(x_list)
            
            
            in_img=convert2img(condition_on.detach().cpu().numpy())
            out_img=convert2img(x_out.detach().cpu().numpy())
            for i in range(x_out.shape[0]):
                # save as outer training ids
                # current_output = x[i][0] + 1
                # current_output[current_output == current_output.max()] = 0
                # out_img = Image.fromarray(x[i][0].detach().cpu().numpy().astype('uint8'))
                # out_img.putpalette(cityspallete)
                # out_img.save(
                #     os.path.join(output_folder, f"{name[i]}_model_output_palette.png"))
                
                

                in_img_save = Image.fromarray((in_img[i])*255)
                in_img_save.convert("L").save(
                    os.path.join(output_folder, "input.png"))
                out_img_save = Image.fromarray((out_img[i])*255)
                out_img_save.convert("L").save(
                    os.path.join(output_folder, "output.png"))

    

