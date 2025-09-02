import torch
import torch.nn as nn
import torch.nn.functional as F
from drr.drr1 import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
from deepfluoro import DeepFluoroDataset, Transforms, get_random_offset41
from metrics import DoubleGeodesic, GeodesicSE3
from SXMIN import SXMIN
import torchvision.transforms as T
import glob
import random
import nibabel as nib

def process_volume(volume,device):
    down = nn.MaxPool3d(2)

    volume = torch.from_numpy(volume)
    padding = ((512 - volume.shape[2]) // 2, (512 - volume.shape[2]) - (512 - volume.shape[2]) // 2, 0, 0, 0, 0)

    volume= F.pad(volume.unsqueeze(0).unsqueeze(0), padding, "constant",
                     0)
    volume = volume.squeeze(0).to(device, dtype=torch.float)
    volume = volume.masked_fill(volume == 0, -1000)
    volume = down(volume)

    i_ct = volume.permute(0, 2, 1, 3)
    i_ct = T.functional.rotate(i_ct, angle=90)

    return i_ct

def load(id_number, height, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)
    dataset = glob.glob("./dataset/datase6*.nii.gz")#[0:60]
    vol = []
    resize_vol = []

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.spacing,
        specimen.focal_len / 2,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height)


    for i in range(len(dataset)):

        v_ct = dataset[i]
        v_ct = nib.load(v_ct)

        v_ct = v_ct.get_fdata()

        i_ct = process_volume(v_ct,device,height)
        i_ct = transforms(i_ct)

        resize_vol.append(i_ct)
        vol.append(v_ct)

        i = i + 1



    return specimen, isocenter_pose, transforms, drr, vol, resize_vol


def train(
    model,
    lr,
    drr,
    transforms,
    data_loader,
    resize_vol,
    isocenter_pose,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
):
    idn = [i for i in range(n_batches_per_epoch)]
    random.shuffle(idn)
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.sdr)
    contrast_distribution = torch.distributions.Uniform(3.0, 6.0)

    best_loss = torch.inf

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    for epoch in range(n_epochs + 1):
        losses = []

        #suffle data
        random.shuffle(data_loader)

        i =0
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            ii = idn[i]
            vol = data_loader[ii]
            resize_v = resize_vol[ii]
            rv = resize_v
            for b in range(batch_size - 1):
                rv = torch.cat((rv, resize_v), dim=0)
            resize_v = rv
            i = i + 1
            vol = torch.from_numpy(vol).to(device, dtype=torch.float)
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset41(batch_size, device)
            pose = isocenter_pose.compose(offset)
            img1 = drr(vol, None, None, None, pose=pose, bone_attenuation_multiplier=contrast)
            img1 = transforms(img1)
            img2 = transforms(resize_v)


            pred_pose = model(img1, img2)

            pred_img = drr(vol, None, None, None, pose=pred_pose)
            pred_img = transforms(pred_img)

            ncc = metric(pred_img, img1)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic)

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                loss=loss.mean().item()
            )

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")

        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(model.state_dict(),f"checkpoints/SXMIN/pre_SXMIN_metal_best.ckpt",)



        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"checkpoints/SXMIN/pre_SXMIN_Metal_epoch{epoch:03d}.ckpt")


def main(
    height=256,
    restart=None,#f"checkpoints/SXMIN/pre_dSXMIN_epoch4950.ckpt",
    lr=3e-4,
    batch_size=2,
    n_epochs=10000,
    n_batches_per_epoch=100,
):

    device = torch.device("cuda")
    specimen, isocenter_pose, transforms, drr, data_loader, resize_vol = load(1, height, device)


    model = SXMIN(norm_layer = "groupnorm")
    model = model.to(device)
    #SGD(model.parameters(), lr=lr, momentum=0.9)
    if restart is not None:
        ckpt = torch.load(restart, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt["model_state_dict"])


    train(
        model,
        lr,
        drr,
        transforms,
        data_loader,
        resize_vol,
        isocenter_pose,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch
    )

if __name__ == '__main__':
    main()
