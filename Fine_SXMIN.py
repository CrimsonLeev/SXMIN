import torch
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_
from tqdm import tqdm
import torch.nn as nn

from deepfluoro import DeepFluoroDataset, Transforms, get_random_offset0
from metrics import DoubleGeodesic, GeodesicSE3
from SXMIN import SXMIN
import torchvision.transforms as T
import torch.nn.functional as F



def load(id_number, height, device):
    specimen = DeepFluoroDataset(id_number)
    isocenter_pose = specimen.isocenter_pose.to(device)

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen.volume,
        specimen.spacing,
        specimen.focal_len / 2,
        height,
        delx,
        x0=specimen.x0,
        y0=specimen.y0,
        reverse_x_axis=True,
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr

def process_volume(volume,device):
    volume = torch.from_numpy((volume))
    down = nn.MaxPool3d(2)
    new_D = int(1.25 * volume.shape[0])
    new_H = int(1.25 * volume.shape[1])
    new_W = int(1.25 * volume.shape[2])
    resize_volume = F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=(new_D, new_H, new_W), mode='trilinear',
                             align_corners=True)
    padding = (
        (512 - new_W) // 2,
        512 - new_W - ((512 - new_W) // 2),
        (512 - new_H) // 2,
        512 - new_H - ((512 - new_H) // 2),
        (512 - new_D) // 2,
        512 - new_D - ((512 - new_D) // 2),
    )

    i_ct = F.pad(resize_volume, padding, "constant", -1000).squeeze(0)
    i_ct = down(i_ct).to(device, dtype=torch.float)
    i_ct = i_ct.permute(0, 2, 1, 3)

    #
    i_ct = T.functional.rotate(i_ct, angle=90)
    return i_ct


def train(
    id_number,
    model,
    lr,
    drr,
    transforms,
    specimen,
    isocenter_pose,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch
):

    criterion = MultiscaleNormalizedCrossCorrelation2d([3,7, 13], [0.5, 0.2, 0.3])
    geodesic = GeodesicSE3()
    double = DoubleGeodesic(drr.detector.sdr)
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)
    v_ct = specimen.volume
    i_ct = process_volume(v_ct,device)
    i_ct2 = i_ct

    for b in range(batch_size-1):#construct batch
        i_ct2 = torch.cat((i_ct2, i_ct), dim=0)

    best_loss = torch.inf

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    model.train()
    for epoch in range(n_epochs + 1):
        losses = []
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):
            contrast = contrast_distribution.sample().item()
            offset = get_random_offset0(batch_size, device)

            pose = isocenter_pose.compose(offset)

            i_ref = drr(specimen.volume,None, None, None, pose=pose, bone_attenuation_multiplier=contrast)

            i_ref = transforms(i_ref)


            pred_pose = model(i_ref,i_ct2)
            pred_img = drr(None, None, None, pose=pred_pose)

            ncc = criterion(pred_img, i_ref)
            log_geodesic = geodesic(pred_pose, pose)
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)

            loss =1 - ncc + 1e-2*(log_geodesic + double_geodesic)# + 1 - ncc1 + 1e-2*(log_geodesic1 + double_geodesic1)# + tre

            optimizer.zero_grad()
            loss.mean().backward()
            adaptive_clip_grad_(model.parameters())
            optimizer.step()
            scheduler.step()

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                loss=loss.mean().item(),
            )


        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(model.state_dict(), f"checkpoints/fine_{id_number:02d}_best.ckpt")

        if epoch % 50 == 0:
            torch.save( model.state_dict(),f"checkpoints/SXMIN/fine_{id_number:02d}_epoch{epoch:03d}.ckpt")


def main(
    id_number,
    height=256,
    restart=None,
    lr=3e-3,
    batch_size=6,
    n_epochs=50,
    n_batches_per_epoch=100,
):
    id_number = int(id_number)

    device = torch.device("cuda")
    specimen, isocenter_pose, transforms, drr = load(id_number, height, device)

    model = SXMIN(norm_layer= "groupnorm").to(device)

    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt)
    model = model.to(device)



    train(
        id_number,
        model,
        lr,
        drr,
        transforms,
        specimen,
        isocenter_pose,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch
    )

if __name__ == '__main__':
    main(id_number=6)
