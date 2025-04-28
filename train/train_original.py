"""
Example usage:
 python -m torch.distributed.launch --nproc_per_node=1 train.py --data ../sample_data/ --object cracker
"""


import argparse
import datetime
import os
import random
import warnings
warnings.filterwarnings("ignore")

try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser

import torch
from torch.autograd import Variable
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import sys
sys.path.insert(1, '../common')

from utils_vit import *
from models_original import *

import wandb
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import numpy as np
import cv2
import json
import trimesh
import torch.nn.functional as F

OBJECT_SIZES_M = {
    "Ketchup": [0.1486, 0.0434, 0.0645],
    "Cookies": [0.1672, 0.0402, 0.1227],
}

def load_model_points(obj_path, num_points=1000):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)

def load_model_points_trimesh(path, num_points=1000):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)  # merge all geometries
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("File is not a valid mesopt.epoch.")
    points = mesh.sample(num_points)
    return points

def get_cuboid_keypoints_3d(object_class="Cookies"):
    """
    Create the 8 corners of a 3D cuboid and its center, centered at origin.
    size: (width, height, depth) in meters
    """
    object_dims = None
    if object_class == "Cookies":
        object_dims = (0.1672, 0.0402, 0.1227)
    elif object_class == "Block_w_sandpaper":
        object_dims = (0.1534, 0.1778, 0.1534)
        
    w, h, l = object_dims
    x, y, z = l / 2, w / 2, h / 2
    corners = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x,  y, -z],
        [-x,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ x,  y,  z],
        [-x,  y,  z],
        [ 0,  0,  0], # centroid
    ], dtype=np.float32)
    return corners


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.gradcheck = False
torch.backends.cudnn.benchmark = True

start_time = datetime.datetime.now()
print("start:", start_time.strftime("%m/%d/%Y, %H:%M:%S"))

conf_parser = argparse.ArgumentParser(
    description=__doc__,  # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False,
)
conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

# Specify Training Data
parser.add_argument("--data", nargs="+", help="Path to training data")
parser.add_argument(
    "--use_s3", action="store_true", help="Use s3 buckets for training data"
)
parser.add_argument(
    "--train_buckets",
    nargs="+",
    default=[],
    help="s3 buckets containing training data. Can list multiple buckets separated by a space.",
)
parser.add_argument("--endpoint", "--endpoint_url", type=str, default=None)

# Specify Training Object
parser.add_argument(
    "--object",
    nargs="+",
    required=True,
    default=[],
    help='Object to train network for. Must match "class" field in groundtruth .json file. For best performance, only put one object of interest.',
)
parser.add_argument(
    "--workers", type=int, help="number of data loading workers", default=8
)
parser.add_argument(
    "--batchsize", "--batch_size", type=int, default=32, help="input batch size"
)
parser.add_argument(
    "--imagesize",
    type=int,
    default=512,
    help="the height / width of the input image to network",
)
parser.add_argument(
    "--lr", type=float, default=0.0001, help="learning rate, default=0.0001"
)
parser.add_argument(
    "--net_path", default=None, help="path to net (to continue training)"
)
parser.add_argument(
    "--namefile", default="epoch", help="name to put on the file of the save weightss"
)
parser.add_argument("--manualseed", type=int, help="manual seed")
parser.add_argument(
    "--epochs",
    "--epoch",
    "-e",
    type=int,
    default=60,
    help="Number of epochs to train for",
)
parser.add_argument("--loginterval", type=int, default=100)
parser.add_argument("--gpuids", nargs="+", type=int, default=[0], help="GPUs to use")
parser.add_argument(
    "--exts",
    nargs="+",
    type=str,
    default=["png"],
    help="Extensions for images to use. Can have multiple entries seperated by space. e.g. png jpg",
)
parser.add_argument(
    "--outf",
    default="output/weights",
    help="folder to output images and model checkpoints",
)
parser.add_argument("--sigma", default=4, help="keypoint creation sigma")
parser.add_argument("--local-rank", type=int, default=0)


parser.add_argument("--save", action="store_true", help="save a batch and quit")
parser.add_argument(
    "--pretrained",
    action="store_true",
    help="Use pretrained weights. Must also specify --net_path.",
)
parser.add_argument("--nbupdates", default=None, help="nb max update to network")

parser.add_argument("--val_data", nargs="+", help="Path to val data")

parser.add_argument(
    "--train_locally",
    action="store_true",
    help="Whether to train locally: True = Fluent's in-lab GPU, False = Great Lakes.",
)

parser.add_argument(
    "--obj_model_path",
    type=str,
    default="",
    help="directory of 3D model",
)

# Read the config but do not overwrite the args written
args, remaining_argv = conf_parser.parse_known_args()
defaults = {"option": "default"}

if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))


parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

local_rank = opt.local_rank

model_points = load_model_points_trimesh(opt.obj_model_path, num_points=1000)
trained_object = opt.object[0] if isinstance(opt.object, list) else str(opt.object)

# Validate Arguments
if opt.use_s3 and (opt.train_buckets is None or opt.endpoint is None):
    raise ValueError(
        "--train_buckets and --endpoint must be specified if training with data from s3 bucket."
    )

if not opt.use_s3 and opt.data is None:
    raise ValueError("--data field must be specified.")

os.makedirs(opt.outf, exist_ok=True)


with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt) + "\n")

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed) + "\n")


if local_rank == 0:
    writer = SummaryWriter(opt.outf + "/runs/")

random.seed(opt.manualseed)


torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="NCCL", init_method="env://")

torch.manual_seed(opt.manualseed)

torch.cuda.manual_seed_all(opt.manualseed)

# Data Augmentation
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59, 0.25]
    transform = transforms.Compose(
        [
            AddRandomContrast(0.2),
            AddRandomBrightness(0.2),
            # transforms.Resize(opt.imagesize),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.Resize(opt.imagesize),
        ]
    )
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose(
        [transforms.Resize(opt.imagesize), transforms.ToTensor()]
    )

# Load Model
"""
if opt.pretrained is False: it will train on a VGG-19 pretrained weight;
if opt.pretrained is True, it will train on the .pth weight provided in arg.
"""
net = DopeNetwork(pretrained=opt.pretrained,
                  stop_at_stage=6)
output_size = 50
opt.sigma = 0.5


train_dataset = CleanVisiiDopeLoader(
    opt.data,
    sigma=opt.sigma,
    output_size=output_size,
    objects=opt.object,
    use_s3=opt.use_s3,
    buckets=opt.train_buckets,
    endpoint_url=opt.endpoint,
    extensions=[".png", ".jpg"],
)
trainingdata = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchsize,
    shuffle=True,
    num_workers=opt.workers,
    pin_memory=True,
)


val_dataset = CleanVisiiDopeLoader(
    opt.val_data,
    sigma=opt.sigma,
    output_size=output_size,
    objects=opt.object,
    use_s3=opt.use_s3,
    buckets=opt.train_buckets,
    endpoint_url=opt.endpoint,
    extensions=[".png", ".jpg"],
    # validation=True  # only if your loader supports it
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=opt.batchsize,
    shuffle=False,
    num_workers=opt.workers,
    pin_memory=True,
)
global best_val_acc
best_val_acc = 0


if not trainingdata is None:
    print("training data: {} batches".format(len(trainingdata)))

print("Loading Model...")

net = torch.nn.parallel.DistributedDataParallel(
    net.cuda(), device_ids=[local_rank], output_device=local_rank
)

if opt.pretrained:
    if opt.net_path is not None:
        net.load_state_dict(torch.load(opt.net_path))
    else:
        print("Error: Did not specify path to pretrained weights.")
        quit()

parameters = filter(lambda p: p.requires_grad, net.parameters())
# optimizer = optim.Adam(parameters, lr=opt.lr) # original code
optimizer = torch.optim.AdamW(net.parameters(), lr=opt.lr, weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=20,       # first restart after 10 epochs
#     T_mult=3,     # then every 20, 40, ...
#     eta_min=1e-5,  # minimum LR to avoid 0
#     last_epoch=-1,  # -1 means it starts from the first epoch
#     verbose=True,
# )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=opt.epochs,
    eta_min=1e-6,
    last_epoch=-1, 
    verbose=True,
)

# Initialize the wandb run
wandb.init(
    project="Our_DOPE",
    config={
        "learning_rate": opt.lr,
        "batch_size": opt.batchsize,
        "epochs": opt.epochs,
    },
    name="original_block_1",
)

print("ready to train!")

nb_update_network = 0
best_results = {"epoch": None, "passed": None, "add_mean": None, "add_std": None}

scaler = torch.cuda.amp.GradScaler()



def belief_accuracy(pred_belief, gt_belief, threshold=0.08):
    """
    pred_belief, gt_belief: shape (N, K, H, W)
    Returns percentage of keypoints correctly predicted within threshold.
    """
    batch_size, num_keypoints, H, W = pred_belief.shape
    correct = 0
    total = batch_size * num_keypoints

    for b in range(batch_size):
        for k in range(num_keypoints):
            pred_pos = torch.nonzero(pred_belief[b, k] == pred_belief[b, k].max(), as_tuple=False)
            gt_pos = torch.nonzero(gt_belief[b, k] == gt_belief[b, k].max(), as_tuple=False)
            if len(pred_pos) == 0 or len(gt_pos) == 0:
                continue
            pred_yx = pred_pos[0].float()
            gt_yx = gt_pos[0].float()
            dist = torch.norm(pred_yx - gt_yx) / H  # normalize to image size
            if dist < threshold:
                correct += 1

    return correct / total


def calc_loss(loss_belief, loss_affinities):
    return 1.5 * loss_belief + 0.5 * loss_affinities


@torch.no_grad()
def evaluate_model(epoch, val_loader):
    net.eval()
    total_loss = 0
    total_accuracy = 0
    total_map = 0
    total_add = 0
    count_add = 0
    count = 0

    for batch in val_loader:
        data = batch["img"].cuda()
        target_belief = batch["beliefs"].cuda().float()
        target_affinities = batch["affinities"].cuda().float()

        with torch.cuda.amp.autocast():
            output_belief, output_aff = net(data)

            for stage in range(len(output_belief)):
                
                if target_belief.shape[2:] != output_belief[stage].shape[2:]:
                    target_belief = F.interpolate(
                        target_belief, 
                        size=output_belief[stage].shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                if target_affinities.shape[2:] != output_aff[stage].shape[2:]:
                    target_affinities = F.interpolate(
                        target_affinities,
                        size=output_aff[stage].shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

            loss_belief = sum(
                ((output_belief[stage] - target_belief) ** 2).mean()
                for stage in range(len(output_belief))
            )
            loss_affinity = sum(
                ((output_aff[stage] - target_affinities) ** 2).mean()
                for stage in range(len(output_aff))
            )
            loss = calc_loss(loss_belief, loss_affinities)

        acc = belief_accuracy(output_belief[-1], target_belief)
        mAP = compute_map(output_belief[-1], target_belief)

        total_loss += loss.item()
        total_accuracy += acc
        total_map += mAP
        count += 1

        for i in range(data.shape[0]):
            pred_keypoints = extract_keypoints_from_belief(output_belief[-1][i].cpu())
            gt_keypoints = extract_keypoints_from_belief(target_belief[i].cpu())

            # Derive JSON path from filename
            img_name = batch["img_path"][i].split("/")[-1].split(".")[0]  # e.g., 0001
            gt_json_path = batch["json_path"][i]

            try:
                quat_gt, t_gt, K = load_pose_and_intrinsics(gt_json_path, object_class=trained_object)
                R_gt = quat_to_matrix(quat_gt)

                object_points = get_cuboid_keypoints_3d(object_class=trained_object)
                success_pred, rvec_pred, t_pred = cv2.solvePnP(object_points, pred_keypoints, K, None)
                if success_pred:
                    R_pred, _ = cv2.Rodrigues(rvec_pred)
                    add = compute_add(R_pred, t_pred, R_gt, t_gt, object_points)
                    if add > 500: # filter out invalid ADD values
                        print(f"[WARN] Invalid TRAIN ADD for sample {i}: too large value")
                        continue
                    total_add += add
                    count_add += 1
            except Exception as e:
                print(f"[WARN] Failed to compute ADD for {img_name}: {e}")
                continue

    avg_loss = total_loss / count
    avg_acc = total_accuracy / count
    avg_map = total_map / count
    avg_add = total_add / count_add if count_add > 0 else 0

    if local_rank == 0:
        writer.add_scalar("loss/val_loss", avg_loss, epoch)
        writer.add_scalar("accuracy/val_accuracy", avg_acc, epoch)
        writer.add_scalar("map/val_mAP", avg_map, epoch)
        writer.add_scalar("pose/val_ADD", avg_add, epoch)

    return avg_loss, avg_acc, avg_map, avg_add



def compute_map(pred_belief, gt_belief, threshold=0.08):
    """
    Compute mean Average Precision (mAP) across keypoints.
    """
    batch_size, num_keypoints, H, W = pred_belief.shape
    aps = []

    for k in range(num_keypoints):
        y_true = []
        y_scores = []

        for b in range(batch_size):
            gt_pos = torch.nonzero(gt_belief[b, k] == gt_belief[b, k].max(), as_tuple=False)
            pred_pos = torch.nonzero(pred_belief[b, k] == pred_belief[b, k].max(), as_tuple=False)

            if len(pred_pos) == 0 or len(gt_pos) == 0:
                continue

            gt_yx = gt_pos[0].float()
            pred_yx = pred_pos[0].float()
            dist = torch.norm(pred_yx - gt_yx) / H

            is_tp = float(dist < threshold)
            score = pred_belief[b, k].max().detach().cpu().item()

            y_true.append(is_tp)
            y_scores.append(score)

        if len(y_true) > 0:
            y_true_np = np.array(y_true)
            y_scores_np = np.array(y_scores)
            ap = average_precision_score(y_true_np, y_scores_np)
            aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def compute_adds(R_pred, t_pred, R_gt, t_gt, object_points):
    """
    ADD-S: average closest-point distance for symmetric objects.
    """
    pred_pts = (R_pred @ object_points.T).T + t_pred.reshape(1, 3)
    gt_pts = (R_gt @ object_points.T).T + t_gt.reshape(1, 3)
    distances = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
    return np.mean(np.min(distances, axis=1))  # [N_pred x N_gt]


def compute_add(R_pred, t_pred, R_gt, t_gt, object_points):
    pred_pts = (R_pred @ object_points.T).T + t_pred.reshape(1, 3)
    gt_pts = (R_gt @ object_points.T).T + t_gt.reshape(1, 3)
    error = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return error.mean()


def extract_keypoints_from_belief(belief_map):
    """
    belief_map: (K, H, W) tensor (single sample)
    Returns: (K, 2) keypoints in pixel coords
    """
    keypoints = []
    for k in range(belief_map.shape[0]):
        max_idx = torch.argmax(belief_map[k])
        y, x = divmod(max_idx.item(), belief_map.shape[2])
        keypoints.append([x, y])
    return np.array(keypoints, dtype=np.float32)


def load_pose_and_intrinsics(json_path, object_class="Cookies"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle intrinsics
    if isinstance(data["camera_data"]["intrinsics"], list):
        K = np.array(data["camera_data"]["intrinsics"])
    else:
        intrin = data["camera_data"]["intrinsics"]
        K = np.array([
            [intrin["fx"], 0, intrin["cx"]],
            [0, intrin["fy"], intrin["cy"]],
            [0, 0, 1]
        ])

    # Find object
    for obj in data["objects"]:
        if obj["class"].lower() == object_class.lower():  # case-insensitive
            location = np.array(obj["location"]).reshape(3, 1)
            quat = np.array(obj["quaternion_xyzw"])
            return quat, location, K

    print(f"[SKIP] Object class '{object_class}' not found in {json_path}")
    return None, None, None


def quat_to_matrix(q):
    return R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()


def _runnetwork(epoch, train_loader, syn=False):
    global nb_update_network
    # net
    # print("Running net.train()")
    net.train()
    # print("Running net.train() done")

    loss_avg_to_log = {}
    loss_avg_to_log["loss"] = []
    loss_avg_to_log["loss_affinities"] = []
    loss_avg_to_log["loss_belief"] = []
    loss_avg_to_log["loss_class"] = []
    loss_avg_to_log["accuracy"] = []
    loss_avg_to_log["mAP"] = []
    loss_avg_to_log["ADD"] = []

    # loss_avg_to_log["output_belief"] = []
    # loss_avg_to_log["output_affinity"] = []

    for batch_idx, targets in enumerate(train_loader):
        # if batch_idx == 0:
        #     print("Running batch_idx: ", batch_idx, " epoch: ", epoch)

        optimizer.zero_grad()

        data = Variable(targets["img"].cuda())
        # target_belief = Variable(targets["beliefs"].cuda())
        # target_affinities = Variable(targets["affinities"].cuda())
        target_belief = Variable(targets["beliefs"].cuda()).float()
        target_affinities = Variable(targets["affinities"].cuda()).float()

        # print("mean target belief:", target_belief.mean(), ", max target belief:", target_belief.max())
        # print("mean target affinity:", target_affinities.mean(), ", max target affinity:", target_affinities.max())

        # output_belief, output_aff = net(data)
        with torch.cuda.amp.autocast():
            output_belief, output_aff = net(data)

            # print("mean output belief:", output_belief[-1].mean().item(), ", mean output affinity:", output_aff[-1].mean().item())
            # print("max output belief:", output_belief[-1].max().item(), ", max output affinity:", output_aff[-1].max().item())

            # loss = None

            loss_belief = torch.tensor(0).float().cuda()
            loss_affinities = torch.tensor(0).float().cuda()
            loss_class = torch.tensor(0).float().cuda()

            for stage in range(len(output_aff)):  # output, each belief map layers.


                if target_affinities.shape != output_aff[stage].shape:
                    target_affinities = F.interpolate(
                        target_affinities, 
                        size=output_aff[stage].shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                if target_belief.shape != output_belief[stage].shape:
                    target_belief = F.interpolate(
                        target_belief, 
                        size=output_belief[stage].shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )


                if target_affinities.shape != output_aff[stage].shape:
                    target_affinities = F.interpolate(
                        target_affinities, 
                        size=output_aff[stage].shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                if target_belief.shape != output_belief[stage].shape:
                    target_belief = F.interpolate(
                        target_belief, 
                        size=output_belief[stage].shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )


                loss_affinities += (
                    (output_aff[stage] - target_affinities)
                    * (output_aff[stage] - target_affinities)
                ).mean()

                loss_belief += (
                    (output_belief[stage] - target_belief)
                    * (output_belief[stage] - target_belief)
                ).mean()

            # loss = loss_affinities + loss_belief # original code
            loss = calc_loss(loss_belief, loss_affinities)



        with torch.no_grad():
            accuracy = belief_accuracy(output_belief[-1], target_belief)
            mAP = compute_map(output_belief[-1], target_belief)
            loss_avg_to_log["accuracy"].append(accuracy)
            loss_avg_to_log["mAP"].append(mAP)

            # Compute training ADD using keypoints and fixed intrinsics
            object_points = get_cuboid_keypoints_3d(object_class=trained_object)
            if "json_path" in targets:
                for i in range(data.shape[0]):
                    pred_keypoints = extract_keypoints_from_belief(output_belief[-1][i].cpu())
                    gt_keypoints = extract_keypoints_from_belief(target_belief[i].cpu())

                    json_path = targets["json_path"][i]
                    try:
                        quat_gt, t_gt, K = load_pose_and_intrinsics(json_path, object_class=trained_object)
                        if quat_gt is None or t_gt is None or K is None:
                            continue

                        R_gt = quat_to_matrix(quat_gt)
                        success_pred, rvec_pred, t_pred = cv2.solvePnP(object_points, pred_keypoints, K, None)
                        if success_pred:
                            R_pred, _ = cv2.Rodrigues(rvec_pred)
                            add = compute_add(R_pred, t_pred, R_gt, t_gt, object_points)
                            if add > 500: # filter out invalid ADD values
                                print(f"[WARN] Invalid TRAIN ADD for sample {i}: too large value")
                                continue
                            loss_avg_to_log["ADD"].append(add)

                    except Exception as e:
                        print(f"[WARN] Failed to compute TRAIN ADD for sample {i}: {e}")
                        continue

        if batch_idx == 0:
            post = "train"

            if local_rank == 0:

                for i_output in range(1):

                    # input images
                    writer.add_image(
                        f"{post}_input_{i_output}",
                        targets["img_original"][i_output],
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps gt
                    imgs = VisualizeBeliefMap(target_belief[i_output])

                    img, grid = save_image(
                        imgs, "some_img.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_ground_truth_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

                    # belief maps guess
                    imgs = VisualizeBeliefMap(output_belief[-1][i_output])
                    img, grid = save_image(
                        imgs, "some_img.png", mean=0, std=1, nrow=3, save=False
                    )
                    writer.add_image(
                        f"{post}_belief_guess_{i_output}",
                        grid,
                        epoch,
                        dataformats="CWH",
                    )

        # print("batch: ",batch_idx, " loss: ", loss.item(), " loss_belief: ", loss_belief.item(), " loss_affinities: ", loss_affinities.item())

        # loss.backward()
        scaler.scale(loss).backward()

        # total_grad = 0
        # count = 0
        # for name, param in net.named_parameters():scaler.update()
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2).item()
        #         total_grad += grad_norm
        #         count += 1
        # print(f"Avg grad norm: {total_grad / max(count,1):.6f}")


        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        nb_update_network += 1

        # log the loss
        loss_avg_to_log["loss"].append(loss.item())
        loss_avg_to_log["loss_class"].append(loss_class.item())
        loss_avg_to_log["loss_affinities"].append(loss_affinities.item())
        loss_avg_to_log["loss_belief"].append(loss_belief.item())
        
        # loss_avg_to_log["output_belief"].append(output_belief[-1].cpu().detach().clone().float())
        # loss_avg_to_log["output_affinity"].append(output_aff[-1].cpu().detach().clone().float())  # save the last affinity map output for logging purposes

        

        if batch_idx % opt.loginterval == 0:
            # print(
            #     "Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.15f} \tLocal Rank: {}".format(
            #         epoch,
            #         batch_idx * len(data),
            #         len(train_loader.dataset),
            #         100.0 * batch_idx / len(train_loader),
            #         loss.item(),
            #         local_rank,
            #     )
            # )
            print(
                "[{}/{} ({:.0f}%)] Epoch {}, Loss: {:.15f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    epoch,
                    loss.item(),
                )
            )

    avg_train_add = np.mean(loss_avg_to_log["ADD"]) if len(loss_avg_to_log["ADD"]) > 0 else -1

    # log the loss values
    if local_rank == 0:

        writer.add_scalar("loss/train_loss", np.mean(loss_avg_to_log["loss"]), epoch)
        writer.add_scalar(
            "loss/train_cls", np.mean(loss_avg_to_log["loss_class"]), epoch
        )
        writer.add_scalar(
            "loss/train_aff", np.mean(loss_avg_to_log["loss_affinities"]), epoch
        )
        writer.add_scalar(
            "loss/train_bel", np.mean(loss_avg_to_log["loss_belief"]), epoch
        )

        writer.add_scalar("accuracy/train_accuracy", np.mean(loss_avg_to_log["accuracy"]), epoch)
        writer.add_scalar("map/train_mAP", np.mean(loss_avg_to_log["mAP"]), epoch)
        writer.add_scalar("pose/train_ADD", avg_train_add, epoch)

    print(
        f"Epoch {epoch} Training completed: \n"
        f"Local Rank: {local_rank} | "
        f"Loss: {np.mean(loss_avg_to_log['loss']):.6f} | "
        f"Loss_Belief: {np.mean(loss_avg_to_log['loss_belief']):.4f} | "
        f"Loss_Affinity: {np.mean(loss_avg_to_log['loss_affinities']):.4f} | \n"
        f"Accuracy: {np.mean(loss_avg_to_log['accuracy']):.2f} | "
        f"mAP: {np.mean(loss_avg_to_log['mAP']):.2f} | "
        f"ADD: {avg_train_add:.2f} | \n"
        f"Updates: {nb_update_network} | "
        f"Samples: {len(train_loader.dataset)}"
    )

    wandb.log({
        "epoch": epoch,
        "train_loss": np.mean(loss_avg_to_log['loss']),
        "train_loss_affinities": np.mean(loss_avg_to_log['loss_affinities']),
        "train_loss_belief": np.mean(loss_avg_to_log['loss_belief']),
        "current_lr": optimizer.param_groups[0]['lr'],
        "train_accuracy": np.mean(loss_avg_to_log["accuracy"]),
        "train_mAP": np.mean(loss_avg_to_log["mAP"]),
        "train_ADD": avg_train_add,
        # "output_belief": np.mean(loss_avg_to_log["output_belief"]),
        # "output_affinity": np.mean(loss_avg_to_log["output_affinity"]),
        # "val_loss": val_loss,
        # "accuracy": accuracy,
    })



start_epoch = 1
if opt.pretrained and opt.net_path is not None:
    # we started with a saved checkpoint, we start numbering
    # checkpoints after the loaded one
    try:
        start_epoch = int(os.path.splitext(os.path.basename(opt.net_path).split('_')[1])[0]) + 1
        print(f"Starting at epoch {start_epoch}")
    except:
        print("Could not parse epoch number from filename")
        raise ValueError(f"Failed to extract epoch number from {opt.net_path}, try again or rename as sth_XX.path, where XX is the epoch number")

for epoch in range(start_epoch, opt.epochs + 1):

    print("Training Epoch: ", epoch)
    _runnetwork(epoch, trainingdata)
    scheduler.step()

    val_loss, val_acc, val_map, val_add = evaluate_model(epoch, val_loader)

    print(
        f"Val Loss: {val_loss:.6f} | "
        f"Val Accuracy: {val_acc:.4f} |"
        f"Val mAP: {val_map:.4f} |"
        f"Val ADD: {val_add:.4f}"
    )

    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_mAP": val_map,
        "val_ADD": val_add,
    })

    # Save best model
    if local_rank == 0:
        if epoch == 1 or val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                net.state_dict(),
                f"{opt.outf}/best_model.pth",
            )


    try:
        if local_rank == 0 and epoch % 10 == 0:

            torch.save(
                net.state_dict(),
                f"{opt.outf}/{opt.namefile}_{str(epoch).zfill(2)}.pth",
            )
    except Exception as e:
        print(f"Encountered Exception: {e}")

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

if local_rank == 0:
    torch.save(
        net.state_dict(), f"{opt.outf}/{opt.namefile}_{str(epoch).zfill(2)}.pth"
    )
else:
    torch.save(
        net.state_dict(),
        f"{opt.outf}/{opt.namefile}_{str(epoch).zfill(2)}_rank_{local_rank}.pth",
    )

print("end:", datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
print(
    "Total time taken: ",
    str(datetime.datetime.now() - start_time).split(".")[0],
)
