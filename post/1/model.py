from typing import List, Tuple
import time
import torch
import numpy as np
import torchvision
import json

from triton_python_backend_utils import get_output_config_by_name, triton_string_to_numpy, get_input_config_by_name, get_input_tensor_by_name
from c_python_backend_utils import Tensor, InferenceResponse, InferenceRequest

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0,6), device=prediction.device)] * prediction.shape[0]
    cnt = 0
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        elif kpt_label:
            kpts = x[:, 6:]
            conf, j = x[:, 5:6].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        cnt += 1
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def output_to_keypoint(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        targets.extend(
            [
                i,
                cls,
                *list(*xyxy2xywh(np.array(box)[None])),
                conf,
                *list(kpts.detach().cpu().numpy()[index]),
            ]
            for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy())
        )
    return np.array(targets)


class TritonPythonModel(object):
    def __init__(self):
        self.input_names = {
            'pred_keypoints': 'pred_keypoints',
            'scale': 'scale',
            'pad': 'pad',
        }
        self.output_names = {
            'kpoints': 'kpoints',
            'boxes': 'boxes',
            'scores': 'scores'
        }

    def initialize(self, args):
        model_config = json.loads(args['model_config'])

        if 'input' not in model_config:
            raise ValueError('Input is not defined in the model config')

        input_configs = {k: get_input_config_by_name(
            model_config, name) for k, name in self.input_names.items()}
        for k, cfg in input_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Input {self.input_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for input {self.input_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for input {self.input_names[k]} is not defined in the model config')

        if 'output' not in model_config:
            raise ValueError('Output is not defined in the model config')

        output_configs = {k: get_output_config_by_name(
            model_config, name) for k, name in self.output_names.items()}
        for k, cfg in output_configs.items():
            if cfg is None:
                raise ValueError(
                    f'Output {self.output_names[k]} is not defined in the model config')
            if 'dims' not in cfg:
                raise ValueError(
                    f'Dims for output {self.output_names[k]} are not defined in the model config')
            if 'name' not in cfg:
                raise ValueError(
                    f'Name for output {self.output_names[k]} is not defined in the model config')
            if 'data_type' not in cfg:
                raise ValueError(
                    f'Data type for output {self.output_names[k]} is not defined in the model config')

        self.output_dtypes = {k: triton_string_to_numpy(
            cfg['data_type']) for k, cfg in output_configs.items()}

    def execute(self, inference_requests: List[InferenceRequest]) -> List[InferenceResponse]:
        responses = []

        for request in inference_requests:
            batch_in = {}
            for k, name in self.input_names.items():
                tensor = get_input_tensor_by_name(request, name)
                if tensor is None:
                    raise ValueError(f'Input tensor {name} not found ' f'in request {request.request_id()}')
                batch_in[k] = tensor.as_numpy()  # shape (batch_size, ...)

            batch_out = {k: [] for k, name in self.output_names.items(
            ) if name in request.requested_output_names()}

            output = batch_in['pred_keypoints']
            batch_scale = batch_in['scale']
            batch_pad = batch_in['pad']


            output = torch.from_numpy(np.array(output))
            batch_outputs = non_max_suppression_kpt(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
            batch_kps = []
            batch_boxes = []
            batch_scores = []
            for i, output in enumerate(batch_outputs):
                with torch.no_grad():
                    output = output_to_keypoint([output])
                image_kps = []
                image_boxes = []
                image_scores= []
                image_scale = batch_scale[i]
                image_pad = batch_pad[i]
                for obj in output:
                    kps = obj[7:]
                    obj_kps = [
                        [
                            (kps[i * 3] - image_pad[0]) / image_scale[0],
                            (kps[i * 3 + 1] - image_pad[1]) / image_scale[1],
                            kps[i * 3 + 2],
                        ]
                        for i in range(len(kps) // 3)
                    ]
                    image_kps.append(obj_kps)
                    box = obj[2:6]
                    cx, cy = (box[0], box[1])
                    w, h = (box[2], box[3])
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    x1 = int((x1-image_pad[0])/image_scale[0])
                    x2 = int((x2-image_pad[0])/image_scale[0])
                    y1 = int((y1-image_pad[1])/image_scale[1])
                    y2 = int((y2-image_pad[1])/image_scale[1])
                    image_boxes.append([x1, y1, x2-x1, y2-y1])
                    image_scores.append(obj[6])
                batch_kps.append(image_kps)
                batch_boxes.append(image_boxes)
                batch_scores.append(image_scores)
            # Add dummy key points for batching
            max_objs = max(len(i) for i in batch_kps)
            for i, image_kps in enumerate(batch_kps):
                batch_out["kpoints"].append(image_kps + [[[-1, -1, -1]] * 17] * (max_objs - len(image_kps)))
                batch_out["boxes"].append(batch_boxes[i] + [[-1, -1, -1, -1]] * (max_objs - len(image_kps)))
                batch_out["scores"].append(batch_scores[i] + [-1] * (max_objs - len(image_kps)))

            # Format outputs to build an InferenceResponse
            output_tensors = [Tensor(self.output_names[k], np.asarray(
                out, dtype=self.output_dtypes[k])) for k, out in batch_out.items()]

            # TODO: should set error field from InferenceResponse constructor to handle errors
            # https://github.com/triton-inference-server/python_backend#execute
            # https://github.com/triton-inference-server/python_backend#error-handling
            response = InferenceResponse(output_tensors)
            responses.append(response)

        return responses
