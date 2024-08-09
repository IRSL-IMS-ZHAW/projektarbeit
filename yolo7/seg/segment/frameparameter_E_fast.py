
#import argparse
import os
import platform
import sys
from pathlib import Path
import time
import cupy as cp
import numpy as np
import torch
from numpy import ones,vstack
from numpy.linalg import lstsq

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

frame_num = 0
cp.get_default_memory_pool().free_all_blocks()
last_save = 0
factor = 2

def principal_component_analysis(points):
    points = np.array(points, dtype=np.float32)
    # Center the points
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean

    # PCA using SVD
    u, s, vt = np.linalg.svd(points_centered)
    pc1 = vt[0]  # First principal component

    # Calculate endpoints for the PCA line
    scale_factor = np.max(np.std(points, axis=0)) * 3  # Scale factor based on standard deviation
    line_points = np.array([pc1*(-scale_factor), pc1*(scale_factor)]) + points_mean
    
    return line_points


def gpu_principal_component_analysis(points):
    # Convert points to a PyTorch tensor and transfer to GPU if available
    points = torch.tensor(points, dtype=torch.float32)
    if torch.cuda.is_available():
        points = points.cuda()

    # Center the points
    points_mean = points.mean(axis=0)
    points_centered = points - points_mean

    # PCA using SVD in PyTorch
    u, s, vt = torch.linalg.svd(points_centered, full_matrices=False)
    pc1 = vt[0]  # First principal component

    # Calculate endpoints for the PCA line
    scale_factor = torch.max(torch.std(points, axis=0)) * 3  # Scale factor based on standard deviation
    line_points = torch.stack([pc1*(-scale_factor), pc1*(scale_factor)]) + points_mean

    # Move the result back to CPU and convert to numpy if further processing is required
    if line_points.is_cuda:
        line_points = line_points.cpu()
    line_points = line_points.numpy()
    
    return line_points


def gpu_max_pooling(matrix, factor=16): #factor 8 works fine -> measurements with 16
    # Convert numpy array to CuPy array for GPU processing
    matrix = cp.asarray(matrix)
    
    # Get the dimensions of the matrix
    n, m = matrix.shape
    
    # New dimensions
    new_n = n // factor
    new_m = m // factor
    
    # Reshape and max over axis to achieve pooling
    downscaled_matrix = matrix.reshape(new_n, factor, new_m, factor).max(axis=(1, 3))
    
    return downscaled_matrix.get()  # Convert back to NumPy array if needed


def highest_concentration_area(matrix):
    # Get the shape of the matrix
    n, m = matrix.shape
    
    # Compute the size of the blocks
    block_size_n = n // 3
    block_size_m = m // 3
    
    #>>>print('===========================================================')
    #>>>print(f'The matrix has dimensions: {n}x{m}')
    #>>>print(f'Its sub blocks has dimension: {block_size_n}x{block_size_m}')
    # Initialize the maximum sum and its index
    max_sum = -1
    max_index = (0, 0)
    
    # Iterate over each possible 3x3 block
    for i in range(3):
        for j in range(3):
            start_row = i * block_size_n
            end_row = (i + 1) * block_size_n
            start_col = j * block_size_m
            end_col = (j + 1) * block_size_m
            
            # Extract the submatrix and calculate its sum
            submatrix = matrix[start_row:end_row, start_col:end_col]
            current_sum = np.sum(submatrix)
            
            # Check if this submatrix has the highest sum
            if current_sum > max_sum:
                max_sum = current_sum
                max_index = (i, j)
    
    return max_index, max_sum

def load_model(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        dnn=False,  # use OpenCV DNN for ONNX inference
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        half=False,  # use FP16 half-precision inference
        dev='',
        imgsz=(640, 640),  # inference size (height, width)
        bs=1
    ):
    # Load model
    device = select_device(dev)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
   
    print('Model Loaded Successfuly')
    return model, stride, names, device

def normalize_vector(command):
    norm = np.linalg.norm(command)
    if norm == 0:
        return command  # Can't normalize a zero vector, return original
    return command / norm

def create_points(line_points):
    # Convert points directly to integer numpy arrays and ensure within image dimensions
    start_point = np.int32(line_points[0])
    end_point = np.int32(line_points[1])

    # Compute mid_point directly in numpy to utilize vectorized operations
    mid_point = (start_point + end_point) // 2

    return start_point, end_point, mid_point

def process_points(line_points, center_point):
    
    start_point, end_point, mid_point = create_points(line_points)

    start_point = np.array(start_point, dtype=np.int32)
    end_point = np.array(end_point, dtype=np.int32)

    # Calculate distance using numpy's norm (vectorized operation)
    dist = np.linalg.norm(mid_point - center_point)
    #print(f'dist = {dist}')

    print("======================================")
    print("End_point:", (end_point))
    #print("Type of end_point:", type(end_point))
    #print("Shape of end_point:", end_point.shape)

    print("Center_point:", (center_point))
    print("Start_point:", (start_point))
    #print("Type of start_point:", type(start_point))
    #print("Shape of start_point:", start_point.shape)
    
    point = None
    speed = 4

    if dist > (center_point[1]*3/5):
        command = mid_point - center_point
        speed = 2
    else:
        point = end_point
        #if end_point[1] > center_point[1]:          # x-axis
        #    command = end_point - center_point
        
        '''if start_point[1] > end_point[1]:           # x-axis
            command = start_point - center_point

        if start_point[0] < end_point[0]:           # y-axis
            command = start_point - center_point'''
        
        #if (start_point[0] > center_point[0]) and (end_point[0] < center_point[0]):        # x-axis
        #    command = start_point - center_point

        #if (start_point[1] < center_point[1]) and (end_point[1] > center_point[1]):        # y-axis
        #    command = start_point - center_point

        dif_x = start_point[0] - end_point[0]
        dif_y = start_point[1] - end_point[1]
        
     
        if (start_point[1] < center_point[1]) and (start_point[1] < end_point[1]):
            point = start_point
            #print('case = 1')
        
        if (start_point[0] > center_point[0]) and (start_point[0] > end_point[0]):#
            point = start_point
            #print('case = 2')
        
        if (end_point[0] > center_point[0]) and (end_point[0] > start_point[0]):
            if abs(dif_x) > abs(dif_y):
                point = end_point
            #print('case = 3')

        #if (end_point[0] > center_point[0]) and (end_point[0] > start_point[0]):
        #    point = end_point

        command = point - center_point


    # Extract x and y commands if needed elsewhere in your code
    #command_x, command_y = command
    command = normalize_vector(np.int32(command))

    #command *= speed

    if np.linalg.norm(command) > dist:
        print("WARNING: TOO STRONG COMMAND")

    #print(f'command: {command}')
    #print("======================================")
    return command


@smart_inference_mode()
def run_seg(
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_img=False,
        ###save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        model='',
        stride ='',
        names='',
        #image_num=1,
        im0 = cv2.imread('/home/rafael2ms/Dev/crack_seg_yolov7/yolov7/myimages/crack_1.jpg')
    ):
    #imgsz = (1000,667)
    #im0 = cv2.imread('/home/rafael2ms/Dev/stopwatch_dev/image3.png')
    
    save_dir = ROOT / 'runs/predict-seg'
    #===================================
    # BRING BACK TO LIFE
    #===================================

    # =============== 26 June 2024 ==========================
    #save_path = '/home/rafael2ms/Dev/oakmax_webrtc/git_projektarbeit/frames'
    save_path = '/home/rafael2ms/Dev/oakmax_webrtc/final_pa'
    img = None #cv2.imread('/home/rafael2ms/Dev/crack_seg_yolov7/yolov7/seg/segment/PCA_Line.jpg')

    #>>>>>>>> save_path = '.'
    global last_save
    #image_num = 1
    global elapsed_pooling
    global elapsed_pca
    global elapsed_gpu_pca

    global frame_num 
    global factor
    #>>>>>>>> elapsed_pooling = []
    #>>>>>>>> elapsed_pca = 0
    #>>>>>>>> elapsed_gpu_pca = 0
    #>#
    view_img=False

    # =============== 26 June 2024 ==========================
    #save_img=False
    frame_mask = None
    
    #new_model = True
    #>#
    #save_txt=True
    seen=0

    dt = (Profile(), Profile(), Profile())

    im = letterbox(im0, imgsz, stride, auto=True)[0]  # padded resize #imsz
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    #print(" * Resized image shape:", im.shape)
    #>>>print(f"im type {type(im)}") # im type <class 'numpy.ndarray'>

    height, width = im.shape[1:3]
    #center_point = np.array([384/2, 208/2])#[imgsz[0] // 4, imgsz[1] // 4]
    center_point = np.array([width//(2*factor),height//(2*factor)])#[imgsz[0] // 4, imgsz[1] // 4]
    #print(f'* center_point = {center_point}')
    #print(f'* im0 dims = {height}, {width}')

    
    #print(f'center_point dtype = {type(center_point)}')

    #print(f'center_point2 = {center_point2}')
    #print(f'center_point2 dtype = {type(center_point2)}')
    command_x = 0
    command_y = 0
    # ===========================================================
    #>#s = "Datails: " 

    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred, out = model(im, augment=augment, visualize=False)
        proto = out[1]

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1, nm=32) #max_det
    #>#print('===========================================================')
    last_save = time.process_time()

    for i, det in enumerate(pred):  # per image
        seen += 1
        #>>>txt_path = "/home/rafael2ms/Dev/crack_seg_yolov7/yolov7/myimages/txtfile"

        #>#s += '%gx%g ' % im.shape[2:]  # print string
        #>>>gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        ###imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names)) #>#s
        if len(det):
            #det = det[det[:, 5] == 0]

            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC 
            matrix = masks[0,:,:]
            matrix2 = matrix.cpu().numpy()
            
            #># start0 = time.process_time()
            # ==================================================================
            # ==================================================================
            #>#start = time.process_time()
            #>>>>>>>>>>max_index, max_sum = highest_concentration_area(matrix2)
            #>#end = time.process_time()
            #>#print(f'Process_time HCA: {(end - start)}')  # will print the time spent on this process in seconds

            #>>>print(f"Region with highest concentration of 1s: {max_index} with sum {max_sum}")
            #>>>np.savetxt('my_file.txt', matrix2)
            
            #start = time.process_time()
            downscaled_matrix = gpu_max_pooling(matrix2, factor=factor)
            #end = time.process_time()
            #elapsed_pooling.append(end - start)
            #print(f'Process_time MPL: {(end - start)}')  # will print the time spent on this process in seconds
           
            #>>>print("Downscaled Matrix Shape:", downscaled_matrix.shape)
            #>>>np.savetxt('my_downscaled_matrix.txt', downscaled_matrix)

            # Extracting coordinates of ones
            #downscaled_matrix = matrix2
            rows, cols = np.where(downscaled_matrix == 1)
            #rows, cols = np.where(matrix2 == 1)
            points = list(zip(cols, rows))  # Notice cols come first to represent the x-axis   # <--OLD--

            #start = time.process_time()
            #line_points = principal_component_analysis(points)
            #end = time.process_time()
            #elapsed_pca = (end - start)

            #start = time.process_time()
            line_points = gpu_principal_component_analysis(points)
            #end = time.process_time()
            #elapsed_gpu_pca.append(end - start)
            #print(f'Process_time PCA: {(end - start)}')  # will print the time spent on this process in seconds

            # Assuming line_points and imgshape are updated appropriately within the loop
            command_x, command_y = process_points(line_points, center_point)
            #command_y = -abs(command_y)
            ##command_y = abs(command_y)
            
            # =============== 26 June 2024 ==========================
            # Draw the PCA line
            if save_img: # time.process_time() - last_save > 5:
                # ==================================================================
                #                   PLAYGROUND, MAY 20
                # =================================================================

                #print('____________________________________')
                #>>>print(f'RawImage Shape:{im0.shape}')
                #>>>print(f'Original Shape:{matrix2.shape}')
                #>>>print(f'Max Pool Shape:{downscaled_matrix.shape}')
                #print(f'Points Shape  :{len(points[0]),len(points[1])}')
                
                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------

                # Write results ---------------------------------------------------------------------------------------
                for *xyxy, conf, cls in reversed(det[:, :6]):  
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    print(f'names[c]: {names[c]}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()
                #cv2.imwrite(f'{save_path}/frame_{frame_num}.jpg', im0)

                # =============== 26 June 2024 ==========================
                #print("cv2.imwrite(/frame.jpg, im0)")
                frame_mask = im0
                # cv2.imwrite(f'{save_path}/frame.jpg', im0)

                # Line points ----------------------------------------------------------------------------------------
                #print(f'line points: {line_points}')
                #print(f'start point: {start_point}')
                #print(f'end point  : {end_point}')
                # ==================================================================
                # Create an image from the matrix for visualization
                img = np.uint8(downscaled_matrix * 255)  # Scale to 0-255 for visualizing
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR color space
                #imgshape = img.shape
                #center_point = (int(imgshape[1]/2), int(imgshape[0]/2))
                #print(f'img Shape  :{img.shape}')
                #print(f'center_point  :{center_point}')

                # Ensure the line endpoints are also converted to integer and in the correct order
                #start_point = (int(line_points[0][0]), int(line_points[0][1]))
                #end_point = (int(line_points[1][0]), int(line_points[1][1]))
                
                # ==================================================================
                start_point, end_point, mid_point = create_points(line_points)

                # Draw red circles at the start and end points
                circle_radius = 5  # You can adjust the size of the circle
                circle_thickness = -1  # Negative thickness makes the circle filled

                cv2.line(img, start_point, end_point, (0, 255, 0), 2)  # Green line

                #cv2.circle(img, start_point, circle_radius, (0, 0, 255), circle_thickness)
                cv2.circle(img, mid_point, circle_radius, (255, 0, 0), circle_thickness)
                #cv2.circle(img, end_point, circle_radius, (0, 0, 255), circle_thickness)

                cv2.circle(img, center_point, circle_radius, (0, 255, 255), circle_thickness)

                #height, width = img.shape[:2]
                #print(f'* pca dims = {height}, {width}')
                #print(f'* center_point = {center_point}')

                #cv2.circle(img, [192,104], circle_radius, (255, 255, 0), circle_thickness)
                #cv2.circle(img, [104,192], circle_radius, (0, 255, 255), circle_thickness)
                dist = np.linalg.norm(mid_point - center_point)
                #print(f'dist {frame_num} = {dist}')
                 # ==================================================================
                if dist > (center_point[1]*3/5):
                    cv2.arrowedLine(img, center_point, mid_point, (255, 255, 50), 3, tipLength = 0.3)  # Green line
                else:
                     # 27 June 2024 - add abs()
                    arrow_x = int(center_point[0] + 150 * command_x)
                    arrow_y = int(center_point[1] + 150 * command_y)

                    if arrow_x > img.shape[1]:
                        arrow_x = int(0.95*img.shape[1])

                    if arrow_x < 0:
                        arrow_x = 5
                    
                    if arrow_y > img.shape[0]:
                        arrow_y = int(0.95*img.shape[0])

                    if arrow_y < 0:
                        arrow_y = 5

                    next_point = (arrow_x, arrow_y)


                    #print(f'img.shape[0]: {img.shape[0]}')
                    #print(f'img.shape[1]: {img.shape[1]}')

                    #print(f'arrow_x: {arrow_x}')
                    #print(f'arrow_y: {arrow_y}')

                    #print(next_point)

                    cv2.arrowedLine(img, center_point, next_point, (0, 0, 255), 3, tipLength = 0.3)  # Green line
                 # ==================================================================
                #cv2.line(img, start_point, end_point, (0, 255, 0), 2)  # Green line

                # Display the result
                #cv2.imshow('PCA Line', img)
                #cv2.imwrite(f'{save_path}/PCA_Line_{frame_num}.jpg', img)
                
                # =============== 26 June 2024 ==========================
                ###cv2.imwrite(f'{save_path}/PCA_Line.jpg', img)
                last_save = time.process_time()
                
                frame_num += 1
                if frame_num > 50:
                    frame_num = 0
            # ==================================================================
        #break
    # =============== 26 June 2024 ==========================
    #return (command_x,command_y) #im0, img
    
    ##cv2.imwrite(f'{save_path}/frame_mask.jpg', frame_mask)
    # =============== 27 June 2024 ==========================
    return (command_x,command_y, frame_mask, img)
