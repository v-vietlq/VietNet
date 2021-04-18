import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from torchvision import transforms
# See the Qubvel GitHub: https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp 

# In[1]: Overlay a segmentation map with an input image (Soan's code)
def visualize(seg_map, img):
    """
    Overlay the segmentation map onto the input color image img
    :param seg_map: segmentation map of size H x W
    :param img:     color image of size H x W x 3
    :return:        image overlaid the color segmentation map
    """
    # Generate the segmentation map in the RGB color with the color code
    # Class 0: black; Class 1: Green; Class 2: Red
    COLOR_CODE = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    segmap_rgb = np.zeros(img.shape)
    for k in np.unique(seg_map):
        segmap_rgb[seg_map == k] = COLOR_CODE[k]
    segmap_rgb = (segmap_rgb * 255).astype('uint8')

    # Super-impose the color segmentation map onto the color image
    overlaid_img = cv2.addWeighted(img, 1, segmap_rgb, 0.9, 0)

    return overlaid_img


# In[2]: Main
if __name__ == '__main__':
    # Parse arguments from the command line
    # Example: python predict_video.py -v test.mov
    #          python predict_video.py --video test.mov
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, 
                        help='Full path of an input video')
    args = parser.parse_args()


    # 1. Set-up CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # 2. Load the trained model
    encoder_name = 'mobilenet_v2'
    num_classes  = 3 # background vs. road vs. marker
    weight_path  = './checkpoints/Unet_epoch_43_acc_0.9696.pt'
    
    # Create a new segmentation model
    model = smp.Unet(encoder_name=encoder_name,
                       classes=num_classes,
                       activation=None,
                       encoder_weights='imagenet')  
    # Move the model to the device
    model = model.to(device) 
    
    # Load the pretrained weight to the model
    model.load_state_dict(torch.load(weight_path, device))
    model.eval()
    print('Loading the trained model done')
    

    # 3. Process the input video
    in_video  = cv2.VideoCapture(args.video)
    fourcc    = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    seg_video = cv2.VideoWriter(args.video.replace('.mov', '_out.mp4'),
                                fourcc, 24, (2 * 640, 640))

    while True:
        # 3.1.  Read a single frame from the video
        result, frame = in_video.read()

        # Quit if end of video is reached
        if not result:
            break

        # Convert to the RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 3.2. Convert the frame to a PIL image
        frame = Image.fromarray(frame)

        # 3.3. Apply transforms and convert the image to a Pytorch tensor
        frame = transforms.Resize((640, 640), 
                                  interpolation=Image.NEAREST)(frame)

        # Note that the class "ToTensor()" already includes the normalization
        # [0,1] and the channel-first conversion
        frame_tensor = T.ToTensor()(frame).unsqueeze(dim=0).to(device)
    
        # 3.4. Perform a forward pass    
        logits = model(frame_tensor)
    
        # 3.5.  Produce a segmentation map from the logits   
        # Remove the first dimension (i.e. batch size of 1)
        logits  = logits.squeeze(0) 
        # Detach from graph and convert to a Numpy array
        logits  = logits.cpu().detach().numpy()
        # Get the segmentation map
        seg_map = np.argmax(logits, axis=0)

        # 3.6. Visualize the segmentation map
        overlaid_img = visualize(seg_map, np.asarray(frame))
    
    
        # 3.7. Combine the input image with the overlaid image
        combined_img = np.concatenate((np.asarray(frame), overlaid_img), 
                                      axis=1)
    
    
        # 3.8. Save the output frame
        seg_video.write(combined_img)
        
        # 3.9.(Optional) Early break if ESC is pressed
        if cv2.waitKey(1) & 0xff == 27:
            seg_video.release()
            in_video.release()
            cv2.destroyAllWindows()
            break

    # 4. Close input and output video files
    seg_video.release()
    in_video.release()
    cv2.destroyAllWindows()
