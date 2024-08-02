import cv2
from easy_ViTPose import VitInference


# set is_video=True to enable tracking in video inference
# be sure to use VitInference.reset() function to reset the tracker after each video
# There are a few flags that allows to customize VitInference, be sure to check the class definition

# model = "apt36k"
model = "ap10k"

model_path = f'./vitpose-h-{model}.pth'
yolo_path = './yolov8s.pt'

# If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
# If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
# dataset and det_class parameters can be inferred from the ckpt name, but you can specify them.
model = VitInference(model_path, yolo_path, model_name='h', yolo_size=96, is_video=False, device="cuda",det_class="animals",dataset=model)


def gen_skeleton(img_path,save_name):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
    # If is_video=True the IDs will be consistent among the ordered video frames.
    keypoints = model.inference(img)
    
    print(keypoints)

    # call model.reset() after each video

    img = model.draw(show_yolo=True)  # Returns RGB image with drawings
    cv2.imwrite(save_name,img)

    model.reset()
    # cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); cv2.waitKey(0)


if __name__=="__main__":
    import os
    for file in os.listdir("./pics/gt")[:1]:
        file_dir = os.path.join("./pics/gt",file)
        print(file_dir)
        gen_skeleton(file_dir,f"./pics/res/gt/{file}")