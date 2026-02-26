import numpy as np
import cv2
import os
import natsort

ROOT_DIR = "" ## Your root directory where download this code --- CHANGE THIS

#==============================================================================
# GUIDED FILTER - Edge-Preserving Smoothing
#==============================================================================

class GuidedFilter:

    # def __init__(self, I, radius=5, epsilon=0.4):
    def __init__(self, I, radius, epsilon):

        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

        # print('radius',self._radius)
        # print('epsilon',self._epsilon)

    def _toFloatImg(self, img):
        if img.dtype == np.float32:
            return img
        return ( 1.0 / 255.0 ) * np.float32(img)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps                                       
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean                                  
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean                                  
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps                            
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean                                  
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps                                                       


        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var                                                      
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var                                                      
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var                                                      
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var                                                      
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var                                                      
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var                                                      
        
        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var                                    
        Irr_inv /= I_cov                                                                                     
        Irg_inv /= I_cov                                                                                     
        Irb_inv /= I_cov                                                                                     
        Igg_inv /= I_cov                                                                                     
        Igb_inv /= I_cov                                                                                     
        Ibb_inv /= I_cov                                                                                     
        
        self._Irr_inv = Irr_inv                                                                              
        self._Irg_inv = Irg_inv                                                                              
        self._Irb_inv = Irb_inv                                                                              
        self._Igg_inv = Igg_inv                                                                              
        self._Igb_inv = Igb_inv                                                                              
        self._Ibb_inv = Ibb_inv                  

    def _computeCoefficients(self, p):
        r = self._radius                                                             
        I = self._I                                                                 
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]                                                          

        p_mean = cv2.blur(p, (r, r))                             
        Ipr_mean = cv2.blur(Ir * p, (r, r))                                                         
        Ipg_mean = cv2.blur(Ig * p, (r, r))                                                    
        Ipb_mean = cv2.blur(Ib * p, (r, r))             

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean                                                 
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean                                                     
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean                                                       
                                                                                                                 
        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov                 
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov                
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov    

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean                                                                                                                                         

        ar_mean = cv2.blur(ar, (r, r))          
        ag_mean = cv2.blur(ag, (r, r))                                                                   
        ab_mean = cv2.blur(ab, (r, r))                                                                      
        b_mean = cv2.blur(b, (r, r))                                                                                                                                              

        return ar_mean, ag_mean, ab_mean, b_mean            

    def _computeOutput(self, ab, I):
    
        ar_mean, ag_mean, ab_mean, b_mean = ab
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
        return q

    def filter(self, p):

        p_32F = self._toFloatImg(p)

        ab = self._computeCoefficients(p)
        return self._computeOutput(ab, self._I)

#==============================================================================
# ATTENUATION ESTIMATION
#==============================================================================

def get_attenuation(image, gamma=1.2):
    # Split image into R, G, B channels (OpenCV uses BGR order)
    image_b = image[:,:,0]  # Blue channel
    image_g = image[:,:,1]  # Green channel
    image_r = image[:,:,2]  # Red channel

    # Compute attenuation using exponential decay model: A = 1 - I^gamma
    attenuation_b = 1 - image_b**(gamma)
    attenuation_g = 1 - image_g**(gamma)
    attenuation_r = 1 - image_r**(gamma)

    # Compute mean attenuation for each channel
    mean_att_b = np.mean(attenuation_b)
    mean_att_g = np.mean(attenuation_g)
    mean_att_r = np.mean(attenuation_r)

    # Sort channels by attenuation (ascending order)
    list_att = [mean_att_b, mean_att_g, mean_att_r]
    index = np.argsort(list_att)  # [least attenuated, medium, most attenuated]

    return index

def getMAxChannel(img):
    imgGray = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            localMax = 0
            for k in range(0, 2):
                if img.item((i, j, k)) > localMax:
                    localMax = img.item((i, j, k))
            imgGray[i, j] = localMax
    return imgGray

def getMaxChannel(img, blockSize):
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    imgMiddle = np.zeros((newHeight, newWidth),'float64')
    imgMiddle[:, :] = 0
    imgMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = img
    # print('imgMiddle',imgMiddle)
    imgDark = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMax = 0
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    if imgMiddle.item((k, l)) > localMax:
                        localMax = imgMiddle.item((k, l))
            imgDark[i - addSize, j - addSize] = localMax
    return imgDark

#==============================================================================
# DEPTH MAP AND TRANSMISSION ESTIMATION
#==============================================================================

def DepthMap(img, blockSize, index_att):
    # Separate channels
    img_c = np.zeros(img[:,:,0:2].shape)
    img_c_star = img[:,:,index_att[-1]]  # Least attenuated channel (e.g., blue)
    img_c[:,:,0] = img[:,:,index_att[0]]  # Most attenuated channel
    img_c[:,:,1] = img[:,:,index_att[1]]  # Medium attenuated channel

    # Apply maximum filter to each
    max_c_star = getMaxChannel(img_c_star, blockSize)  # Max of least attenuated
    img_c = getMAxChannel(img_c)  # Combine most & medium channels
    max_c  = getMaxChannel(img_c, blockSize)  # Max of most attenuated

    # Depth proportional to difference
    largestDiff = max_c_star  - max_c
    return largestDiff

def  Refinedtransmission(transmission,img):
    gimfiltR = 50   # Filter radius
    eps = 0.001     # Regularization parameter

    guided_filter = GuidedFilter(img, gimfiltR, eps)
    transmission = guided_filter.filter(transmission)
    transmission = np.clip(transmission, 0, 255)

    return transmission

class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

#==============================================================================
# BACKGROUND LIGHT (ATMOSPHERIC LIGHT) ESTIMATION
#==============================================================================

def getAtomsphericLight(depth, img):
    height = len(depth)
    width = len(depth[0])

    # Create list of all pixels with their depth values
    nodes = []
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, depth[i, j])
            nodes.append(oneNode)

    # Sort by depth (ascending), select pixel with smallest depth
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]

    return atomsphericLight

def estimateBackgroundLight(image):

    index = get_attenuation(image)
    largestDiff = DepthMap(image, 9, index)
    atomsphericLight = getAtomsphericLight(largestDiff, image)
    return atomsphericLight

#==============================================================================
# MAIN SCRIPT - Process All Images in Dataset
#==============================================================================

if __name__ == '__main__':
    # Configuration
    dataset_dirs = [d.name for d in ROOT_DIR.iterdir() if d.is_dir()]
    for folder in dataset_dirs:
        base_dir = os.path.join(str(ROOT_DIR), folder)
        
        img_input_path = os.path.join(base_dir, "images")
        
        if not os.path.exists(img_input_path):
            print(f"Skipping: {img_input_path} does not exist.")
            continue

        files = os.listdir(img_input_path)
        files = natsort.natsorted(files)

        t_prior_dir = os.path.join(base_dir, 't_prior')
        B_prior_dir = os.path.join(base_dir, 'B_prior')

        os.makedirs(t_prior_dir, exist_ok=True)
        os.makedirs(B_prior_dir, exist_ok=True)
        print(f"Directory ready: {t_prior_dir}")
        print(f"Directory ready: {B_prior_dir}")

        # Process each image
        for i in range(len(files)):
            file = files[i]
            filepath = os.path.join(img_input_path, file)
            prefix = file.split('.')[0]

            if os.path.isfile(filepath):
                print(f'Processing file: {file}')

                # Load images
                img = cv2.imread(filepath)
                if img is None:
                    print(f"Failed to load {file}")
                    continue

                image = img / 255.0
                index = get_attenuation(image)
                largestDiff = DepthMap(image, 7, index)
                
                transmission = largestDiff + (1 - np.max(largestDiff))
                transmission = np.clip(transmission, 0.1, 0.9) 
                
                transmission = Refinedtransmission(transmission * 255, image * 255)
                
                BackgroundLight = estimateBackgroundLight(image)
                D = np.ones(image.shape)
                B = D * BackgroundLight

                # Step 6: Save priors
                t_out_file = os.path.join(t_prior_dir, f"{prefix}.jpg")
                B_out_file = os.path.join(B_prior_dir, f"{prefix}.jpg")
                
                cv2.imwrite(t_out_file, np.uint8(transmission))
                cv2.imwrite(B_out_file, np.uint8(B * 255))

    print("\nAll processing complete!")
    print(f"Transmission priors saved to: {t_prior_dir}")
    print(f"Background light priors saved to: {B_prior_dir}")