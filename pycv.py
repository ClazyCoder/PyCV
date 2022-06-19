import numpy as np
import math
import utils.utils as utils

def convolution(img, kernel):
    assert len(img.shape) == 2, 'Input 1-Channel image only!'
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, 'Wrong Kernel Size!'
    kernel = np.flipud(np.fliplr(kernel))
    pad = kernel.shape[0] // 2
    value = np.sum(kernel)
    new_image = np.pad(img,((pad,pad),(pad,pad)),'constant',constant_values=0)
    new_image = new_image.astype(np.float64)
    new_image = np.abs(new_image)
    new_image /= np.max(new_image)
    sub_matrices = np.lib.stride_tricks.as_strided(new_image,
                                                   shape = tuple(np.subtract(new_image.shape, kernel.shape)+1)+kernel.shape, 
                                                   strides = new_image.strides * 2)
    if value > 0:
        return np.einsum('ij,klij->kl', kernel, sub_matrices) / value
    else:
        return np.einsum('ij,klij->kl', kernel, sub_matrices)

def rotate(img, angle):
    rad = -angle * math.pi / 180.0
    if len(img.shape) == 3:
        new_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(3):
                    cy = img.shape[0] // 2
                    cx = img.shape[1] // 2
                    y = i - cy
                    x = j - cx
                    newY = y * math.cos(rad) + x * math.sin(rad) + cy
                    newX = y * -math.sin(rad) + x * math.cos(rad) + cx
                    alpha = 1 - (newX-math.floor(newX))
                    beta = 1 - (newY-math.floor(newY))
                    newX = round(newX)
                    newY = round(newY)
                    if newX >=0 and newX < img.shape[1]-1 and newY >=0 and newY < img.shape[0]-1:
                        f1 = (1-alpha)*img.item(newY,newX,k)+alpha*img.item(newY,newX+1,k)
                        f2 = (1-alpha)*img.item(newY+1,newX,k)+alpha*img.item(newY+1,newX+1,k)
                        f3 = (1-beta)*f1+beta*f2
                        new_img.itemset(i,j,k,int(f3))
    elif len(img.shape) == 2:
        new_img = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cy = img.shape[0] // 2
                cx = img.shape[1] // 2
                y = i - cy
                x = j - cx
                newY = y * math.cos(rad) + x * math.sin(rad) + cy
                newX = y * -math.sin(rad) + x * math.cos(rad) + cx
                alpha = 1 - (newX-math.floor(newX))
                beta = 1 - (newY-math.floor(newY))
                newX = round(newX)
                newY = round(newY)
                if newX >=0 and newX < img.shape[1]-1 and newY >=0 and newY < img.shape[0]-1:
                    f1 = (1-alpha)*img.item(newY,newX)+alpha*img.item(newY,newX+1)
                    f2 = (1-alpha)*img.item(newY+1,newX)+alpha*img.item(newY+1,newX+1)
                    f3 = (1-beta)*f1+beta*f2
                    new_img.itemset(i,j,int(f3))
    return new_img

def canny(img, t1, t2):
    pass

def draw_line(img, x1, y1, x2, y2):
    pass

def threshold(img, T, max=255, min=0):
    assert len(img.shape) == 2, 'Input 1-Channel image only!'
    new_img = np.zeros(img.shape)
    new_img[img >= T] = max
    new_img[img < T] = min
    return new_img

def otsu(img, max=255, min=0):
    assert len(img.shape) == 2, 'Input 1-Channel image only!'
    img_hist = np.zeros(256)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            img_hist[img.item(j,i)] += 1
    img_hist /= img.shape[0] * img.shape[1]
    ave = 0
    for i in range(256):
        ave += i * img_hist[i]
    T = 0
    pw = img_hist[0]
    pu = 0
    best = 0
    for i in range(1,256):
        w = pw + img_hist[i]
        if w == 0 or w == 1:
            continue
        u1 = (pw * pu + i * img_hist[i] ) / w
        u2 = (ave - w * u1) / (1-w)
        t = (w*(1-w))*((u1 - u2)**2)
        if t >= best:
            best = t
            T = i
        pw = w
        pu = u1
    new_img = np.zeros(img.shape)
    new_img[img >= T] = max
    new_img[img < T] = min
    return new_img

def calibrate_camera(img_points, obj_points):
    H_Set = []
    for img_ps, obj_ps in zip(img_points, obj_points):
        p_Set = []
        for img_p, obj_p in zip(img_ps, obj_ps):
            px = [-obj_p[0], -obj_p[1],-1,0,0,0,img_p[0]*obj_p[0],img_p[0]*obj_p[1],img_p[0]]
            py = [0,0,0,-obj_p[0],-obj_p[1],-1,img_p[1]*obj_p[0],img_p[1]*obj_p[1],img_p[1]]
            p_Set.append(px)
            p_Set.append(py)
        p_Set = np.array(p_Set)
        H = utils.null_space(p_Set)[1]
        H_Set.append(H)
    V_Set = []
    for H in H_Set:
        H = np.reshape(H,(-1,3))
        v1 = [H[0][0]*H[0][1], H[0][0]*H[1][1] + H[1][0]*H[0][1], H[2][0]*H[0][1] + H[0][0]*H[2][1], H[1][0]*H[1][1], H[2][0]*H[1][1] + H[1][0]*H[2][1], H[2][0]*H[2][1]]
        v2 = [H[0][0]*H[0][0], H[0][0]*H[1][0] + H[1][0]*H[0][0], H[2][0]*H[0][0] + H[0][0]*H[2][0], H[1][0]*H[1][0], H[2][0]*H[1][0] + H[1][0]*H[2][0], H[2][0]*H[2][0]]
        v3 = [H[0][1]*H[0][1], H[0][1]*H[1][1] + H[1][1]*H[0][1], H[2][1]*H[0][1] + H[0][1]*H[2][1], H[1][1]*H[1][1], H[2][1]*H[1][1] + H[1][1]*H[2][1], H[2][1]*H[2][1]]
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        V_Set.append(v1)
        V_Set.append(v2-v3)
    V_Set = np.array(V_Set)
    b_vec = utils.null_space(V_Set)[1]
    B = [b_vec[0],b_vec[1],b_vec[2],b_vec[1],b_vec[3],b_vec[4],b_vec[2],b_vec[4],b_vec[5]]
    B = np.array(B)
    B = B.reshape(3,3)
    A = np.linalg.cholesky(B)
    K = np.linalg.inv(A.T)
    K /= K[2][2]
    return K