import numpy as np

def null_space(a, rtol=1e-5):
    _, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

def convolution(img, kernel):
    res = img.copy()
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1, 'Wrong Kernel Size!'
    pass

def rotate(img, angle):
    pass

def canny(img, t1, t2):
    pass

def draw_line(img, x1, y1, x2, y2):
    pass

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
        H = null_space(p_Set)[1]
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
    b_vec = null_space(V_Set)[1]
    B = [b_vec[0],b_vec[1],b_vec[2],b_vec[1],b_vec[3],b_vec[4],b_vec[2],b_vec[4],b_vec[5]]
    B = np.array(B)
    B = B.reshape(3,3)
    A = np.linalg.cholesky(B)
    K = np.linalg.inv(A.T)
    K /= K[2][2]
    return K