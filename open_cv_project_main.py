import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
src_clicked = np.array([[0,0]],dtype=np.float32)
i = 0
warp_one_more_time = True
def mouse_click(event,x,y,args,params):
    global src_clicked , i
    if event == cv2.EVENT_LBUTTONDOWN :
        print("(",x,y,")")
        src_clicked = np.append(src_clicked,[[x,y]] , axis=0)
        src_clicked = src_clicked[np.argsort(src_clicked[:,1])]
        src12 = np.array(src_clicked[1:3],dtype = np.float32)
        src34 = np.array(src_clicked[3:5],dtype = np.float32)
        src12 = src12[np.argsort(src12[:,0])]
        src34 = src34[np.argsort(src34[:,0])[::-1]]
        if i == 3:
            src_clicked = np.append(src12,src34,axis=0)
        i+=1


#img directory 
img =cv2.imread(r"warp")

input_auto_or_custom = input("do u want to warp urself or let our code to do it? ('custom'/'code'/'skip')")
if input_auto_or_custom == "custom":
    while warp_one_more_time : 
        i = 0
        src_clicked = np.array([[0,0]],dtype=np.float32)
        cv2.imshow("img" , img)
        cv2.setMouseCallback("img",mouse_click)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q') or i == 4:
                break
        width = np.mean([np.power(np.power(src_clicked[2,0] - src_clicked[3,0],2) + np.power(src_clicked[2,1] - src_clicked[3,1],2),1/2) , np.power(np.power(src_clicked[0,0] - src_clicked[1,0],2) + np.power(src_clicked[0,1] - src_clicked[1,1],2),1/2)])
        width = int(width)

        height = np.mean([np.power(np.power(src_clicked[2,0] - src_clicked[1,0],2) + np.power(src_clicked[2,1] - src_clicked[1,1],2),1/2) , np.power(np.power(src_clicked[0,0] - src_clicked[3,0],2) + np.power(src_clicked[0,1] - src_clicked[3,1],2),1/2)])
        height = int(height)

        dst = np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
        warpedimagewh = (width,height)
        matrix = cv2.getPerspectiveTransform(src_clicked,dst)
        warped = cv2.warpPerspective(img,matrix,warpedimagewh)
    
        cv2.imshow("warped image",warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        input_warp = input("do u want to warp the img one more time? (Y/N)")
        if input_warp == "Y" or input_warp == "y":
            warp_one_more_time = True
        else:
            warp_one_more_time = False
        img = warped
elif input_auto_or_custom == "code":
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30 ,200)
    shape_edged = np.shape(edged)
    _, thresh = cv2.threshold(gray,140,200,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img,contours,-1,(0,0,0),2)
    contour_img = cv2.cvtColor(contour_img,cv2.COLOR_BGR2GRAY)
    edged_copy = np.copy(edged)
    edged_copy = np.bitwise_not(edged_copy)

    for d2 in range(0,shape_edged[1]):
        for d1 in range(0,shape_edged[0]):
            if edged[d1,d2] != contour_img[d1,d2]:
                edged_copy[d1,d2] = 0
    lines = cv2.HoughLines(edged_copy,1.8,np.pi/180,200)
    black_pic = np.zeros_like(edged_copy)
    for r_theta in lines:
        arr = np.array(r_theta[0],dtype=np.float64)
        r,theta = arr
        a = np.cos(theta)
        b= np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 +1000*(-b))
        y1 = int(y0 +1000*(a))
        x2 = int(x0 -1000*(-b))
        y2 = int(y0 -1000*(a))
        cv2.line(black_pic,(x1,y1),(x2,y2),(255,0,0),2)
    corners = cv2.goodFeaturesToTrack(black_pic,maxCorners=20 , qualityLevel= 0.01 , minDistance=10)
    corners = np.int0(corners)
    minx= np.min(np.min(corners,axis=1),axis=0)[0]
    miny= np.min(np.min(corners,axis=1),axis=0)[1]
    maxx= np.max(np.min(corners,axis=1),axis=0)[0]
    maxy= np.max(np.min(corners,axis=1),axis=0)[1]
    periodx = int(shape_edged[0]*5/100)
    periody = int(shape_edged[1]*5/100)
    cornerss = np.array([[]])
    corner1 = shape_edged
    corner2 = np.array([0,shape_edged[1]])
    corner3 = np.array([shape_edged[0],0])
    corner4 = np.array([0,0])

    for cor in corners:
        for c in cor:
            if c[1] <= miny + periody * 5:
                if c[0] < corner1[0]:
                    corner1 = c
            if c[1] <= miny + periody * 5:
                if c[0] > corner2[0]:
                    corner2 = c
            if c[1] >= maxy - periody * 5:
                if c[0] < corner3[0]:
                    corner3 = c
            if c[1] >= maxy - periody :
                if c[0] > corner4[0]:
                    corner4 = c

    corners = [corner1,corner2,corner4,corner3]
    edged_copy_color = np.dstack((edged_copy,np.zeros_like(edged_copy)))
    edged_copy_color = np.dstack((edged_copy_color,np.zeros_like(edged_copy)))
    for c in corners:
        x, y = c.ravel()
        edged_copy = cv2.circle(img,center = (x,y),radius = 5 ,color = (0,0,255) , thickness=-1)
    src_clicked = np.array(corners , dtype=np.float32)
    cv2.imshow("corners",edged_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    width = np.mean([np.power(np.power(src_clicked[2,0] - src_clicked[3,0],2) + np.power(src_clicked[2,1] - src_clicked[3,1],2),1/2) , np.power(np.power(src_clicked[0,0] - src_clicked[1,0],2) + np.power(src_clicked[0,1] - src_clicked[1,1],2),1/2)])
    width = int(width)

    height = np.mean([np.power(np.power(src_clicked[2,0] - src_clicked[1,0],2) + np.power(src_clicked[2,1] - src_clicked[1,1],2),1/2) , np.power(np.power(src_clicked[0,0] - src_clicked[3,0],2) + np.power(src_clicked[0,1] - src_clicked[3,1],2),1/2)])
    height = int(height)

    dst = np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
    warpedimagewh = (width,height)
    matrix = cv2.getPerspectiveTransform(src_clicked,dst)
    warped = cv2.warpPerspective(img,matrix,warpedimagewh)

    cv2.imshow("warped image",warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows
else:
    warped = img



resize_img = input("if u want to resize img write : ('A472' : 595x842) ('A4150' : 1240x1754) ('custom') otherwise skip  ")
if resize_img == "A472" or resize_img == "a472":
    warped = cv2.resize(warped,dsize=(595,842),interpolation=cv2.INTER_CUBIC)
elif resize_img == "A4150" or resize_img == "a4150":
    warped = cv2.resize(warped,dsize=(1240,1754),interpolation=cv2.INTER_CUBIC)
elif resize_img == "custom":
    wresize = float(input("please enter width of ur desire resize: (in pixel)"))
    hresize = float(input("please enter height of ur desire resize: (in pixel)"))
    warped =cv2.resize(warped,dsize= (wresize,hresize),interpolation=cv2.INTER_CUBIC)
cv2.imshow("warped image",warped)
cv2.waitKey(0)
cv2.destroyAllWindows
# input_crop = input("do u want to crop the picture ? (Y/N) ")
# if input_crop == "y" or input_crop == "Y":
#     is_cut = True
#     while is_cut:
#         i = 0
#         src_clicked = np.array([[0,0]],dtype=np.float32)
#         cv2.imshow("warped" , warped)
#         cv2.setMouseCallback("warped",mouse_click)
#         while True:
#             if cv2.waitKey(1) & 0xFF == ord('q') or i == 2:
#                 break
#         cuted = warped[int(src_clicked[0,0]):int(src_clicked[1,0]+1) , int(src_clicked[0,1]):int(src_clicked[1,1])+1]
#         cv2.imshow("cuted" , cuted)
#         redo_cut = input("if u want to redo the cut please type 'redo' and if u want to redo and then cut type'cut' and if u want to proceed enter nothing : ")
#         if redo_cut == "redo":
#             cuted = warped 
#         elif redo_cut == "cut":
#             is_cut = True
#         else :
#             is_cut = False
img = warped
while True:
    edit_pic = input("""choose and then type : (note: better to first make it grayscale then use suggest)
                            black and white ('b&w')
                            grayscale ('grsc')
                            ed kernel ('edkernel')
                            sobel xy ('sxy')
                            sharpen ('sharp')
                            blur ('blur')
                            redo edit ('redo')
                            sugested ('suggest')
                            skip (enter key) :
                                            """)
    if edit_pic == "grsc":
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    elif edit_pic == "b&w":
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        input_black = int(input("please enter how much black u want the pic to be (0 to 255) :"))
        ret , img = cv2.threshold(img,input_black,255,cv2.THRESH_BINARY)
    elif edit_pic == "edkernel":
        ed_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        red = cv2.filter2D(img,-1,ed_kernel)
    elif edit_pic == "sxy":
        input_sobel = input("choose and type : xy ('xy') x('x') y('y')")
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_y = np.transpose(sobel_x)
        rx = cv2.filter2D(img,-1,sobel_x)
        ry = cv2.filter2D(img,-1,sobel_y)
        rxy = np.add(rx,ry)
        if input_sobel == "xy":
            img = rxy
        elif input_sobel == "x":
            img = rx
        elif input_sobel == "y" : 
            img = ry
    elif edit_pic == "sharp":
        sharp_level = int(input("please enter how much sharp u want ur pic to be :"))
        sharp_kernel = np.array([[0,-1,0],[-1,sharp_level,-1],[0,-1,0]])
        sharp = cv2.filter2D(img,-1,sharp_kernel)
        img = sharp
    elif edit_pic == "suggest":
        ret , img = cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
        sharp_level = int(input("please enter how much sharp u want ur pic to be :"))
        sharp_kernel = np.array([[0,-1,0],[-1,sharp_level,-1],[0,-1,0]])
        sharp = cv2.filter2D(img,-1,sharp_kernel)
        img = sharp
    elif edit_pic == "blur":
        input_blur = input("choose and enter gausian ('gaus') median('med')")
        if input_blur == "gaus":
            img = cv2.GaussianBlur(img,(5,5),-1)
        elif input_blur == "med":
            img = cv2.medianBlur(img,5)
    elif edit_pic == "redo":
        img = warped
    else:
        break
    cv2.imshow("edited", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    repeat_edit = input("if u want to continue editing ('edit') otherwise please use enter to skip : ")
    if repeat_edit != "edit":
        break
