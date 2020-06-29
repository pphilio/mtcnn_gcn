import os
import cv2

filename=[]
face_order=[]
bb=[]
face_box=[0,0,0,0]

for f_name in os.listdir('./output/after_mtcnn'):
    if f_name.startswith('bounding'):
        bounding_boxes_filename = './output/after_mtcnn/'+f_name
with open(bounding_boxes_filename,"r") as f:
    lines = f.readlines()
    for a in lines:
        a=a.split(' ')
        face_order.append(a[0].split('\\')[-1])
        a[0]=a[0].split('\\')[-1].split('.jpg')[0]
        filename.append(a[0].split('_')[0])
        a[-1]=a[-1].split('\n')[0]
        a[1:]=map(int,a[1:])
        bb.append(a[1:])
        print(filename)
        print(face_order)
        print(bb)


print(bb)
#filename은 사실 필요없음 후에 지울예정(face_order에서 추출하면됨)

impath='./input/images/'
print(impath+filename[0])
img=cv2.imread(impath+filename[0]+'.jpg',cv2.IMREAD_COLOR)
print('\n\n')
drawpath='./output/after_mtcnn/draws/'

cropped_img=[]
roi=[]
i=0
for face in face_order:
    print(face)

    cropped_img.append(cv2.imread(drawpath+face,cv2.IMREAD_COLOR))
    i+=1
i=0
print(bb)
for box in bb:
    print(box[0])
    print(box[2])
    print(box[1])
    print(box[3])
    #cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,255),3)

    #cv2.circle(img,(box[0],box[1]),1,(255,0,0))
    #cv2.circle(img,(box[0],box[3]),1,(255,0,0))
    x=25/178
    print("2-0")
    print(box[2]-box[0])
    y=45/218
    face_box[0]=box[0]+int((box[2]-box[0])*x)
    face_box[2]=box[2]-int((box[2]-box[0])*x)
    face_box[1]=box[1]+int((box[3]-box[1])*y)
    face_box[3]=box[3]-int((box[3]-box[1])*y)
    print(face_box)


    cropped_img[i]=cv2.resize(cropped_img[i],(98,95))
    #cv2.rectangle(img,(face_box[0],face_box[1]),(face_box[2],face_box[3]),(255,0,255),1)

    #cv2.circle(img,(face_box[0],face_box[1]),2,(255,0,0))
    #cv2.circle(img,(face_box[0],face_box[3]),2,(255,0,0))


    img[face_box[1]:face_box[1]+95,face_box[0]:face_box[0]+98]=cropped_img[i]

    cv2.imshow('image', img[box[1]:box[3],box[0]:box[2]])
    cv2.imshow('crop',cropped_img[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i+=1

    cv2.imwrite('saveimage.png', img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()