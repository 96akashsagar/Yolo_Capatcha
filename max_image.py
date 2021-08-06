import cv2
 
img = cv2.imread('C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/test_images/2582.jpg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ', img.shape)
 
scale_percent = 1000 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()