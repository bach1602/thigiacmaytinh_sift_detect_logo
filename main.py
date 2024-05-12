import cv2

img1 = cv2.imread('logokyniemdaihoc.jpg')   #logo to be identified
#img1 = cv2.imread('rooney.jpg')
img2 = cv2.imread('logodaihochanghai.jpg')  #raw image

sift = cv2.SIFT.create() # gọi phương thức SIFT_create() để tìm các điểm đặc trưng

#getting features
kp_img1, desc_img1 = sift.detectAndCompute(img1, None) # dò tìm và mô tả điểm đặc trưng cho ảnh img1 None = mask
img1_features = cv2.drawKeypoints(img1, kp_img1, img1) # vẽ các điểm đặc trưng của ảnh img1


cv2.imshow("img1 features", img1_features)

kp_img2, desc_img2 = sift.detectAndCompute(img2, None) # dò tìm và mô tả điểm đặc trưng cho ảnh img2
img2_features = cv2.drawKeypoints(img2, kp_img2, img2) # vẽ các điểm đặc trưng của ảnh img2, img2 = inImage, img2 = outImage

cv2.imshow("img2 features", img2_features)

#Feature Matching (Flann)
"""
index_params = dict(algorithm = 0, trees = 5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_img1, desc_img2, k = 2)
"""
#BF Matcher

match = cv2.BFMatcher()
matches = match.knnMatch(desc_img1,desc_img2, k=2)


#For taking only good matches

good_pts=[]
for m, n in matches:
    if m.distance < 0.7*n.distance:
       good_pts.append(m)

#Drawing Matches Made
img3 = cv2.drawMatches(img1, kp_img1, img2, kp_img2, good_pts, img1)
cv2.imshow("Matches",img3)
cv2.waitKey()

