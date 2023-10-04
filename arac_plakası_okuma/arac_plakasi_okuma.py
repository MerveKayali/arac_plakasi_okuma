import cv2
import numpy as np
import pytesseract
import imutils

img=cv2.imread("licence_plate.jpg")

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

gray=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
#Kenarları korurken görüntüleri yumuşatmak ve gürültüyü azaltmak için
# bilateralFilter() kullanılır
filtered=cv2.bilateralFilter(gray,6,250,250 )#çap sigma color ve sigma space parametreleri
edged=cv2.Canny(filtered,30,200)#eğer filtrelemeden gray imgye uygulsaydık daha ayrıntılı karışık bir görsel olurdu bu şekilde ayrıntıları azaltıp plkaya odaklandık


contours=cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(contours)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:10]#kordinatları  0'dan 10'a kadar olanları alanlarına göre sıralıyoruz
screen=None

#bu kısımda daha önceden kanıtlanmış formüller ile şekli bozuk dörtgenleri yakalıyoruz
for c in cnts:
    epsilon=0.018*cv2.arcLength(c,True)
    approx= cv2.approxPolyDP(c,epsilon,True)
    if len(approx)==4:# 4 köşe tespit edildiyse dikdörtgen var demektir
        screen=approx
        break

mask=np.zeros(gray.shape,np.uint8)#siyah bir ekran oluşturuyoruz
new_img=cv2.drawContours(mask,[screen],0,(255,255,255),-1)#elimizde olan dörtgen kordinatlarını bu siyah ekranda beyaza çeviriyoruz
new_img=cv2.bitwise_and(img,img,mask=mask)#arabayı yapıştırıyoruz ama sadece plaka alanı açık kalıyor

(x,y)=np.where(mask==255)#beyaz olan yerlerin kordinatlarını aldık
(topx,topy)=(np.min(x),np.min(y))
(bottomx,bottomy)=(np.max(x),np.max(y))
cropped=gray[topx:bottomx+1,topy:bottomy+1]

text=pytesseract.image_to_string(cropped,lang="eng")
print("detected text",text)


cv2.imshow("mask",cropped)

"""
cv2.imshow("gray",gray)
cv2.imshow("filtered",filtered)
cv2.imshow("edged",edged)"""
cv2.waitKey(0)
cv2.destroyAllWindows()