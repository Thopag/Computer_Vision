from code.utils.ROI import roi
import cv2 as cv

def do_trick2_1obj(img , mask ):
       
    #INIT
    result = img # at start
    status = False  # False =  no change were applied 
    #Then compute a Region Of interest around the needed object 
    
    out= roi(mask , s_erod=10 , s_dil=200)
    new_image, roi_mask  = out[0] , out[1]

    # we want to detect the red wand 
    
    RGB_red     = [255,0,10] #  # those parameter are tested and work correctly but could be adapted 
    lower_red   = [50 , 50]
    upper_red   = [255,255]
    # detect red color
    mask_red   = detect_color(new_image,RGB_red,lower_red,upper_red, tuning = 25) 
    
    #some filtering again on the red mask
    SE= cv.getStructuringElement(cv.MORPH_RECT,(s_erod,s_erod))
    eroded_mask_red = cv.erode(mask_red,SE)
    SE= cv.getStructuringElement(cv.MORPH_ELLIPSE,(s_dil,s_dil))
    dilated_mask_red= cv.dilate(eroded_mask_red,SE)
    
    # now we count how much white pixel we have on the new mask 
    n_pixel = np.shape(np.where(dilated_mask_red == 255))[1]
    #print("detect pixel = " , n_pixel)
    #print("threshold" , threshold)
    
    if n_pixel < threshold  :
        print("there is NOT enough pixel to consider  the wand as detected so we do not change the color")
        return result  , status
    
    # if we are here it means that we detected the wand 
    
    print("there is  enough pixel to consider the wand  as detected to apply the trick 2")

    
    # now we start the changing color process
     
    orig_rgb   = RGB
    target_rgb = [200,5,6] # to be changed !!!!!!
    
    img_test = change_color_mask(new_image,orig_rgb, target_rgb , dilated_mask_object)
    print("img_test have a shape of ",np.shape(img_test))
    
    # here we inverse the roi mask to keep in the original image the unprocessed part
    mask_inv = cv.bitwise_not(roi_mask)
    img_not_processed = cv.bitwise_and(img,img, mask=mask_inv)
    
    if not np.shape(img_not_processed)== np.shape(img_test):
        print("There is something wrong with the shape of img_not_processed and img_test")
        return result , status
    
    # If everything is ok , we combine the processed and unprocessed part and return the result + a true status 
    combined = cv2.add(img_test, img_not_processed)
    status = True
    print(" the combination of the processed and unprocessed part work successfully")
    return combined  , status 