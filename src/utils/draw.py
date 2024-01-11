import cv2


def drawPerson(frame,bbox_left,bbox_top,bbox_w,bbox_h,tracked_name,classColor):
    linewidth = min(int(bbox_w *0.15), int(bbox_h *0.15))
    
    cv2.rectangle(frame, (bbox_left, bbox_top),(bbox_w,bbox_h), color=classColor, thickness=1)
    cv2.putText(frame, f"{tracked_name}", (bbox_left, bbox_top-3), cv2.FONT_HERSHEY_COMPLEX, 1, color=classColor, thickness=2)
    ###### Top Left
    cv2.line(frame,(bbox_left, bbox_top),(bbox_left+linewidth, bbox_top),color=classColor,thickness=3)
    cv2.line(frame,(bbox_left, bbox_top),(bbox_left, bbox_top+linewidth),color=classColor,thickness=3)
    ###### Top Right
    cv2.line(frame,(bbox_w, bbox_top),(bbox_w -linewidth, bbox_top),color=classColor,thickness=3)
    cv2.line(frame,(bbox_w, bbox_top),(bbox_w, bbox_top +linewidth),color=classColor,thickness=3)
    ###### Bottom Left
    cv2.line(frame,(bbox_left, bbox_h),(bbox_left+linewidth, bbox_h),color=classColor,thickness=3)
    cv2.line(frame,(bbox_left, bbox_h),(bbox_left, bbox_h -linewidth),color=classColor,thickness=3)
    ###### Bottom Right
    cv2.line(frame,(bbox_w, bbox_h),( bbox_w -linewidth, bbox_h),color=classColor,thickness=3)
    cv2.line(frame,(bbox_w, bbox_h),( bbox_w, bbox_h -linewidth),color=classColor,thickness=3)

