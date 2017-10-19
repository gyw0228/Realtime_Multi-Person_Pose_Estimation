def get_useful_kpt_num(list, kind):
    num = 0
    for i in range(0, len(list) / 3):
        if list[i*3 + 2] <= kind:
            num += 1
    return num

def get_area(bbox): #x1, y1, x2, y2
    if bbox[2]- bbox[0] > 0:
        width = bbox[2] - bbox[0]
    else:
        width = 0
    if(bbox[3] - bbox[1]) > 0:
        height = bbox[3] - bbox[1]
    else:
        height = 0

    return width * height
