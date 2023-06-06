def menu_visible(image, playable_images, image_dict, img_index):
    image[image_dict["menu"]].set_visible()
    image[image_dict["back"]].set_visible()
    playable_images[img_index].set_visible()


def menu_un_visible(image, playable_images, image_dict, img_index):
    image[image_dict["menu"]].set_un_visible()
    image[image_dict["back"]].set_un_visible()
    playable_images[img_index].set_un_visible()


def loading_visible(gif, txt, gif_index, txt_index):
    gif[gif_index].set_visible()
    txt[txt_index].set_visible()


def loading_un_visible(gif, txt, gif_index, txt_index):
    gif[gif_index].set_un_visible()
    txt[txt_index].set_un_visible()


def update_acupuncture_txt(ap_img, txt, txt_dict, acupuncture_info, ap_index, now_index, acupuncture_txt):
    line_list = ["line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8"]  # line5  # 23、15
    close_acupuncture_txt(ap_img, txt, txt_dict)
    way, summary, types, massage, content, name = acupuncture_info
    txt[txt_dict["acupuncture_title"]].set_txt(acupuncture_txt)
    txt[txt_dict["acupuncture_title"]].set_visible()
    short_len = 15
    long_len = 23

    if ap_index == now_index:
        update_ap_txt(txt, txt_dict, line_list, way, short_len, start_line=0, end_line=4)
        update_ap_txt(txt, txt_dict, line_list, summary, long_len, start_line=5, end_line=7)
        ap_img.set_visible()
        ap_img.reset_img("UI/acupuncture/" + acupuncture_txt + ".png")

    if ap_index + 1 == now_index:
        update_ap_txt(txt, txt_dict, line_list, massage, long_len, start_line=0, end_line=4)
        update_ap_txt(txt, txt_dict, line_list, types, long_len, start_line=5, end_line=7)

    if ap_index + 2 == now_index or ap_index + 3 == now_index:  # set this two part!
        muti_page_update_ap_txt(txt, txt_dict, content, 2, long_len, "；", [ap_index + 2, ap_index + 3], now_index)

    if ap_index + 4 == now_index:
        update_ap_txt(txt, txt_dict, line_list, name, long_len)


def muti_page_update_ap_txt(txt, txt_dict, t_str, pages, split_len, split_char, acupuncture_index_list, now_index):
    # t_str: the long content, acupuncture_index_list is regarding to pages
    line_list = ["line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8"]
    content_list = t_str.split(split_char)
    content_len = len(content_list)
    cl = content_len  # judge how to two page
    for _ in range(pages):
        start_l = 0
        start = content_len - cl
        for i in range(start, content_len):
            end_l = len(content_list[i]) // split_len + 1 + start_l
            if end_l <= 7 and (cl > 1 or _ == pages - 1):
                if acupuncture_index_list[_] == now_index:
                    start_l = update_ap_txt(txt, txt_dict, line_list, content_list[i], split_len,
                                            start_line=start_l, end_line=end_l)
                else:
                    start_l = update_ap_txt(txt, txt_dict, line_list, content_list[i], split_len,
                                            start_line=start_l, end_line=end_l, unable=True)
            else:
                break
            cl -= 1


def update_ap_txt(txt, txt_dict, line_list, t_str, t_len, start_line=0, end_line=7, unable=False):
    down = 0
    up = t_len
    ln = len(t_str) // t_len

    for i in range(start_line, end_line + 1):
        ln -= 1
        if ln >= 0:
            if not unable:
                txt[txt_dict[line_list[i]]].set_txt(t_str[down:up])
                txt[txt_dict[line_list[i]]].set_visible()
            down = up
            up += t_len
        elif ln == -1:
            if not unable:
                txt[txt_dict[line_list[i]]].set_txt(t_str[down:len(t_str)])
                txt[txt_dict[line_list[i]]].set_visible()
            return i


def close_acupuncture_txt(ap_img, txt, txt_dict):
    line_list = ["line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8"]
    txt[txt_dict["acupuncture_title"]].set_un_visible()
    ap_img.set_un_visible()
    for i in line_list:
        txt[txt_dict[i]].set_un_visible()


def classifyArea_visible(image, txt, img_index, txt_index, disease_txt):
    image[img_index].set_visible()
    txt[txt_index].set_visible()
    txt[txt_index].set_txt(disease_txt)


def classifyArea_un_visible(image, txt, img_index, txt_index):
    image[img_index].set_un_visible()
    txt[txt_index].set_un_visible()


def enqueue(list, value):
    list.append(value)


def dequeue(list):
    del list[0]


def playNextImage(dynG_txt, image, index):
    img = image[index]
    if dynG_txt == "next":
        n_index = img.get_next_image_index()
        if n_index == -1:
            return index
        img.set_un_visible()
        image[n_index].set_visible()
        return n_index
    if dynG_txt == "back":
        b_index = img.get_back_image_index()
        if b_index == -1:
            return index
        img.set_un_visible()
        image[b_index].set_visible()
        return b_index
    return index


def SetGestureImageOn(gesture_txt, image):
    img_dict = {"left": 1, "choose": 2, "right": 3, "menu": 4, "cancel": 5}
    for k, v in img_dict.items():
        if k == gesture_txt:
            image[v].set_visible()
        else:
            image[v].set_un_visible()


gesture_list = ["no gesture", "no gesture", "no gesture"]  # for 5 time
labels = ("cancel", "choose", "menu", "right", "left", "no gesture")


def pre_txt2real_txt(pre_txt):
    dequeue(gesture_list)
    enqueue(gesture_list, pre_txt)
    labels_times = [0, 0, 0, 0, 0, 0]
    for i in gesture_list:
        for c in range(len(labels)):
            if i == labels[c]:
                labels_times[c] += 1
                if pre_txt == labels[c]:
                    labels_times[c] += 1
    max_n = 0
    max_l = "no gesture"
    count = 0

    for i in labels_times:
        if count > 4:
            break
        if i > max_n:
            max_n = i
            max_l = labels[count]
        count += 1
    return max_l
