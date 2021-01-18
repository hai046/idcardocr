# -*- coding: utf-8 -*-
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pytesseract
from PIL import Image

rate = 1.0
global_height = 1280.00 / rate
global_width = 3840.00 / rate
x = global_height / global_width
pixel_x = int(x * global_width)
# print(x, pixel_x)

mask_color = (255, 255, 255)

debug = False


class idcardocr:

    def callback_ocr(self, v):
        self.result_dict[v[0]] = v[1]

    # mode0:识别姓名，出生日期，身份证号； mode1：识别所有信息
    def ocr(self, imgname, mode=1):
        print(u'进入身份证光学识别流程...')
        if mode == 1:
            # generate_mask(x)
            t1 = round(time.time() * 1000)
            img_data_gray, img_org = self.img_resize_gray(imgname)
            t2 = round(time.time() * 1000)
            print("================ cost1=", (t2 - t1))
            t1 = t2

            # result_dict = dict()

            name_left_top, name_right_bottom = self.find_name(img_data_gray, img_org)
            address_pic_left_top, address_pic_right_bottom = self.find_address(img_data_gray, img_org)
            sex_pic_left_top, sex_pic_right_bottom = self.find_sex(img_data_gray, img_org)
            nation_pic_left_top, nation_pic_right_bottom = self.find_nation(img_data_gray, img_org)
            t2 = round(time.time() * 1000)
            print("============== cost2=", (t2 - t1))
            t1 = t2
            # print(name_left_top, name_right_bottom)
            # print("sex", sex_pic_left_top, sex_pic_right_bottom)
            # print(address_pic_left_top, address_pic_right_bottom)

            x_arrays = []
            x_arrays.append(name_left_top[0])
            x_arrays.append(sex_pic_left_top[0])
            x_arrays.append(address_pic_left_top[0])

            x_arrays = sorted(x_arrays)
            # print(x_arrays, x_arrays[1])
            name_right_bottom = (name_right_bottom[0] + (x_arrays[1] - name_left_top[0]), name_right_bottom[1])
            name_left_top = (x_arrays[1], name_left_top[1])

            # sex_pic_left_top = (x_arrays[1], sex_pic_left_top[1])
            sex_pic_right_bottom = (
                sex_pic_right_bottom[0] + (x_arrays[1] - sex_pic_left_top[0]), sex_pic_right_bottom[1])
            sex_pic_left_top = (x_arrays[1], sex_pic_left_top[1])

            # address_pic_left_top = (x_arrays[1], address_pic_left_top[1])
            address_pic_right_bottom = (
                address_pic_right_bottom[0] + (x_arrays[1] - address_pic_left_top[0]), address_pic_right_bottom[1])
            address_pic_left_top = (x_arrays[1], address_pic_left_top[1])
            # 姓名、性别、地址 对其，去除识别位置错误产生的影响
            t2 = round(time.time() * 1000)
            print("============== ready pool    =", (t2 - t1))
            t1 = t2
            p = Pool(5)
            # self.queue.put((
            #     self.get_mat(img_data_gray, img_org, name_left_top, name_right_bottom), "name",))
            name = p.apply_async(self.get_name, args=(
                self.get_mat_data(img_data_gray, img_org, name_left_top, name_right_bottom),))
            address = p.apply_async(self.get_address, args=(
                self.get_mat_data(img_data_gray, img_org, address_pic_left_top, address_pic_right_bottom),))
            sex = p.apply_async(self.get_sex, args=(
                self.get_mat_data(img_data_gray, img_org, sex_pic_left_top, sex_pic_right_bottom),))
            nation = p.apply_async(self.get_nation, args=(
                self.get_mat_data(img_data_gray, img_org, nation_pic_left_top, nation_pic_right_bottom),))

            id_pic_left_top, id_pic_right_bottom = self.find_idnum(img_data_gray, img_org)
            idnum = p.apply_async(self.get_idnum_and_birth, args=(
                self.get_mat_data(img_data_gray, img_org, id_pic_left_top, id_pic_right_bottom),))

            t2 = round(time.time() * 1000)
            print("==============  pool start   =", (t2 - t1))
            t1 = t2

            p.close()
            p.join()
            result_dict = dict()
            result_dict["name"] = name.get()
            result_dict["address"] = address.get()
            result_dict["sex"] = sex.get()
            result_dict["nation"] = nation.get()
            result_dict["idnum"] = idnum.get()
            result_dict["birth"] = result_dict["idnum"][6:14]
            t2 = round(time.time() * 1000)
            print("==============  pool   =", (t2 - t1))

            if debug:
                self.showimg(img_data_gray)
            return result_dict
        elif mode == 0:
            # generate_mask(x)
            img_data_gray, img_org = self.img_resize_gray(imgname)
            result_dict = dict()
            name_pic = self.find_name(img_data_gray, img_org)
            # showimg(name_pic)
            # print 'name'
            result_dict['name'] = self.get_name(name_pic)
            # print result_dict['name']

            idnum_pic = self.find_idnum(img_data_gray, img_org)
            # showimg(idnum_pic)
            # print 'idnum'

            result_dict['idnum'], result_dict['birth'] = self.get_idnum_and_birth(idnum_pic)
            result_dict['sex'] = ''
            result_dict['nation'] = ''
            result_dict['address'] = ''

        else:
            print(u"模式选择错误！")

        # showimg(img_data_gray)
        return result_dict

    # def print(*args):
    #     pass

    def generate_mask(self, x):
        name_mask_pic = cv2.UMat(cv2.imread('name_mask.jpg'))
        sex_mask_pic = cv2.UMat(cv2.imread('sex_mask.jpg'))
        nation_mask_pic = cv2.UMat(cv2.imread('nation_mask.jpg'))
        birth_mask_pic = cv2.UMat(cv2.imread('birth_mask.jpg'))
        year_mask_pic = cv2.UMat(cv2.imread('year_mask.jpg'))
        month_mask_pic = cv2.UMat(cv2.imread('month_mask.jpg'))
        day_mask_pic = cv2.UMat(cv2.imread('day_mask.jpg'))
        address_mask_pic = cv2.UMat(cv2.imread('address_mask.jpg'))
        idnum_mask_pic = cv2.UMat(cv2.imread('idnum_mask.jpg'))
        name_mask_pic = self.img_resize_x(name_mask_pic)
        sex_mask_pic = self.img_resize_x(sex_mask_pic)
        nation_mask_pic = self.img_resize_x(nation_mask_pic)
        birth_mask_pic = self.img_resize_x(birth_mask_pic)
        year_mask_pic = self.img_resize_x(year_mask_pic)
        month_mask_pic = self.img_resize_x(month_mask_pic)
        day_mask_pic = self.img_resize_x(day_mask_pic)
        address_mask_pic = self.img_resize_x(address_mask_pic)
        idnum_mask_pic = self.img_resize_x(idnum_mask_pic)
        cv2.imwrite('name_mask_%s.jpg' % pixel_x, name_mask_pic)
        cv2.imwrite('sex_mask_%s.jpg' % pixel_x, sex_mask_pic)
        cv2.imwrite('nation_mask_%s.jpg' % pixel_x, nation_mask_pic)
        cv2.imwrite('birth_mask_%s.jpg' % pixel_x, birth_mask_pic)
        cv2.imwrite('year_mask_%s.jpg' % pixel_x, year_mask_pic)
        cv2.imwrite('month_mask_%s.jpg' % pixel_x, month_mask_pic)
        cv2.imwrite('day_mask_%s.jpg' % pixel_x, day_mask_pic)
        cv2.imwrite('address_mask_%s.jpg' % pixel_x, address_mask_pic)
        cv2.imwrite('idnum_mask_%s.jpg' % pixel_x, idnum_mask_pic)

    # 用于生成模板
    def img_resize_x(self, imggray):
        # print ('dheight:%s' % dheight)
        crop = imggray
        size = crop.get().shape
        dheight = int(size[0] * x / rate)
        dwidth = int(size[1] * x / rate)
        # print("dheight=",dheight," dwidth=",dwidth," size=",size)
        crop = cv2.resize(src=crop, dsize=(dwidth, dheight), interpolation=cv2.INTER_CUBIC)
        return crop

    # idcardocr里面resize以高度为依据, 用于get部分
    def img_resize(self, imggray, dheight):
        # print 'dheight:%s' % dheight
        crop = imggray
        size = crop.get().shape
        height = size[0]
        width = size[1]
        width = width * dheight / height
        crop = cv2.resize(src=crop, dsize=(int(width), dheight), interpolation=cv2.INTER_CUBIC)
        return crop

    def img_resize_gray(self, imgorg):
        # imgorg = cv2.imread(imgname)
        crop = imgorg
        size = cv2.UMat.get(crop).shape
        # print size
        height = size[0]
        width = size[1]
        # 参数是根据3840调的
        height = int(height * global_width * x / width)
        # print (height)
        crop = cv2.resize(src=crop, dsize=(int(global_width * x), height), interpolation=cv2.INTER_CUBIC)
        return self.hist_equal(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), crop

    template_mao = {}

    def get_mask(self, name):
        if name in self.template_mao:
            return self.template_mao[name]
        template = cv2.UMat(cv2.imread(name, 0))
        w, h = cv2.UMat.get(template).shape[::-1]
        self.template_mao[name] = {template, w, h}
        return template, w, h
        pass

    def find_name(self, crop_gray, crop_org):
        # template = cv2.UMat(cv2.imread('name_mask_%s.jpg' % pixel_x, 0))
        # # showimg(crop_org)
        # w, h = cv2.UMat.get(template).shape[::-1]
        template, w, h = self.get_mask('name_mask_%s.jpg' % pixel_x)
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        print(w, h, min_val, max_val, min_loc, max_loc)
        # print(max_loc)
        top_left = (max_loc[0] + int(w - 10 / rate), max_loc[1] - int(20 * x))
        # top_left = (max_loc[0] + w, max_loc[1] - int(20 * x/ rate))
        bottom_right = ((top_left[0] + int(700 / rate * x)), (top_left[1] + int(300 / rate * x)))
        # result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # print(top_left, bottom_right)
        # # showimg(result)
        # return cv2.UMat(result)
        return top_left, bottom_right

    def find_sex(self, crop_gray, crop_org):
        # template = cv2.UMat(cv2.imread('sex_mask_%s.jpg' % pixel_x, 0))
        # # showimg(template)
        # w, h = cv2.UMat.get(template).shape[::-1]
        template, w, h = self.get_mask('sex_mask_%s.jpg' % pixel_x)
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (int(top_left[0] + int(300 * x) / rate), int(top_left[1] + int(300 * x) / rate))
        # result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # print(top_left, bottom_right)
        # # showimg(crop_gray)
        # # showimg(result)
        # return cv2.UMat(result)
        return top_left, bottom_right

    def find_nation(self, crop_gray, crop_org):
        # template = cv2.UMat(cv2.imread('nation_mask_%s.jpg' % pixel_x, 0))
        #
        # # showimg(template)
        # w, h = cv2.UMat.get(template).shape[::-1]

        template, w, h = self.get_mask('nation_mask_%s.jpg' % pixel_x)

        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w - int(20 * x), max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(500 * x / rate), top_left[1] + int(300 * x / rate))
        # result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # print(top_left, bottom_right)
        # # showimg(crop_gray)
        # return cv2.UMat(result)
        return top_left, bottom_right

    # def find_birth(crop_gray, crop_org):
    #         template = cv2.UMat(cv2.imread('birth_mask_%s.jpg'%pixel_x, 0))
    #         # showimg(template)
    #         w, h = cv2.UMat.get(template).shape[::-1]
    #         res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    #         #showimg(crop_gray)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #         top_left = (max_loc[0] + w, max_loc[1] - int(20*x))
    #         bottom_right = (top_left[0] + int(1500*x), top_left[1] + int(300*x))
    #         # 提取result需要在rectangle之前
    #         date_org = cv2.UMat.get(crop_org)[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #         date = cv2.cvtColor(date_org, cv2.COLOR_BGR2GRAY)
    #         cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
    #         # cv2.imwrite('date.png',date)
    #
    #         # 提取年份
    #         template = cv2.UMat(cv2.imread('year_mask_%s.jpg'%pixel_x, 0))
    #         year_res = cv2.matchTemplate(date, template, cv2.TM_CCOEFF_NORMED)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(year_res)
    #         bottom_right = (max_loc[0]+int(20*x), int(300*x))
    #         top_left = (0, 0)
    #         year = date_org[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #         # cv2.imwrite('year.png',year)
    #         cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
    #
    #         # 提取月
    #         template = cv2.UMat(cv2.imread('month_mask_%s.jpg'%pixel_x, 0))
    #         month_res = cv2.matchTemplate(date, template, cv2.TM_CCOEFF_NORMED)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(month_res)
    #         bottom_right = (max_loc[0]+int(40*x), int(300*x))
    #         top_left = (max_loc[0] - int(220*x), 0)
    #         month = date_org[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #         # cv2.imwrite('month.png',month)
    #         cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
    #
    #         # 提取日
    #         template = cv2.UMat(cv2.imread('day_mask_%s.jpg'%pixel_x, 0))
    #         day_res = cv2.matchTemplate(date, template, cv2.TM_CCOEFF_NORMED)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(day_res)
    #         bottom_right = (max_loc[0]+int(20*x), int(300*x))
    #         top_left = (max_loc[0] - int(220*x), 0)
    #         day = date_org[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #         # cv2.imwrite('day.png',day)
    #         cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
    #         showimg(crop_gray)
    #         return cv2.UMat(year), cv2.UMat(month), cv2.UMat(day)

    def find_address(self, crop_gray, crop_org):
        # template = cv2.UMat(cv2.imread('address_mask_%s.jpg' % pixel_x, 0))
        # # showimg(template)
        # # showimg(crop_gray)
        # w, h = cv2.UMat.get(template).shape[::-1]
        template, w, h = self.get_mask('address_mask_%s.jpg' % pixel_x)
        # t1 = round(time.time()*1000)
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        # t2 = round(time.time()*1000)
        # print 'time:%s'%(t2-t1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(1760 * x / rate), top_left[1] + int(544 * x / rate))
        # result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # print(top_left, bottom_right)
        # showimg(crop_gray)
        # return cv2.UMat(result)
        return top_left, bottom_right

    def find_idnum(self, crop_gray, crop_org):
        # template = cv2.UMat(cv2.imread('idnum_mask_%s.jpg' % pixel_x, 0))
        # # showimg(template)
        # # showimg(crop_gray)
        # w, h = cv2.UMat.get(template).shape[::-1]
        template, w, h = self.get_mask('idnum_mask_%s.jpg' % pixel_x)
        res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
        bottom_right = (top_left[0] + int(2340 * x), top_left[1] + int(300 * x))
        # result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        # cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # showimg(crop_gray)
        # return cv2.UMat(result)
        return top_left, bottom_right

    def get_mat_data(self, crop_gray, crop_org, top_left, bottom_right):
        result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
        cv2.rectangle(crop_gray, top_left, bottom_right, mask_color, 2)
        # return cv2.UMat(result)
        return result

    def showimg(self, img):
        cv2.namedWindow("contours", 0);
        cv2.resizeWindow("contours", 1280, 720);
        cv2.imshow("contours", img)
        # cv2.waitKey()

    # psm model:
    #  0    Orientation and script detection (OSD) only.
    #  1    Automatic page segmentation with OSD.
    #  2    Automatic page segmentation, but no OSD, or OCR.
    #  3    Fully automatic page segmentation, but no OSD. (Default)
    #  4    Assume a single column of text of variable sizes.
    #  5    Assume a single uniform block of vertically aligned text.
    #  6    Assume a single uniform block of text.
    #  7    Treat the image as a single text line.
    #  8    Treat the image as a single word.
    #  9    Treat the image as a single word in a circle.
    #  10    Treat the image as a single character.
    #  11    Sparse text. Find as much text as possible in no particular order.
    #  12    Sparse text with OSD.
    #  13    Raw line. Treat the image as a single text line,
    # 			bypassing hacks that are Tesseract-specific

    def get_name(self, img):
        #    cv2.imshow("method3", img)
        #    cv2.waitKey()
        # fix can't pickle 'cv2.UMat' object
        img = cv2.UMat(img)
        _, _, red = cv2.split(img)  # split 会自动将UMat转换回Mat
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 50)
        #    red = cv2.medianBlur(red, 3)
        red = self.img_resize(red, 150)
        img = self.img_resize(img, 150)
        # showimg(red)
        # cv2.imwrite('name.png', red)
        #    img2 = Image.open('address.png')
        # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        # return get_result_vary_length(red, 'chi_sim', img, '-psm 7')
        result = self.get_result_vary_length(red, 'chi_sim', img, '--psm 7')
        # self.result_dict[name] = result
        # print(result_dict)
        return result
        # return punc_filter(pytesseract.image_to_string(img, lang='chi_sim', config='-psm 13').replace(" ",""))

    def get_sex(self, img):
        img = cv2.UMat(img)
        _, _, red = cv2.split(img)
        print('sex')
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
        #    red = cv2.medianBlur(red, 3)
        #    cv2.imwrite('address.png', img)
        #    img2 = Image.open('address.png')
        red = self.img_resize(red, 150)
        # cv2.imwrite('sex.png', red)
        # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        # return get_result_fix_length(red, 1, 'sex', '--psm 10').replace("\n", "").replace("\f", "")
        return self.get_result_fix_length(red, 1, 'sex', '--psm 10').replace("\n", "").replace("\f", "")

        # return get_result_fix_length(red, 1, 'chi_sim', '--psm 10')
        # return pytesseract.image_to_string(img, lang='sex', config='-psm 10').replace(" ","")

    def get_nation(self, img):
        img = cv2.UMat(img)
        _, _, red = cv2.split(img)
        print('nation')
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
        red = self.img_resize(red, 150)
        # cv2.imwrite('nation.png', red)
        # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        return self.get_result_fix_length(red, 1, 'nation', '--psm 10').replace("\n", "").replace("\f", "")
        # return get_result_fix_length(red, 1, 'chi_sim', '--psm 10')
        # return pytesseract.image_to_string(img, lang='nation', config='-psm 13').replace(" ","")

    # def get_birth(year, month, day):
    #         _, _, red = cv2.split(year)
    #         red = cv2.UMat(red)
    #         red = hist_equal(red)
    #         red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    #         red = img_resize(red, 150)
    #         # cv2.imwrite('year_red.png', red)
    #         year_red = red
    #
    #         _, _, red = cv2.split(month)
    #         red = cv2.UMat(red)
    #         red = hist_equal(red)
    #         red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    #         #red = cv2.erode(red,kernel,iterations = 1)
    #         red = img_resize(red, 150)
    #         # cv2.imwrite('month_red.png', red)
    #         month_red = red
    #
    #         _, _, red = cv2.split(day)
    #         red = cv2.UMat(red)
    #         red = hist_equal(red)
    #         red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    #         red = img_resize(red, 150)
    #         # cv2.imwrite('day_red.png', red)
    #         day_red = red
    #         # return pytesseract.image_to_string(img, lang='birth', config='-psm 7')
    #         return get_result_fix_length(year_red, 4, 'eng', '-c tessedit_char_whitelist=0123456789 -psm 13'), \
    #                get_result_vary_length(month_red, 'eng', '-c tessedit_char_whitelist=0123456789 -psm 13'), \
    #                get_result_vary_length(day_red, 'eng', '-c tessedit_char_whitelist=0123456789 -psm 13')

    def get_address(self, img):
        # _, _, red = cv2.split(img)get_address
        # red = cv2.medianBlur(red, 3)
        img = cv2.UMat(img)
        _, _, red = cv2.split(img)
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
        red = self.img_resize(red, 300)
        # img = img_resize(img, 300)
        # cv2.imwrite('address_red.png', red)
        img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        # return punc_filter(get_result_vary_length(red,'chi_sim', img, '-psm 6'))
        return self.punc_filter(self.get_result_vary_length(red, 'chi_sim', img, '--psm 6'))
        # return punc_filter(pytesseract.image_to_string(img, lang='chi_sim', config='-psm 3').replace(" ",""))

    def get_idnum_and_birth(self, img):
        img = cv2.UMat(img)
        _, _, red = cv2.split(img)
        print('idnum')
        red = cv2.UMat(red)
        red = self.hist_equal(red)
        red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
        red = self.img_resize(red, 150)
        # cv2.imwrite('idnum_red.png', red)
        # idnum_str = get_result_fix_length(red, 18, 'idnum', '-psm 8')
        # idnum_str = get_result_fix_length(red, 18, 'eng', '--psm 8 ')
        img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
        idnum_str = self.get_result_vary_length(red, 'eng', img, '--psm 8 ')
        # return idnum_str, idnum_str[6:14]
        return idnum_str

    def get_result_fix_length(self, red, fix_length, langset, custom_config=''):
        # return pytesseract.image_to_string(cv2.UMat.get(red), lang=langset, config=custom_config)
        t1 = round(time.time() * 1000)

        red_org = red

        cv2.fastNlMeansDenoising(red, red, 4, 7, 35)
        rec, red = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
        image, contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        # 描边一次可以减少噪点
        cv2.drawContours(red, contours, -1, (0, 255, 0), 1)
        color_img = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
        # showimg(color_img)

        # for x, y, w, h in contours:
        #     imgrect = cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # showimg(color_img)

        h_threshold = 54
        numset_contours = []
        calcu_cnt = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > h_threshold:
                numset_contours.append((x, y, w, h))
        while len(numset_contours) != fix_length:
            if calcu_cnt > 50:
                print(u'计算次数过多！目前阈值为：', h_threshold)
                break
            numset_contours = []
            calcu_cnt += 1
            if len(numset_contours) > fix_length:
                h_threshold += 1
                contours_cnt = 0
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > h_threshold:
                        contours_cnt += 1
                        numset_contours.append((x, y, w, h))
            if len(numset_contours) < fix_length:
                h_threshold -= 1
                contours_cnt = 0
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > h_threshold:
                        contours_cnt += 1
                        numset_contours.append((x, y, w, h))
        result_string = ''
        numset_contours.sort(key=lambda num: num[0])
        print(numset_contours)
        for x, y, w, h in numset_contours:
            result_string += pytesseract.image_to_string(cv2.UMat.get(red_org)[y - 10:y + h + 10, x - 10:x + w + 10],
                                                         lang=langset, config=custom_config)
        # print(new_r)
        # cv2.imwrite('fixlengthred.png', cv2.UMat.get(red_org)[y-10:y + h +10 , x-10:x + w + 10])
        print(result_string)
        return result_string

    def get_result_vary_length(self, red, langset, org_img, custom_config=''):
        # return pytesseract.image_to_string(cv2.UMat.get(red), lang=langset, config=custom_config)
        red_org = red
        # cv2.fastNlMeansDenoising(red, red, 4, 7, 35)
        rec, red = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
        image, contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        # 描边一次可以减少噪点
        cv2.drawContours(red, contours, -1, (255, 255, 255), 1)
        # color_img = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
        numset_contours = []
        height_list = []
        width_list = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height_list.append(h)
            # print(h,w)
            width_list.append(w)
        height_list.remove(max(height_list))
        width_list.remove(max(width_list))
        height_threshold = 0.70 * max(height_list)
        width_threshold = 1.4 * max(width_list)
        # print('height_threshold:'+str(height_threshold)+'width_threshold:'+str(width_threshold))
        big_rect = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > height_threshold and w < width_threshold:
                # print(h,w)
                numset_contours.append((x, y, w, h))
                big_rect.append((x, y))
                big_rect.append((x + w, y + h))
        big_rect_nparray = np.array(big_rect, ndmin=3)
        x, y, w, h = cv2.boundingRect(big_rect_nparray)
        # imgrect = cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # showimg(imgrect)
        # showimg(cv2.UMat.get(org_img)[y:y + h, x:x + w])

        result_string = ''
        result_string += pytesseract.image_to_string(cv2.UMat.get(red_org)[y - 10:y + h + 10, x - 10:x + w + 10],
                                                     lang=langset,
                                                     config=custom_config)
        print(result_string)
        # cv2.imwrite('varylength.png', cv2.UMat.get(org_img)[y:y + h, x:x + w])
        # cv2.imwrite('varylengthred.png', cv2.UMat.get(red_org)[y:y + h, x:x + w])
        # numset_contours.sort(key=lambda num: num[0])
        # for x, y, w, h in numset_contours:
        #     result_string += pytesseract.image_to_string(cv2.UMat.get(color_img)[y:y + h, x:x + w], lang=langset, config=custom_config)
        return self.punc_filter(result_string)

    def punc_filter(self, str):
        temp = str
        xx = u"([\u4e00-\u9fff0-9A-Z]+)"
        pattern = re.compile(xx)
        results = pattern.findall(temp)
        string = ""
        for result in results:
            string += result
        return string

    # 这里使用直方图拉伸，不是直方图均衡
    def hist_equal(self, img):
        # clahe_size = 8
        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(clahe_size, clahe_size))
        # result = clahe.apply(img)
        # test

        # result = cv2.equalizeHist(img)

        image = img.get()  # UMat to Mat
        # result = cv2.equalizeHist(image)
        lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
        # lut = np.zeros(256)
        hist = cv2.calcHist([image],  # 计算图像的直方图
                            [0],  # 使用的通道
                            None,  # 没有使用mask
                            [256],  # it is a 1D histogram
                            [0, 256])
        minBinNo, maxBinNo = 0, 255
        # 计算从左起第一个不为0的直方图柱的位置
        for binNo, binValue in enumerate(hist):
            if binValue != 0:
                minBinNo = binNo
                break
        # 计算从右起第一个不为0的直方图柱的位置
        for binNo, binValue in enumerate(reversed(hist)):
            if binValue != 0:
                maxBinNo = 255 - binNo
                break
        # print minBinNo, maxBinNo
        # 生成查找表
        for i, v in enumerate(lut):
            if i < minBinNo:
                lut[i] = 0
            elif i > maxBinNo:
                lut[i] = 255
            else:
                lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)
        # 计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表
        # print lut
        result = cv2.LUT(image, lut)
        # print type(result)
        # showimg(result)
        return cv2.UMat(result)


if __name__ == "__main__":
    # debug = True
    t1 = round(time.time() * 1000)
    id = idcardocr()
    idocr = id.ocr(cv2.UMat(cv2.imread('/Users/denghaizhu/Downloads/name1.jpg')))
    # idocr = id.jpg(cv2.UMat(cv2.imread('/Users/denghaizhu/Downloads/name2.png')))
    # idocr = idcardocr(cv2.UMat(cv2.imread('/Users/denghaizhu/Downloads/xiaobao.jpg')))
    # idocr = idcardocr(cv2.UMat(cv2.imread('/Users/denghaizhu/Downloads/haizhu.png')))
    print(idocr)
    t2 = round(time.time() * 1000)
    print("耗时：", (t2 - t1))
    if debug:
        cv2.waitKey()
    # for i in range(15):
    #     idocr = idcardocr(cv2.UMat(cv2.imread('testimages/%s.jpg'%(i+1))))
    #     print(idocr['idnum'])
