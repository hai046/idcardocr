# -*- coding: utf-8 -*-
import cgi
import json
import os
import socketserver
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

from findidcard import findidcard
from idcardocr import idcardocr


def process(img_name, use_find_card=False):
    try:
        if use_find_card:
            idfind = findidcard()
            idcard_img = idfind.find(img_name)
        else:
            idcard_img = cv2.UMat(cv2.imread(img_name))
        result_dict = dict()
        idocr = idcardocr()
        result_dict["data"] = idocr.ocr(idcard_img)
        result_dict['code'] = 200
    except Exception as e:
        result_dict = {'code': 1, 'message': "%s" % e}
        print(e)
    return result_dict


# SocketServer.ForkingMixIn, SocketServer.ThreadingMixIn
class ForkingServer(socketserver.ForkingMixIn, HTTPServer):
    pass


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # self.end_headers()

    def do_GET(self):
        self._set_headers()
        # self.wfile.write("<html><body><h1>hi!</h1></body></html>")

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        if not self.path.startswith('/api/v1/ocr'):
            result = {"code": 404, "message": "404"}
            self._set_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return
            # content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        # post_data = self.rfile.read(content_length) # <--- Gets the data itself
        t1 = round(time.time() * 1000)
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        print(pdict)
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        multipart_data = cgi.parse_multipart(self.rfile, pdict)
        filename = uuid.uuid1()
        fo = open("tmp/%s.jpg" % filename, "wb")
        # print(str(multipart_data))
        # print(multipart_data.get('pic')[0])
        fo.write(multipart_data.get('pic')[0])
        fo.close()
        path = "tmp/%s.jpg" % filename
        result = process("tmp/%s.jpg" % filename, use_find_card=False)
        if result['code'] != 200 or ('data' in result and len(result['data']) < 5):
            print("没有识别到信息，重新识别")
            result = process("tmp/%s.jpg" % filename, use_find_card=True)
            result['findCard'] = True

        t2 = round(time.time() * 1000)
        result['costTime'] = (t2 - t1)
        # print result
        self._set_headers()
        self.send_header("Content-Length", str(len(json.dumps(result).encode('utf-8'))))
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))
        os.remove(path)


def http_server(server_class=ForkingServer, handler_class=S, port=8588):
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    cv2.ocl.setUseOpenCL(False)
    print('Starting httpd %d...' % port)
    print(u"是否启用OpenCL：%s" % cv2.ocl.useOpenCL())
    httpd.serve_forever()


if __name__ == "__main__":
    # cv2.ocl.setUseOpenCL(True)
    # process("/Users/denghaizhu/Downloads/haizhu.png")
    # process("/Users/denghaizhu/Downloads/2.png")
    # p = Pool()
    # r9 = p.apply_async(process, args=('9.jpg',))
    # r14 = p.apply_async(process, args=('14.jpg',))
    # p.apply_async(http_server)
    # p.apply_async(http_server)
    # p.apply_async(http_server)
    # p.apply_async(http_server)
    # p.close()
    # p.join()
    # print r9.get(), r14.get()
    http_server()
    # cv2.ocl.setUseOpenCL(True)
    # t1 = round(time.time() * 1000)
    # for i in range(1,15):
    #     print(process('./testimages/%s.jpg'%i))
    # t2 = round(time.time() * 1000)
    # print('time:%s' % (t2 - t1))
