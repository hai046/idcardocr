# -*- coding: utf-8 -*-
import cgi
import json
import os
import socketserver
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import time

import findidcard
import idcardocr


def process(img_name):
    try:
        idfind = findidcard.findidcard()
        idcard_img = idfind.find(img_name)
        result_dict = dict()
        result_dict["data"] = idcardocr.idcardocr(idcard_img)
        result_dict['code'] = 0
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
        result = process("tmp/%s.jpg" % filename)
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
