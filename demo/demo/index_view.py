# -*- coding: utf-8 -*-
#from django.http import HttpResponse
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import csrf

# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
# from .Model import UploadedFile
import csv
import io
import pymysql



def index(request):  # index页面需要一开始就加载的内容写在这里
	context = {}
	return render(request, 'index.html', context)

def upload_file(request):
    # print(111,'\n')
    # print(request.method,'\n')
    # print(request.FILES,'\n')
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        file_content = uploaded_file.read()
        file_description=request.POST['description']
        print(file_description)

        # 连接到 MySQL 数据库并插入记录
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='140166',
            database='hdfs',
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            max_allowed_packet=64*1024*1024
        )
        cursor = db.cursor()
        length=len(file_content)
        #转成KB大小
        size = length/1024
        print(size)
        # 执行SQL语句
        cursor.execute("INSERT INTO uploaded_files (filename, file_size,file_content,file_description) VALUES (%s,%s, %s,%s)", (uploaded_file.name, size,file_content,file_description))
        # 提交到数据库执行
        db.commit()

        # 获取插入的记录ID
        record_id = cursor.lastrowid

        # 关闭游标和连接
        cursor.close()
        db.close()

        return JsonResponse({'message': '文件上传成功'})
    return JsonResponse({'error': '没有上传文件'}, status=400)