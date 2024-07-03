# # -*- coding: utf-8 -*-
from django.shortcuts import render
from matplotlib import pyplot as plt
import pymysql
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import os

import sys
# 获取当前脚本所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 Model 目录的路径
model_dir = os.path.join(current_dir, '../Model')

# 将 Model 目录添加到 sys.path
sys.path.append(model_dir)


from PCA import PCA
import dataloader
import preprocessing

from LogClustering import LogClustering
import dataloader, preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE

# 配置数据库连接

# from django.http import HttpResponse
# from toolkit.pre_load import neo_con
# from django.http import JsonResponse
# import os

# import json
# relationCountDict = {}
# filePath = os.path.abspath(os.path.join(os.getcwd(),"."))
# with open(filePath+"/toolkit/relationStaticResult.txt","r",encoding='utf8') as fr:
# 	for line in fr:
# 		relationNameCount = line.split(",")
# 		relationName = relationNameCount[0][2:-1]
# 		relationCount = relationNameCount[1][1:-2]
# 		relationCountDict[relationName] = int(relationCount)
# def sortDict(relationDict):
# 	for i in range( len(relationDict) ):
# 		relationName = relationDict[i]['rel']['type']
# 		relationCount = relationCountDict.get(relationName)
# 		if(relationCount is None ):
# 			relationCount = 0
# 		relationDict[i]['relationCount'] = relationCount

# 	relationDict = sorted(relationDict,key = lambda item:item['relationCount'],reverse = True)

# 	return relationDict

def search_entity(request):
	# ctx = {}
	# #根据传入的实体名称搜索出关系
	# if(request.GET):
	# 	entity = request.GET['user_text']
	# 	#连接数据库
	# 	db = neo_con
	# 	entityRelation = db.getEntityRelationbyEntity(entity)
	# 	if len(entityRelation) == 0:
	# 		#若数据库中无法找到该实体，则返回数据库中无该实体
	# 		ctx= {'title' : '<h1>数据库中暂未添加该实体</h1>'}
	# 		return render(request,'entity.html',{'ctx':json.dumps(ctx,ensure_ascii=False)})
	# 	else:
	# 		#返回查询结果
	# 		#将查询结果按照"关系出现次数"的统计结果进行排序
	# 		entityRelation = sortDict(entityRelation)

	# 		return render(request,'entity.html',{'entityRelation':json.dumps(entityRelation,ensure_ascii=False)})

	return render(request,"entity.html")

def search_relation(request):
	db = pymysql.connect(
		host='localhost',
		user='root',
		password='140166',
		database='HDFS',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor,
	)
	cursor = db.cursor()
	# 查询数据库
	cursor.execute("SELECT * FROM uploaded_files")
	rows = cursor.fetchall()
    
    # 将数据传递给模板
	context = {
        'rows': rows
    }
	db.close()
	#动态显示已经传到数据库的日志文件
	return render(request,'relation.html',context) 

@csrf_exempt
def data_analysis(request):
	if request.method == 'POST':
		db = pymysql.connect(
			host='localhost',
			user='root',
			password='140166',
			database='HDFS',
			charset='utf8mb4',
			cursorclass=pymysql.cursors.DictCursor,
		)
		data = json.loads(request.body)
		# 算法类型
		# 1: PCA 2: clustering 3: Decision Tree 4: Random Forest 
		# 5: SVM 6: Transformer 7: BERT 8: LSTM
		option = data.get('option')
		fileid = data.get('fileid')

		# 读文件
		cursor = db.cursor()
		# 查询数据库
		cursor.execute(f"select filename, file_content from uploaded_files where id={fileid}")
		all_data = cursor.fetchall()
		name = all_data[0]['filename']
		content = all_data[0]['file_content']
		print(name)
		# print(content)
		#打印当前路径
		print(os.getcwd())
		save_path = os.path.join('./data/HDFS', name)
		save_path=os.path.abspath(save_path)
		print(save_path)
		with open(save_path, 'wb') as f:
			f.write(content)
		
		
		# 处理数据
		if option == 1:
			# PCA
			(x_train, _), (x_test, _), _ = dataloader.load_HDFS(save_path,label_file=None,
																		window='session',
																		train_ratio=0.5,
																		split_type='uniform')
			feature_extractor = preprocessing.FeatureExtractor()
			x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
													normalization='zero-mean')
			x_test = feature_extractor.transform(x_test)

			model = PCA(n_components=2)
			model.fit(x_train)
			spe_threshold=model.threshold
			print(spe_threshold)

			y_pred = model.predict(x_test)

			print(y_pred)
			result = {
            'x_test': x_test.tolist(),
            'y_pred': y_pred.tolist(),
            'spe_threshold': spe_threshold  # assuming spe_threshold is defined
        	}
		elif option == 2:
			max_dist = 0.3  # the threshold to stop the clustering process
			anomaly_threshold = 0.3  # the threshold for anomaly detection
			max_samples = 1000  # maximum number of samples to plot

			(x_train, _), (x_test, _),_ = dataloader.load_HDFS(save_path,
                                                                label_file=None,
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')
			feature_extractor = preprocessing.FeatureExtractor()
			x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
			x_test = feature_extractor.transform(x_test)

			model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
			model.fit(x_train)

			y_pred=model.predict(x_test)
			pca = sklearnPCA(n_components=3)
			x_pca = pca.fit_transform(x_test)

			tsne = TSNE(n_components=3, init='pca', learning_rate='auto', random_state=42)
			x_tsne = tsne.fit_transform(x_test)

			result = {
				'x_pca': x_pca.tolist(),
				'x_tsne': x_tsne.tolist(),
				'y_pred': y_pred.tolist()
			}

		db.close()
		# response_data = {
		# 	'message': 'Data analysised',
		# 	'x_test': x_test.tolist(),
		# 	'y_pred': y_pred.tolist(),
		# 	'spe_threshold': spe_threshold
		# }
		return JsonResponse(result)
	return JsonResponse({'message': 'Only POST method is allowed'}, status=400)
@csrf_exempt
def delete_file(request):
	if request.method == 'POST':
		db = pymysql.connect(
			host='localhost',
			user='root',
			password='140166',
			database='HDFS',
			charset='utf8mb4',
			cursorclass=pymysql.cursors.DictCursor,
		)
		data = json.loads(request.body)
		fileid = data.get('fileid')
		cursor = db.cursor()
		# 查询数据库
		cursor.execute(f"delete from uploaded_files where id={fileid}")
		db.commit()
		db.close()
		return JsonResponse({'message': 'File deleted'})
	return JsonResponse({'message': 'Only POST method is allowed'}, status=400)


@csrf_exempt
def show_content(request):
	if request.method == 'POST':
		db = pymysql.connect(
			host='localhost',
			user='root',
			password='140166',
			database='HDFS',
			charset='utf8mb4',
			cursorclass=pymysql.cursors.DictCursor,
		)
		data = json.loads(request.body)
		fileid = data.get('fileid')
		cursor = db.cursor()
		# 查询数据库
		cursor.execute(f"select file_content from uploaded_files where id={fileid}")
		all_data = cursor.fetchall()
		content = all_data[0]['file_content']
		#从content中读取100个字符
		content = content[:500]
		content = content.decode('utf-8')
		print(content)
		db.close()
		return JsonResponse({'content': content})
	return JsonResponse({'message': 'Only POST method is allowed'}, status=400)