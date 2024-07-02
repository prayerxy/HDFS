# # -*- coding: utf-8 -*-
from django.shortcuts import render
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


# 配置数据库连接
db = pymysql.connect(
	host='localhost',
	user='root',
	password='140166',
	database='HDFS',
	charset='utf8mb4',
	cursorclass=pymysql.cursors.DictCursor,
)
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

	cursor = db.cursor()
	# 查询数据库
	cursor.execute("SELECT * FROM uploaded_files")
	rows = cursor.fetchall()
    
    # 将数据传递给模板
	context = {
        'rows': rows
    }
	#动态显示已经传到数据库的日志文件
	return render(request,'relation.html',context) 

@csrf_exempt
def data_analysis(request):
	if request.method == 'POST':
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
		# print(name)
		# print(content)
		save_path = os.path.join('D:\大三下学习\项目\HDFS\demo\data\HDFS', name)
		with open(save_path, 'wb') as f:
			f.write(content)
		
			
		# 处理数据
		if option == 1:
			# PCA
			(x_train, _), (x_test, _), _ = dataloader.load_HDFS(save_path,
																		window='session',
																		train_ratio=0.5,
																		split_type='uniform')
			feature_extractor = preprocessing.FeatureExtractor()
			x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
													normalization='zero-mean')
			x_test = feature_extractor.transform(x_test)

			model = PCA()
			model.fit(x_train)

			y_pred = model.predict(x_test)

			print(y_pred)


		response_data = {
			'message': 'Data analysised',
			'x_test': x_test.tolist(),
			'y_pred': y_pred.tolist(),
		}
		return JsonResponse(response_data)
	return JsonResponse({'message': 'Only POST method is allowed'}, status=400)

