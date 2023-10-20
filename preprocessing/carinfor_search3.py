# file_name    damage    part    category_id
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
import csv
import re

folder_path_damage = r'C:\work\TL_damage\damage'  # 읽어올 폴더의 경로
folder_path_part = r'C:\work\TL_damage_part\damage_part'
json_files = [f for f in os.listdir(folder_path_damage) if f.endswith('.json')]

carinfo_data = []
cnt = 0
cnt_not_have_part = 0
cnt_have_part = 0
cnt_same_category = 0 
damage =''
pre_category=''
# 상위 100개까지만 처리
# for json_file in json_files[:100]:
for json_file in json_files[:100000]:
    cnt = cnt + 1
    print(cnt)
    
    category = re.findall(r'-(\d+)\.json', json_file)[0]  # 파일명에서 yyy 값을 추출
    if category == pre_category:  # 이전 파일과 yyy 값이 동일한 경우
        cnt_same_category = cnt_same_category + 1
        continue  # 현재 파일 건너뛰기
    pre_category = category
    
    
    file_path = os.path.join(folder_path_damage, json_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        
        # part 추출
        data = json.load(f)
        # print(annotations)
        annotations = data['annotations']
        category_id = annotations[0]['category_id']
        file_name = os.path.splitext(json_file)[0]+'.json'
        print('file_name',file_name)
              
        json_files_part = [f for f in os.listdir(folder_path_part) if f.endswith('.json') and category_id in f]
        if len(json_files_part) > 0:
            cnt_have_part = cnt_have_part + 1
            json_file_part = json_files_part[0]
            file_path_part = os.path.join(folder_path_part, json_file_part) # part json 읽기
            
            for annotation in annotations: # damage의 annotations key 루프
                damage_values = [annotation["damage"] for item in annotation]
                if all(damage == "Scratched" for damage in damage_values):
                    damage = 'Scratched'
                else:
                    damage = 'damaged'
                        
            with open(file_path_part, 'r', encoding='utf-8') as f_part:
                data_part = json.load(f_part)
                annotations = data_part['annotations'] 
                car_part = set()
                for annotation in annotations: # annotations key 루프
                    # print(annotation)
                    if annotation['part'] is not None:
                        print('category_id:',category_id,' damage:',damage,' part:', annotation['part']) 
                        carinfo_data.append((file_name, damage, annotation['part'], category_id))
        else:
            print('해당 파일을 포함하는 json 파일이 없습니다.')
            cnt_not_have_part = cnt_not_have_part + 1
            
            
print('part있음: ', cnt_have_part)           
print('part없음: ', cnt_not_have_part)           
print('중복 카테고리: ', cnt_same_category)           

filename = 'car_damage_part_data1.csv'

# Write the training data to the CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_name', 'damage', 'part','category_id'])  # Write the header row

    # Write each training example as a row in the CSV file
    for file_name, damage, annotation['part'], category_id in carinfo_data:
        writer.writerow([file_name, damage, annotation['part'], category_id])
