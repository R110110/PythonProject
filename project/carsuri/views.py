from django.shortcuts import render
import MySQLdb
from carsuri.models import Maker, Model, Detail
from django.http import JsonResponse
from django.core import serializers
import json


# Create your views here.
def MainFunc(request):
    datas = Maker.objects.all()
    return render(request, 'main.html', {'maker':datas})
    
def ModelFunc(request):
    makerId=request.GET.get('makerId')
    car_Model = Model.objects.filter(maker_num=makerId.replace("maker", ""))
    serialized_data = serializers.serialize('json', car_Model)
    data = json.loads(serialized_data)
    return JsonResponse(data, safe=False)

def DetailFunc(request):
    modelId=request.GET.get('modelId')
    model_detail = Detail.objects.filter(model_num=modelId.replace("maker", ""))
    serialized_data = serializers.serialize('json', model_detail)
    data = json.loads(serialized_data)
    return JsonResponse(data, safe=False)