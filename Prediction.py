import numpy as np
import time

def predict(model, x, showTime=True):
    '''
    This function is to make prediction of single model
    :param model: The model needs to make the prediction
    :param x: Test data (not include labels)
    :param showTime: Show prediction time or not. Default is True
    :return: A prediction list.
    '''
    start_time = time.time()
    pred = model.predict(x)
    predict_time = time.time() - start_time
    if showTime:
        print("Predict complete, time used: ", predict_time)
    result_list = pred.argmax(1)
    return result_list

def ensemble_prediction(model, x, include_GoogLeNet=False, GoogLeNetModel=None):
    '''
    This function is to make prediction of ensemble model
    :param model: A list of base models. This list is not include GoogLeNet.
    :param x: Test data (not include labels)
    :param include_GoogLeNet: True if ensemble model has GoogLeNet as base model. Default is False.
    :param GoogLeNetModel: If include_GoogLeNet is True, then this should be GoogLeNet model.
    :return: A prediction list.
    '''
    modelLength = len(model)
    pred_list = []
    result_list = []
    start_time = time.time()
    for i in range(modelLength):
        pred_y = np.argmax(model[i].predict(x), axis=1)
        pred_list.append(pred_y)
    predict_time = time.time()-start_time
    if include_GoogLeNet:
        pred_y = np.asarray(googlenet_prediction(GoogLeNetModel, x, showTime=False)[2])
        pred_list.append(pred_y)
    print("Predict complete, time used: ", predict_time)
    
    for i in range(len(pred_list[0])):
        temp_list = []
        for j in range(len(pred_list)):
            temp_list.append(pred_list[j][i])
        result_list.append(max(temp_list, key=temp_list.count))

    return np.asarray(result_list)

def googlenet_prediction(model, x, showTime=True):
    '''
    This function is to make prediction of GoogLeNet
    :param model: The GoogLeNet model needs to make the prediction
    :param x: Test data (not include labels)
    :param showTime: Show prediction time or not. Default is True
    :return: A list consist of four prediction list. First is softmax0 prediction list. 
             Second is softmax1 prediction list. Third is softmax2 prediction list. 
             Fourth is ensemble prediction list by these three softmax prediction.
    '''

    start_time = time.time()
    pred = model.predict(x)
    predict_time = time.time()-start_time
    if showTime:
        print("GoogLeNet prediction time: ", predict_time)

    sm0_pred = pred[0].argmax(1).tolist()
    sm1_pred = pred[1].argmax(1).tolist()
    sm2_pred = pred[2].argmax(1).tolist()
    ensemble_pred = []

    for i in range(len(sm0_pred)):
        if sm0_pred[i] == sm1_pred[i]:
            ensemble_pred.append(sm0_pred[i])
        elif sm0_pred[i] == sm2_pred[i]:
            ensemble_pred.append(sm0_pred)
        elif sm1_pred[i] == sm2_pred[i]:
            ensemble_pred.append(sm1_pred[i])
        else:
            ensemble_pred.append(sm2_pred[i])

    return np.asarray([sm0_pred, sm1_pred, sm2_pred, ensemble_pred])