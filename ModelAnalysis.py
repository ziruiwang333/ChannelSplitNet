import numpy as np

def calculate_accuracy(true_label_y, pred_label_y):
    '''
    This function is to calculate the accuracy
    :param true_label_y: A list of true labels
    :param pred_label_y: A list of predicted labels
    :return: Accuracy
    '''

    if len(true_label_y)!=len(pred_label_y):
        print("length of true label and predicted label should be same")
        return None

    result_list = []
    for i in range(len(true_label_y)):
        if true_label_y[i] == pred_label_y[i]:
            result_list.append(1)
        else:
            result_list.append(0)
    
    return np.sum(result_list)/len(result_list)

def calculate_specific_label_acc(true_label_y, pred_label_y, specific_label):
    '''
    This function is to calculate the accuracy
    :param true_label_y: A list of true labels
    :param pred_label_y: A list of predicted labels
    :param specific_label: A label number needs to be calculated accuracy
    :return: Specific label accuracy
    '''

    if len(true_label_y)!=len(pred_label_y):
        print("length of true label and predicted label should be same")
        return None
    
    result_list = []
    misclassified_list = []

    for i in range(len(true_label_y)):
        if true_label_y[i] == specific_label:
            if pred_label_y[i] == specific_label:
                result_list.append(1)
            else:
                result_list.append(0)
                misclassified_list.append(pred_label_y[i])
    
    return np.sum(result_list)/len(result_list), misclassified_list