import numpy as np
from scipy.optimize import minimize


def get_data():
    data = []
    dict = {'Iris-setosa':[1, -1, -1],
            'Iris-versicolor':[-1, 1, -1],
            'Iris-virginica':[-1, -1, 1]}
    with open("iris.data", "r") as file:
        for line in file.readlines():
            splitted_line = line.split(",")
            if len(splitted_line) == 5:
                data.append([list(map(float, splitted_line[:4])), dict[splitted_line[4][:-1]]])
    return data

def divide_data(data, test_ratio, train_ratio):
    validation_ratio = 1 - test_ratio - train_ratio
    if (validation_ratio <= 0):
        print("Wrong test_ratio and train_ratio")
        return
    np.random.shuffle(data)
    train_size = int(train_ratio*len(data))
    test_size = int(test_ratio*len(data))
    train_set = data[:train_size - 1]
    test_set = data[train_size:train_size+test_size]
    validation_set = data[train_size+1+test_size:len(data)-1]
    return train_set, test_set, validation_set

def classify_data(data):
    class1 = []
    class2 = []
    class3 = []
    for row in data:
        if row[1][0] == 1:
            class1.append(row)
        elif row[1][1] == 1:
            class2.append(row)
        elif row[1][2] == 1:
            class3.append(row)
    return class1, class2, class3

def make_4D_set(index, data):
    data_set = np.array([[row[0][0], row[0][1], row[0][2], row[0][3], row[1][index]] for row in data])
    return data_set

def dzeta(w, train_set):
    dzeta = []
    for i in train_set:
        f = np.matmul(i[:len(train_set[0]) - 1], w[:len(train_set[0]) - 1]) - w[len(train_set[0]) - 1]
        y = i[len(train_set[0]) - 1]
        dzeta.append(max(1-f*y, 0))
    return np.array(dzeta).sum()

def mistake(w, train_set, l):
    return dzeta(w, train_set) + l*np.linalg.norm(w)


def SVM(train_set, l):
    res = minimize(mistake, np.random.random(len(train_set[0])), args = (train_set, l))
    w = res.x[:len(train_set[0]) - 1]
    b = res.x[len(train_set[0]) - 1]
    return w, b

def classify_f(f):
    if f > 0:
        return 1
    return -1

def classify(x, w_12, w_23, w_31, b_12, b_23, b_31):
    result = [classify_f(np.matmul(w_12, x) - b_12),
              classify_f(np.matmul(w_23, x) - b_23),
              classify_f(np.matmul(w_31, x) - b_31)]
    if result[0] == 1:
        return [1, -1, -1]
    elif result[1] == 1:
        return [-1, 1, -1]
    elif result[2] == 1:
        return [-1, -1, 1]

def validation(train_set, validation_set, lambdas):
    set1, set2, set3 = classify_data(train_set)
    accuracies = []
    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    best_accuracy = 0
    accuracy1 = 1
    accuracy2 = 1
    accuracy3 = 1
    for l in lambdas:
        w_12, b_12 = SVM(make_4D_set(0, set1 + set2), l)
        w_23, b_23 = SVM(make_4D_set(1, set2 + set3), l)
        w_31, b_31 = SVM(make_4D_set(2, set3 + set1), l)
        vset1, vset2, vset3 = classify_data(validation_set)
        errors1 = 0
        errors2 = 0
        errors3 = 0
        
        for row in vset1:
            if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [1, -1, -1]):
                errors1 += 1
        accuracy1 = 1 - errors1/len(vset1)
        accuracies1.append(accuracy1)
        for row in vset2:
            if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, 1, -1]):
                errors2 += 1
        accuracy2 = 1 - errors2/len(vset2)
        accuracies2.append(accuracy2)
        for row in vset3:
            if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, -1, 1]):
                errors3 += 1
        accuracy3 = 1 - errors3/len(vset3)
        accuracies3.append(accuracy3)
        accuracy = (accuracy1 + accuracy2 + accuracy3)/3
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda = l
            best_w_12, best_w_23, best_w_31, best_b_12, best_b_23, best_b_31 = w_12, w_23, w_31, b_12, b_23, b_31
    with open("validation.txt", 'w') as file:
        file.write(f"Lambda: {lambdas}\nAccuracy: {accuracies}\nAccuracy1: {accuracies1}\nAccuracy2: {accuracies2}\nAccuracy3{accuracies3}")
    return best_lambda, best_w_12, best_w_23, best_w_31, best_b_12, best_b_23, best_b_31


def main(test_ratio, train_ratio,  lambdas):
    data = get_data()
    train_set, test_set, validation_set = divide_data(data, test_ratio, train_ratio)
    l, w_12, w_23, w_31, b_12, b_23, b_31 = validation(train_set, validation_set, lambdas)
    tset1, tset2, tset3 = classify_data(test_set)
    errors1 = 0
    errors2 = 0
    errors3 = 0
    print(f"Lambda: {l}")
    for row in tset1:
        if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [1, -1, -1]):
            errors1 += 1
    print(f"Class Iris-setosa: accuracy {(1 - errors1/len(tset1))*100:.2f}%")

    for row in tset2:
        if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, 1, -1]):
            errors2 += 1
    print(f"Class Iris-versicolor: accuracy {(1 - errors2/len(tset2))*100:.2f}%")

    for row in tset3:
        if (classify(row[0], w_12, w_23, w_31, b_12, b_23, b_31) != [-1, -1, 1]):
            errors3 += 1
    print(f"Class Iris-virginica: accuracy {(1 - errors3/len(tset3))*100:.2f}%")
    print(f"Total accuracy {(1 - (errors1 + errors2 + errors3)/(len(tset1) + len(tset2) + len(tset3)))*100:.2f}%")

main(0.2, 0.6, [0.01, 0.1, 1, 10, 100, 1000])