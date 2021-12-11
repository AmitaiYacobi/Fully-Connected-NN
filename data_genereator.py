import sys
import numpy as np



if __name__ == "__main__":
    train_data = np.loadtxt(sys.argv[1])
    targets = np.loadtxt(sys.argv[2])

    union_data = list(zip(train_data, targets))
    np.random.shuffle(union_data)

    train_data, targets = zip(*union_data)

    data_for_play = train_data[: 5000]
    targets_for_play = targets[: 5000]

    np.savetxt("data_for_play.txt", data_for_play)
    np.savetxt("targets_for_play.txt", targets_for_play)


    # arr1 = np.array([[1,2,3], [4,5,6]])
    # arr2 = np.array([0, 1])

    # print(arr1)
    # print(arr2)

    # union_data = list(zip(arr1, arr2))
    # np.random.shuffle(union_data)

    # arr1, arr2 = zip(*union_data)

    # print(arr1[0])
    # print(arr2[0])
