import numpy as np


class solver():
    def __init__(self):
        pass

    # 2.1
    def dichotimizer(self, weight, bias, feature_vectors):
        assert len(weight) == len(feature_vectors[0])
        label_hat_list = []
        for vector in feature_vectors:
            label_hat = bias
            for i in range(len(weight)):
                label_hat += weight[i] * vector[i]
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag}})
        for label in label_hat_list:
            print(label)
        print("\n\n\n")
        return label_hat_list

    # 2.2 augmented
    def augmented_dichotimiezer(self, a_t, y):
        """

        :param a: it's a, not transpose one.
        :param y:
        :return:
        """
        label_hat_list = []
        for vector in y:
            vector = np.transpose(vector)
            label_hat = np.matmul(a_t, vector)
            label_hat_tag = 1 if label_hat > 0 else 2
            label_hat_list.append({str(vector): {"value": label_hat, "label_hat": label_hat_tag}})
        for label in label_hat_list:
            print(label)
        print("\n\n\n")
        return label_hat_list

    # 2.6
    def batch_perceptron_learning_algorithm_with_normalisation(self, x_list, a_t, learning_rate):
        """
        This algorithm is only for linear separably
        :param x_list:
        :param label:
        :param a_t:
        :param learning_rate:
        :return:
        """
        score = 0  # score plus 1 when there is a true match
        while score != len(x_list):
            score = 0
            miss_match = []
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # if it's a miss
                    miss_match.append(vector)
                else:
                    score += 1
            # in the end, update the a_t value
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
        return a_t

    def batch_perceptron_learning_algorithm_without_sn(self, a_t, x_list, label, learning_rate):
        score = 0
        while score != len(x_list):
            score = 0
            miss_match = []
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                # since it's not been sample normalised, label_hat not equal = label is a mismatch
                if label_hat != label[index]:  # a miss
                    # if it's a miss
                    miss_match.append(label[index] * np.array(vector))
                else:
                    score += 1
            # in the end, update the a_t value
            a_t = np.transpose(np.transpose(a_t) + learning_rate * sum(miss_match))
        return a_t

    # By default, it's been normalised
    def sequential_gradient_descent_dichotomizers(self, a_t, x_list, learning_rate):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        score = 0
        while score != len(x_list):
            score = 0
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                label_hat = np.matmul(a_t, vector)
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if not label_hat > 0:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    a_t += np.transpose(learning_rate * vector)
                else:
                    score += 1
        return a_t

    # When it's not sample normalised
    def sequential_gradient_descent_dichotomizers_withou_sn(self, a_t, x_list, label, learning_rate):
        """
        label_hat = a_t * y
        a_t = a_t + learning_rate * yk where yk is wrongly label feature.
        :param a_t: augmented variable a_t = [w_0 wT]_T
        :param y: augmented variable y = [1 xT]_T
        :param learning_rate:
        :return:
        """
        score = 0
        while score != len(x_list):
            score = 0
            for index, vector_t in enumerate(x_list):
                vector = np.transpose(vector_t)
                label_hat = 1 if np.matmul(a_t, vector) > 0 else -1
                # since it's been sample normalised, g(x) < 0 is a miss classification
                if label_hat != label[index]:  # a miss
                    # since it's a sequential gradient descent, the a_t changes immediately
                    a_t += np.transpose(learning_rate * vector) * label[index]
                else:
                    score += 1
        return a_t

    def heaviside_func(self, value):
        return 1 if value > 0 else 0

    # 3 Sequential Delta Learning Algorithm this is for Linear Threshold Units
    # Kimhom Question:
    # According to Page 20 iteration stops when w does not change
    # but in the very next page, the condition when iteration stops becomes all match
    def sequential_delta_learning_algorithm_stop_when_all_match(self, w, label, x, learning_rate=1):
        """
        in this algorithm, parameters must be augmented first.
        :param w:
        :param label:
        :param x:
        :return:
        """
        w = np.array(w)
        x = np.array(x)
        score = 0
        assert len(label) == len(x)
        while score != len(label):
            score = 0
            for index, vector_t in enumerate(x):
                vector = vector_t.transpose()
                zxx = (label[index] - self.heaviside_func(np.matmul(w, vector)))
                zx = (label[index] - self.heaviside_func(np.matmul(w, vector))) * np.array(vector_t)
                new_w = w + learning_rate * (label[index] - self.heaviside_func(np.matmul(w, vector))) * np.array(
                    vector_t)
                w = new_w
                if label[index] - self.heaviside_func(np.matmul(w, vector)) == 0:
                    score += 1
        return w

    def check_all_match(self, w, x, label):
        assert len(label) == len(x)
        for index, vector_t in enumerate(x):
            label_hat = np.matmul(w, np.transpose(vector_t))
            if label[index] != self.heaviside_func(label_hat):
                return False
        return True

    # stop when no changes
    def sequential_delta_learning_algorithm_stop_when_no_changes(self, w, label, x, learning_rate=1):
        """
        in this algorithm, parameters must be augmented first.
        :param w:
        :param label:
        :param x:
        :return:
        """
        w = np.array(w)
        x = np.array(x)
        score = 0
        assert len(label) == len(x)
        while True:
            score = 0
            old_w = w
            for index, vector_t in enumerate(x):
                vector = vector_t.transpose()
                new_w = w + learning_rate * (label[index] - self.heaviside_func(np.matmul(w, vector))) * np.array(
                    vector_t)
                w = new_w
            if np.array_equal(new_w, old_w):
                return w
