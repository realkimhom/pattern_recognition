import functions

if __name__ == '__main__':
    solver = functions.solver()
    a_t = [-1.5, 5, -1]
    # strictly
    y = [[1, 0, 0],
         [1, 1, 0],
         [1, 2, 1],
         [1, 0, 1],
         [1, 1, 2]]
    label = [1, 1, 1, -1, -1]
    w = [2, 1]
    w0 = -5
    feature_vector = [[1, 1], [2, 2], [3, 3]]
    solver.dichotimizer(w, w0, feature_vector)

    a = [-5, 2, 1]
    y = [[1, 1, 1], [1, 2, 2], [1, 3, 3]]
    solver.augmented_dichotimiezer(a, y)

    x = [[1, 1, 5], [1, 2, 5], [-1, -4, -1], [-1, -5, -1]]
    label = [1, 1, -1, -1]
    a_t = [-25, 6, 3]
    learning_rate = 1
    x_not = [[1, 1, 5], [1, 2, 5], [1, 4, 1], [1, 5, 1]]

    print(solver.batch_perceptron_learning_algorithm_with_normalisation(x_list=x, a_t=a_t, learning_rate=learning_rate))

    print(solver.sequential_gradient_descent_dichotomizers(x_list=x, a_t=a_t, learning_rate=learning_rate))
    print("test!")
    print(solver.batch_perceptron_learning_algorithm_without_sn(x_list=x_not, a_t=a_t, label=label,
                                                                learning_rate=learning_rate))
    print(
        solver.sequential_gradient_descent_dichotomizers_withou_sn(x_list=x_not, a_t=a_t, label=label, learning_rate=1))

    # 9
    # when it's sample_normalised
    x = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [-1, 3, -1], [-1, 2, 1], [-1, 3, 2]]
    label = [1, 1, 1, -1, -1, -1]
    a_t = [1, 0, 0]
    print(solver.sequential_gradient_descent_dichotomizers(x_list=x, a_t=a_t, learning_rate=learning_rate))
    x_not = [[1, 0, 2], [1, 1, 2], [1, 2, 1], [1, -3, 1], [1, -2, -1], [1, -3, -2]]
    print(
        solver.sequential_gradient_descent_dichotomizers_withou_sn(x_list=x_not, a_t=a_t, label=label, learning_rate=1))

    w = [[-1.5, 5, -1]]
    x = [[1, 0, 0], [1, 1, 0], [1, 2, 1], [1, 0, 1], [1, 1, 2]]
    y = [1, 1, 1, 0, 0]
    print(solver.sequential_delta_learning_algorithm_stop_when_no_changes(w, y, x))
    new_w = solver.sequential_delta_learning_algorithm_stop_when_no_changes(w, y, x)
    print(solver.check_all_match(new_w, x, y))
