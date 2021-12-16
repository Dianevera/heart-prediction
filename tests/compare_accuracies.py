import pickle

def compare_accuracies(accuracies_file_path, actual_accuracies_file_path):
        accuracies_file = open(accuracies_file_path, "rb")
        new_values = pickle.load(accuracies_file)
        accuracies_file.close()

        accuracies_file = open(actual_accuracies_file_path, "rb")
        actual_values = pickle.load(accuracies_file)
        accuracies_file.close()

        keys = list(actual_values)

        lower_accuracies = []
        no_upper = True

        for key in keys:
            is_lower = actual_values[key] >= new_values[key]

            if not is_lower:
                no_upper = False

            comparaison_str = str(actual_values[key]) + " <= " + str(new_values[key])
            lower_accuracies.append(key + " : " + str(is_lower) + ' ' + comparaison_str)
        
        return no_upper, lower_accuracies