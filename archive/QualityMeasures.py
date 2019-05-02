class QualityMeasure:
    @staticmethod
    def accuracy(correct_predictions, false_predictions):
        return correct_predictions / (correct_predictions + false_predictions)

    @staticmethod
    def precision(true_positive, false_positive):
        return true_positive / (true_positive + false_positive)

    @staticmethod
    def recall(true_positive, false_negative):
        return true_positive / (true_positive + false_negative)

    @staticmethod
    def f1measure(true_positive, false_positive, false_negative):
        return 2 / (1 / QualityMeasure.precision(true_positive, false_positive) + 1 / QualityMeasure.recall(true_positive, false_negative))

    @staticmethod
    def specificity(false_positive, true_negative):
        return false_positive / (false_positive + true_negative)

    @staticmethod
    def sensitivity(true_positive, false_negative):
        return true_positive / (true_positive + false_negative)
