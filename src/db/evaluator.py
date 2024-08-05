# evaluator.py

import numpy as np
from sklearn.metrics import mutual_info_score
import time

class TaskEvaluator:
    def __init__(self, relatedness_matrix=None, task_complexities=None):
        self.relatedness_matrix = relatedness_matrix
        self.task_complexities = task_complexities

    def calculate_average_relatedness(self):
        if self.relatedness_matrix is not None:
            return self.relatedness_matrix.mean(axis=1)
        return None

    @staticmethod
    def calculate_mutual_information(task1_data, task2_data):
        return mutual_info_score(task1_data, task2_data)

    def identify_complex_tasks(self, complexity_threshold):
        if self.task_complexities is not None:
            return [i for i, complexity in enumerate(self.task_complexities) if complexity > complexity_threshold]
        return []

    @staticmethod
    def measure_execution_time(task_function):
        start_time = time.time()
        task_function()
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def calculate_comprehensive_scores(relatedness_scores, information_gain_scores, complexity_scores, execution_time_scores, weights):
        return [
            weights['relatedness'] * relatedness_scores[i] +
            weights['information_gain'] * information_gain_scores[i] +
            weights['complexity'] * complexity_scores[i] +
            weights['execution_time'] * execution_time_scores[i]
            for i in range(len(relatedness_scores))
        ]

    @staticmethod
    def identify_tasks_to_split(scores, threshold):
        return [i for i, score in enumerate(scores) if score > threshold]
