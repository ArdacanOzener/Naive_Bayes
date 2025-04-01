import json
import pandas as pd

class Bayesian:
    def __init__(self, train_dataset=None, test_dataset=None):
        self.priors = {}
        self.likelihoods = {}
        print("Training dataset summary\n")
        self.train_dataset = self.get_dataset(train_dataset)
        print("\n\nTest dataset summary\n")
        self.test_dataset = self.get_dataset(test_dataset)

    def train(self):
        # Calculate priors
        total_instances = len(self.train_dataset)
        class_counts = self.train_dataset['PlayTennis'].value_counts().to_dict()
        self.priors = {k: v / total_instances for k, v in class_counts.items()}

        # Initialize likelihoods
        self.likelihoods = {feature: {outcome: {} for outcome in self.priors} 
                            for feature in self.train_dataset.columns[:-1]}

        # Calculate likelihoods with Laplace smoothing
        for feature in self.train_dataset.columns[:-1]:
            for outcome in self.priors:
                feature_values = self.train_dataset[self.train_dataset['PlayTennis'] == outcome][feature].value_counts().to_dict()
                total_count = class_counts[outcome]
                unique_values = len(self.train_dataset[feature].unique())
                for value in self.train_dataset[feature].unique():
                    # Laplace smoothing
                    self.likelihoods[feature][outcome][value] = (feature_values.get(value, 0) + 1) / (total_count + unique_values)

    def get_dataset(self, filepath):
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        dataset = pd.DataFrame(dataset)
        summary = {
            "Total Instances": len(dataset),
            "Class Counts": dataset["PlayTennis"].value_counts().to_dict(),
            "Attribute Summary": dataset.describe(include='all').to_dict()
        }

        # Print the summary
        print("Total Instances:", summary["Total Instances"])
        print("\nClass Counts:")
        for class_label, count in summary["Class Counts"].items():
            print(f"  {class_label}: {count}")

        print("\nAttribute Summary:")
        for attribute, stats in summary["Attribute Summary"].items():
            print(f"\n{attribute}:")
            for stat_name, stat_value in stats.items():
                print(f"  {stat_name}: {stat_value}")
        print(dataset)
        return dataset

        

    def save_model(self, filepath):
        # Save model to JSON
        model = {
            'priors': self.priors,
            'likelihoods': self.likelihoods
        }
        with open(filepath, 'w') as f:
            json.dump(model, f)

    def load_model(self, filepath):
        # Load model from JSON
        with open(filepath, 'r') as f:
            model = json.load(f)
        self.priors = model['priors']
        self.likelihoods = model['likelihoods']


    def test(self):
        def predict(instance):
            # Calculate probability for each class
            labels = {}
            for outcome in self.priors:
                probability = self.priors[outcome]
                for feature, value in instance.items():
                    # For all outcomes ignore the undefined feature
                    probability *= self.likelihoods[feature][outcome][value] if feature in self.likelihoods else 1
                labels[outcome] = probability

            # Get the class with the highest label
            return max(labels, key=labels.get)
        

        def evaluation(confusion_matrix):
            TP, FN = confusion_matrix[0][0], confusion_matrix[0][1]
            FP, TN = confusion_matrix[1][0], confusion_matrix[1][1]

            # Calculate metrics
            total = TP + FN + FP + TN
            accuracy = (TP + TN) / total if total > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Print metrics
            print(f"True Positives: {TP}")
            print(f"False Positives: {FP}")
            print(f"True Negatives: {TN}")
            print(f"False Negatives: {FN}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}")

        confusion_matrix = [[0,0],[0,0]] #[[TP, FN], [FP, TN]]
        for i, row in self.test_dataset.iterrows():
            instance = row.to_dict()
            actual = instance.pop('PlayTennis')
            predicted = predict(instance)
            actual_index = 0 if actual == "Yes" else 1
            predicted_index = 0 if predicted == "Yes" else 1
            confusion_matrix[actual_index][predicted_index] += 1

        evaluation(confusion_matrix)

    

