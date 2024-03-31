class MetricBase:
    def __init__(self, predictions, labels):
        """
        Initialize the base class for metrics.
        
        Parameters:
        - predictions (list/array): The predictions.
        - labels (list/array): The actual labels or ground truth.

        """
        self.predictions = predictions
        self.labels = labels

    def compute_metric(self):
        """
        Compute the metric. This method should be overridden by specific metric implementations.
        
        Returns:
        - float/dict: The metric or metrics computed.
        """
        raise NotImplementedError("This method should be overridden by subclass")
