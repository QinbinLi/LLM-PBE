class AttackBase:
    def __init__(self, model, data, prompt, metric):
        """
        Initialize the base class for attacks on language models.
        
        Parameters:
        - model (object): The language model object to be attacked.
        - data (list/dataset): The data that was used to train/fine-tune the model.
        - prompt (str): The prompt that was injected in the model.
        - metric (str/function): The metric used to evaluate the success of the attack.
        """
        
        self.model = model  # Language model to attack
        self.data = data  # Data for performing the attack
        self.prompt = prompt  # Prompt for the attack
        self.metric = metric  # Metric to evaluate the attack
    
    def execute(self):
        """
        Execute the attack. This method should be overridden by specific attack implementations.
        
        Returns:
        - dict: Metrics to evaluate the success or failure of the attack.
        """
        
        raise NotImplementedError("This method should be overridden by subclass")

    def evaluate(self, results):
        """
        Evaluate the attack based on the metric.
        
        Parameters:
        - results (list/dict): The raw results of the attack execution.

        Returns:
        - float/dict: Evaluation score or metrics based on the specified metric.
        """

        # Implement metric evaluation logic here
        # This can be a custom function or a string-based switch case for known metrics
        pass
