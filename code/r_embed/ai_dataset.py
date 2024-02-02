from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer

# --------------- Dataset ----------------------------------------------#


class AiDataset:
    """
    Dataset class for LLM Detect AI Generated Text competition
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path)

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

        return tz

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df):
        """
        Main api for creating the Science Exam dataset
        :param df: input dataframe
        :type df: pd.DataFrame
        :return: the created dataset
        :rtype: Dataset
        """
        df = deepcopy(df)
        task_dataset = Dataset.from_pandas(df)

        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)

        return task_dataset
