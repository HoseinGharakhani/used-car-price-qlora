from pydantic import BaseModel
from datasets import Dataset, DatasetDict, load_dataset
from typing import Optional, Self


PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class CarItem(BaseModel):
    """
    A CarItem is a data-point of a used car with a Price.
    """

    brand: str
    model: str
    model_year: int
    milage: float
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str
    price: float
    prompt: Optional[str] = None
    completion: Optional[str] = None

    def summary(self) -> str:
        """Build the raw description text for this car."""
        return (
            f"Brand: {self.brand}\n"
            f"Model: {self.model}\n"
            f"Year: {self.model_year}\n"
            f"milage: {self.milage}\n"
            f"Fuel: {self.fuel_type}\n"
            f"Engine: {self.engine}\n"
            f"Transmission: {self.transmission}\n"
            f"Exterior: {self.ext_col}\n"
            f"Interior: {self.int_col}\n"
            f"Accident: {self.accident}\n"
            f"Clean title: {self.clean_title}"
        )

    def make_prompt(self, text: str):
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.brand} {self.model} = ${self.price}>"

    @staticmethod
    def push_to_hub(
        dataset_name: str,
        train: list[Self],
        val: list[Self],
        test: list[Self],
    ):
        """Push CarItem lists to HuggingFace Hub."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load from HuggingFace Hub and reconstruct CarItems."""
        ds = load_dataset(dataset_name)
        
        # 1. If the dataset has only one split (e.g., train), split it into train/validation/test
        if not isinstance(ds, DatasetDict):
            print("Dataset has only one split. Splitting into train/validation/test...")
            train_testvalid = ds.train_test_split(test_size=0.3, seed=42)
            test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)
            ds = DatasetDict({
                "train": train_testvalid["train"],
                "validation": test_valid["train"],
                "test": test_valid["test"]
            })
        
        # 2. If validation split is missing, create it from train
        if "validation" not in ds:
            print("Validation split not found. Creating it from train...")
            if "test" in ds:
                train_val = ds["train"].train_test_split(test_size=0.1, seed=42)
                ds = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"],
                    "test": ds["test"]
                })
            else:
                train_val = ds["train"].train_test_split(test_size=0.2, seed=42)
                ds = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"],
                    "test": train_val["test"]
                })

        # 3. Helper function to fix column names and skip incomplete rows
        def process_row(row):
            # Fix column name if needed (e.g., 'mileage' to 'milage')
            if "mileage" in row and "milage" not in row:
                row["milage"] = row.pop("mileage")
            
            # Try to validate. If data is incomplete (None), it raises an error and we return None
            try:
                return cls.model_validate(row)
            except Exception:return None # Skip this row

        # Filter out None (incomplete) rows
        train_items = [item for item in (process_row(row) for row in ds["train"]) if item is not None]
        val_items = [item for item in (process_row(row) for row in ds["validation"]) if item is not None]
        test_items = [item for item in (process_row(row) for row in ds["test"]) if item is not None]

        return train_items, val_items, test_items

    def count_tokens(self, tokenizer):
        """Count tokens in the summary."""
        return len(tokenizer.encode(self.summary(), add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens, do_round):
        """Make prompts and completions."""
        text = self.summary()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            text = tokenizer.decode(tokens[:max_tokens]).rstrip()
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
        self.completion = (
            f"{round(self.price)}.00" if do_round else str(self.price)
        )

    def count_prompt_tokens(self, tokenizer):
        """Count tokens in the prompt."""
        full = self.prompt + self.completion
        tokens = tokenizer.encode(full, add_special_tokens=False)
        return len(tokens)

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str,
        train: list[Self],
        val: list[Self],
        test: list[Self],
    ):
        """Push CarItem lists to HuggingFace Hub in prompt-completion format for SFT training."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.to_datapoint() for item in train]),
                "val": Dataset.from_list([item.to_datapoint() for item in val]),
                "test": Dataset.from_list([item.to_datapoint() for item in test]),
            }
        ).push_to_hub(dataset_name)