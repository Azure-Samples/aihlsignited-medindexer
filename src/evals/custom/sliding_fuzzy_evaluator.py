from dataclasses import dataclass

from rapidfuzz import fuzz

from src.evals.custom.custom_evaluator import CustomEvaluator
from src.utils.ml_logging import get_logger


# Define a dataclass for returning fuzzy (indel) similarity
@dataclass
class IndelSimilarity:
    indel_similarity: float


class SlidingFuzzyEvaluator(CustomEvaluator):
    def __init__(self, **kwargs):
        """
        Initialize the evaluator with any number of keyword arguments.

        All keyword arguments provided during initialization are set as attributes of the instance.

        Example:
            evaluator = SlidingFuzzyEvaluator(param1="value1", param2=42)
            print(evaluator.param1)  # Output: "value1"
            print(evaluator.param2)  # Output: 42

        Parameters:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.logger = get_logger()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _clean_text(self, text: str) -> str:
        """
        Removes newline and tab characters from the given text.

        Parameters:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # Replace newline and tab with a space, then collapse multiple spaces.
        return " ".join(text.replace("\n", " ").replace("\t", " ").split())

    def _sliding_window_fuzzy_match(self, ground_truth: str, response: str, step: int = 1) -> tuple:
        """
        Computes the best fuzzy matching score between the ground truth and any substring of the response
        using a sliding window of the same length as the ground truth.

        The window moves by the specified number of characters (step) each iteration.

        Parameters:
            ground_truth (str): The reference text.
            response (str): The larger text (e.g., OCR output) where the ground truth is searched.
            step (int): The number of characters to shift the window at each iteration (default is 1).

        Returns:
            tuple: (best_score, best_window, best_index)
                best_score (float): The highest fuzzy matching score.
                best_window (str): The substring of the response that produced the best score.
                best_index (int): The starting index of the best matching substring in the response.
        """
        window_size = len(ground_truth)
        best_score = 0
        best_window = ""
        best_index = -1

        # Slide through the response by the given step size (number of characters)
        for i in range(0, len(response) - window_size + 1, step):
            window = response[i:i + window_size]
            score = fuzz.ratio(ground_truth, window)
            if score > best_score:
                best_score = score
                best_window = window
                best_index = i
                # Early exit if a perfect match is found.
                if best_score == 100:
                    break
        return best_score, best_window, best_index

    def __call__(self, *, response: str, ground_truth: str, **kwargs) -> IndelSimilarity:
        """
        Computes fuzzy similarity between the response and ground_truth using a sliding window approach.
        Newline and tab characters are removed from the ground truth prior to matching.

        The sliding window moves by a number of characters defined by the 'step' keyword argument (default is 1).

        Signature:
            __call__(*, response: str, ground_truth: str, **kwargs) -> IndelSimilarity

        Returns:
            An IndelSimilarity instance containing the computed fuzzy (indel) similarity.
        """
        try:
            # Clean the ground truth to remove newline and tab characters.
            clean_ground_truth = self._clean_text(ground_truth)
            # Retrieve a custom step if provided, otherwise default to 10.
            step = kwargs.get("step", 10)
            best_score, best_window, best_index = self._sliding_window_fuzzy_match(
                clean_ground_truth, response, step=step
            )
            self.logger.info(f"Best score: {best_score} at index {best_index}.")
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            best_score = 0

        return IndelSimilarity(indel_similarity=best_score)