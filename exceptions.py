class MissingLLMModelException(Exception):
    def __init__(self):
        self.message = "The requested LLM model was not found."
        super().__init__(self.message)


class NoLLMOutputException(Exception):
    def __init__(self):
        self.message = "The LLM responded with no output."
        super().__init__(self.message)
