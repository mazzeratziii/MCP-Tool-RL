from abc import ABC, abstractmethod
class BaseTool(ABC):
    def __init__(self, name:str, description:str ):
        self.name = name
        self.description = description

    @abstractmethod
    def invoke(self, query:str) -> str:
        pass