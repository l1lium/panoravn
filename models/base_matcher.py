from abc import ABC, abstractmethod

class MatcherInterface(ABC):
    @abstractmethod
    def match(self, image_a, image_b):
        pass
