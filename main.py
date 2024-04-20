import dkvs.bptree
import datetime
import msgspec
from typing import Any


print("Started...")


if __name__ == "__main__":

    class Key(msgspec.Struct, frozen=True, order=True):
        """
            Ключ индекса
            пока это строка, но в будущем - все что угодно что можно сравнивать
            - Ключи - неизменяемые сущности
        """
        value: str
        t: datetime.datetime

    class Value(msgspec.Struct):
        """
            значение соответсвующее ключу
        """
        # bucket: int = random.randint(0, 16)  # автоматически будет сгенерировано
        data: Any = None  # данные

    tree = dkvs.bptree.BPTree[Key, Value]()
