import msgspec
import string
from typing import Optional, List, Any, Tuple, Generic, TypeVar
import random
import sys
import datetime


KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


class PageSize:
    """Мы хотим оперировать страницами только определенных размеров"""
    def __init__(self, size: int) -> None:
        self.__sz = size

    def value(self):
        return self.__sz


class MetaData(msgspec.Struct):
    """
        У нас есть также желание хранить метаданные в заголовке индексного файла
    """
    file_name: str = "".join(random.choices(string.ascii_lowercase, k=10)) + ".index"
    page_size: int = 128


class BPTree(Generic[KeyType, ValueType]):
    """
        Каждая страница индекса имеет свой уникальный номер.
        Это значит что положение страницы в файле может быть вычислено умножением
        номера страницы на размер страницы
    """
    meta: MetaData

    class INode[KeyType, ValueType](msgspec.Struct):
        """
            IntermediateNode - промежуточный узел
            - может являться корневым
            - не содержит указатели на реальные данные, содержит указатели на узлы следующего уровня
            32 + 8 + 8 + 8  = 56
        """
        dirty: bool = True  # страницу нужно сохранить
        size: int = 8 + 8

        p_node: Optional[int] = None  # 8/ указавает на предыдущую страницу
        n_node: Optional[int] = None  # 8/ указавает на следующую страницу
        order: List[int] = []   # 8/ отсортированный список указателей на ключи, порядок определяется значениеями ключей
        keys: List[Tuple[KeyType, int]] = []  # 8/ Ключ/страница с ключами где значения меньше значения ключа

        def sort(self):
            self.order.sort(key=lambda p: self.keys[p][0])  # type: ignore # noqa


    class LNode[KeyType, ValueType](msgspec.Struct):
        """
            Листовой узел
            - не может являться корневым
            - в отличие от промежуточных содержит указатели на реальные данные
        """
        dirty = True

        p_node: Optional[int] = None  # указавает на предыдущую страницу
        n_node: Optional[int] = None  # указавает на следующую страницу

        keys: List[Tuple[KeyType, int, Any]] = []
        order: List[int] = []

        def sort(self):
            self.order.sort(key=lambda p: self.keys[p][0], reverse=(True, False))  # type: ignore # noqa

        def insert(self, key: KeyType, value: ValueType):
            self.keys.append((key, random.randint(0, 16), value))
            self.order.append(len(self.keys) - 1)
            self.sort()

        def print(self) -> None:
            for i, p in enumerate(self.order):
                print(" ->", i, p, self.keys[p])

    def __init__(self, file_name: Optional[str] = None, page_size: int = 64):

        fn = "".join(random.choices(string.ascii_lowercase, k=10)) + ".index" if file_name is None else file_name

        self.meta = MetaData(file_name=fn, page_size=page_size)
        self.root_node = None
        self.free: Optional[List[BPTree.LNode | BPTree.INode]] = None   # массив свободных страниц
        self.index: Optional[List[BPTree.LNode | BPTree.INode]] = None  # страницы
        self.dirty_map = {}  # грязные страницы


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



if __name__ == "__main__":
    print("test")
    ln: BPTree.LNode = BPTree.LNode()
    inode: BPTree.INode = BPTree.INode()
    # inode.keys
    print(sys.getsizeof(ln))
    print(sys.getsizeof(inode))
    print(Key("3", datetime.datetime.fromisoformat("1978-01-12")) > Key("3", datetime.datetime.now()))
    inode.keys.extend([
        (Key("1", datetime.datetime.now()), 0),
        (Key("2", datetime.datetime.now()), 10)
    ])

    lnode = BPTree.LNode()

    lnode.insert(Key("1", datetime.datetime.now()), Value("1"))
    lnode.insert(Key("0", datetime.datetime.fromisoformat("1966-09-13")), Value("1"))
    lnode.insert(Key("2", datetime.datetime.now()), Value("10"))
    lnode.insert(Key("9", datetime.datetime.fromisoformat("1966-09-14")), Value("1"))
    lnode.insert(Key("0", datetime.datetime.now()), Value("1"))
    lnode.insert(Key("0", datetime.datetime.fromisoformat("1966-09-14")), Value("Слава"))
    lnode.insert(Key("11", datetime.datetime.fromisoformat("1966-09-14")), Value("Слава"))

    # for i in range(10000):
    #     lnode.insert()

    lnode.print()
    keys = [
        Key(
            "".join(random.choices(string.ascii_lowercase, k=100)),
            datetime.datetime.now()
        )
        for _ in range(10)
    ]
    t_start = datetime.datetime.now()
    for k in keys:
        lnode.insert(k, Value("x"))

    t_end = datetime.datetime.now()

    print(t_end - t_start)
    lnode.print()


    # print(msgspec.msgpack.encode(Key("1", datetime.datetime.now())))
