# pyright: reportOperatorIssue=false
import msgspec
import numpy
from typing import Any, TypeVar, Optional, List, Generic, Type, cast, Tuple, Literal   # noqa
from tqdm.auto import tqdm  # noqa


KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


class BPTree(Generic[KeyType, ValueType]):

    MAX_LEAF_SIZE: int = 4   # максимальное количество ключей на странице
    MAX_INODE_SIZE: int = 4
    HALF_LEAF_SIZE: int = MAX_LEAF_SIZE // 2  # при разбивании страницы попалам
    HALF_INODE_SIZE: int = MAX_INODE_SIZE // 2

    # HALF_INODE_SIZE = 64
    # MAX_INODE_SIZE = HALF_INODE_SIZE * 2
    # HALF_LEAF_SIZE = 64
    # MAX_LEAF_SIZE = HALF_LEAF_SIZE * 2

    class Node(msgspec.Struct):
        node_index: int | None = None   # индекс узла в списке узлов дерева
        level: int = 0                  # 0 - листовой уровень, > 0 - промежуточные уровни
        prev_node: int | None = None    # индекс предыдущего узела того же уровня
        next_node: int | None = None    # индекс следующего узела того же уровня
        pointers: numpy.ndarray = numpy.array([], dtype=numpy.int32)  # указатель на ма ключ в массиве ключей

        def is_leaf(self):
            """узел является листовым? если нет - то внутренний"""
            return self.level == 0

        def print(self, keys: list):
            print()
            print(f" level: {self.level}, no#:{self.node_index} prev:{self.prev_node}, next:{self.next_node}")
            for i in self.pointers:
                print(f" -> {keys[i]}")

    class INode(Node, msgspec.Struct):
        """
            Внутренний узел (inner node)
            - descendants - Указатели на узлы следующего уровня
              т.е. узлы-потомки. Количество узлов - потомков всегда на 1 больше чем количество ключей
              - Количество потомков должно быть на один больше чем ключей в промежуточном узле
              - (len(descendants) == len(pointers) + 1)
            - у всех промежуточных узлов level >= 1, т.е. level=0 является признаком листового узла
            - если внутренний узел создан, то
              - len(pointers) >= 1
              - len(descandands) >= 2
        """
        descendants: numpy.ndarray = numpy.ndarray([], dtype=numpy.int32)

    class LeafNode(Node, msgspec.Struct):
        """
            - Листовой узел.
              - для данной реализации характерно то, что сами ключи (и данные)
                не содержатся в узлах индекса. В узлах хранятся только указатели на ключи.
                чтобы избежать копирование данных
              - у всех листовых улов level == 0
        """

    def __init__(self):
        self.tree: List[BPTree.LeafNode | BPTree.INode | BPTree.Node] = []  # массив узлов дерева
        self.keys: List[KeyType] = []      # массив с ключами

        # создание корневого узла
        _ = self.node_factory()
        self.root_node: int = 0

    def node_factory(self, leaf_node: bool = True) -> Node:
        """ - Создание узла - листового или внутреннего

            ```python
            leaf = cast(BPTree.LeafNode, self.node_factory(leaf_node=True))
            inode = cast(BPTree.INode, self.node_factory(leaf_node=True))
            ```
        """
        if leaf_node:
            node = BPTree.LeafNode()
        else:
            node = BPTree.INode()
        node.node_index = len(self.tree)
        self.tree.append(node)
        return node

    def print_node(self, node: Node):
        print()
        space = "   " * node.level
        if (node.next_node is None) and (node.prev_node is None):
            root = " <============ *ROOT*"
        else:
            root = ""
        print(f"node-idx:{node.node_index}, prev:{node.prev_node}, next:{node.next_node}, level: {node.level} {root}")
        if node.is_leaf():
            for i in node.pointers:
                print(f"{space}|-> {self.keys[i]}")
        else:
            n = cast(BPTree.INode, node)
            for i, p in enumerate(n.pointers):
                print(f"{space}|-> {self.keys[p]}: {n.descendants[i]}, {n.descendants[i + 1]}")

    def split_inode(self, inode: INode, path) -> Tuple[INode, INode]:
        """подразумеваем, что узел полностью заполнен"""

        # new_node станет следующим за inode
        new_node = cast(BPTree.INode, self.node_factory(leaf_node=False))
        new_node.level = inode.level

        # горизонтальное связывание
        if inode.next_node is not None:
            old_next_node = cast(BPTree.INode, self.tree[inode.next_node])
            new_node.next_node = old_next_node.node_index
            old_next_node.prev_node = new_node.node_index
        inode.next_node = new_node.node_index
        new_node.prev_node = inode.node_index

        # копируем ссылки
        new_node.pointers = inode.pointers[self.HALF_INODE_SIZE:]
        inode.pointers = inode.pointers[:self.HALF_INODE_SIZE]
        new_node.descendants = inode.descendants[self.HALF_INODE_SIZE:]
        inode.descendants = inode.descendants[:self.HALF_INODE_SIZE + 1]  # нужно скопировать на один узел больше

        # вертикальное связывание
        if len(path) == 0:   # подразумеваем, что: inode.node_index == self.root
            n_root = cast(BPTree.INode, self.node_factory(leaf_node=False))
            n_root.level = inode.level + 1
            n_root.pointers = new_node.pointers[:1]
            n_root.descendants = numpy.array([inode.node_index, new_node.node_index])
            self.root_node = cast(int, n_root.node_index)
        else:
            parent_node = cast(BPTree.INode, self.tree[path.pop()])
            self.insert_into_parent(parent_node, new_node.pointers[0], cast(int, new_node.node_index), path)

        return (inode, new_node)

    def insert_into_parent(self, inode: INode, k_ind: int, descendant: int, path) -> int | None:
        """
            Вставка ключа в родительский узел (в случае когда выполняется split)
            - если родительский узел существует, значит он всегда не пустой
            - когда родительский узел создан - в нем всегда есть хотя бы один ключ
        """
        low, high = 0, len(inode.pointers) - 1
        k = cast(KeyType, self.keys[k_ind])

        if len(inode.pointers) < self.MAX_INODE_SIZE:

            if k < self.keys[inode.pointers[low]]:   # меньше самого маленького
                inode.pointers = numpy.insert(inode.pointers, 0, k_ind)
                inode.descendants = numpy.insert(inode.descendants, 1, descendant)
                return k_ind

            if k > self.keys[inode.pointers[high]]:  # больше самого большого
                inode.pointers = numpy.insert(inode.pointers, len(inode.pointers), k_ind)
                inode.descendants = numpy.insert(inode.descendants, len(inode.pointers), descendant)
                return k_ind

            while low < high:
                # после этого цикла у нас только два варианта - два ключа,
                # между которыми мы должны вставить еще один ключ (вместо high)

                mid = (high + low) // 2
                if k < self.keys[inode.pointers[mid]]:
                    high = mid
                else:
                    low = mid

                if mid == (high + low) // 2:  # если mid не изменится
                    break

            inode.pointers = numpy.insert(inode.pointers, high, k_ind)  # +
            inode.descendants = numpy.insert(inode.descendants, high + 1, descendant)
            return k_ind

        else:  # len(inode.pointers) >= self.MAX_INODE_SIZE:
            """расщепляем родительский узел"""
            low_node, high_node = self.split_inode(inode, path)
            if self.keys[k_ind] < self.keys[cast(int, high_node.pointers[0])]:
                return self.insert_into_parent(low_node, k_ind, descendant, path)
            else:
                return self.insert_into_parent(high_node, k_ind, descendant, path)

    def split_leaf(self, leaf: LeafNode, path: list[int]) -> Tuple[LeafNode, LeafNode]:
        """
           Расщепить листовой узел (напопалам), а также выполнить корректировку всех предков рекурсивно
           и венуть страницу, к которую можно вставить запись
        """
        # if leaf.node_index == self.root_node:
        if len(path) == 0:  # особый случай - когда листовой узел является корневым
            # создадим родительский узел, если его нет
            parent_node = cast(BPTree.INode, self.node_factory(leaf_node=False))
            parent_node.level = leaf.level + 1
            # создадим новый листовой узел
            n_leaf = cast(BPTree.LeafNode, self.node_factory(leaf_node=True))
            # разделим листья поровну
            n_leaf.pointers = leaf.pointers[self.HALF_LEAF_SIZE:]  # вторую половину копируем на новый узел
            leaf.pointers = leaf.pointers[:self.HALF_LEAF_SIZE]  # первую половину оставляем на месте

            # горизонтальное связывание
            if leaf.next_node is not None:
                self.tree[leaf.next_node].prev_node = n_leaf.node_index
                n_leaf.next_node = leaf.next_node
            leaf.next_node = n_leaf.node_index
            n_leaf.prev_node = leaf.node_index

            split_key_ind = n_leaf.pointers[0]

            parent_node.pointers = numpy.array([split_key_ind], numpy.int32)  # копируем ссылку на наименьший ключ
            parent_node.descendants = numpy.array([leaf.node_index, n_leaf.node_index], dtype=numpy.int32)

            # обновляем корневой узел
            self.root_node = cast(int, parent_node.node_index)
            # возвращаем два вновь получившихся узла в порядке возрастания ключей
            # в один из узлов будет вставлена новая запись
            return (leaf, n_leaf)

        else:
            # листовой узел не является корневым

            split_key_ind: int = leaf.pointers[self.HALF_LEAF_SIZE]  # это индекс ключа по которому будем разбивать
            # создаем новый листовой узел
            new_leaf = cast(BPTree.LeafNode, self.node_factory(leaf_node=True))
            # копируем указатели
            new_leaf.pointers = leaf.pointers[self.HALF_LEAF_SIZE:]  # большие значения ключей
            leaf.pointers = leaf.pointers[:self.HALF_LEAF_SIZE]  # маленькие значения

            parent_node = cast(BPTree.INode, self.tree[path.pop()])  # родительский узел

            # горизонтальное связывание
            if leaf.next_node is not None:
                old_next_leaf = self.tree[leaf.next_node]
                old_next_leaf.prev_node = new_leaf.node_index
                new_leaf.next_node = old_next_leaf.node_index
            new_leaf.prev_node = leaf.node_index
            leaf.next_node = new_leaf.node_index

            # рекурсивное обновление верхних узлов /узел/ключ/узел-потомок/стэк-узлов)
            self.insert_into_parent(parent_node, split_key_ind, cast(int, new_leaf.node_index), path)

            return (leaf, new_leaf)

    def get_leaf(self, idx: int) -> LeafNode | None:   # noqa
        if self.tree[idx].level == 0:
            return cast(BPTree.LeafNode, self.tree[idx])
        return None

    def get_inode(self, idx: int) -> INode | None:   # noqa
        if self.tree[idx].level >= 1:
            return cast(BPTree.INode, self.tree[idx])
        return None

    def insert_into_leaf(self, leaf: LeafNode, k: KeyType, path: list[int]) -> Tuple[int | None, int | None]:
        """
            вставка в листовой узел.
            keys - список, куда будет добавлен новый ключ
            k - сам ключ
            возвращаем (индекс добавленного ключа, None) в случае удачи
            или возвращаем (None, индекс существующего люча) в случае неудачи - дублирующийся ключ
        """
        # сначала добавляем ключ к списку всех ключей индекса, т.е. он становится последним элементом списка
        # это нужно, чтобы получить указатель на новый ключ
        # если вставка не получится сделаем self.keys.pop()

        # узнаем, сколько элементов в узле уже есть
        page_len: int = len(leaf.pointers)

        if page_len >= self.MAX_LEAF_SIZE:  # если место в узле закончилось
            # расщепить страницы и вставить в ту страницу, куда полагается
            low_node, high_node = self.split_leaf(leaf, path)
            if k < self.keys[high_node.pointers[0]]:
                return self.insert_into_leaf(low_node, k, path)
            else:
                return self.insert_into_leaf(high_node, k, path)

        # если узел пустой
        if page_len == 0:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, 0, key_ptr)
            return (key_ptr, None)

        # Двоичный поиск места, куда можно вставить
        # самый большой и самый маленький элементы - в виде tuple
        # где первое поле - это номер элемента в массиве указателей
        low, high = 0, page_len - 1

        # если искомый элемент меньше самого маленького
        if k < self.keys[leaf.pointers[low]]:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, 0, key_ptr)
            return (key_ptr, None)
        if k == self.keys[leaf.pointers[high]]:  # уникальность
            return (None, leaf.pointers[high])

        # если искомый элемент больше самого большого
        if k > self.keys[leaf.pointers[high]]:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, page_len, key_ptr)
            return (key_ptr, None)
        if k == self.keys[leaf.pointers[high]]:  # уникальность
            return (None, leaf.pointers[high])

        while low < high:
            mid = (high + low) // 2
            if self.keys[leaf.pointers[mid]] == k:  # уникальность
                return (None, leaf.pointers[mid])
            if self.keys[leaf.pointers[mid]] > k:
                high = mid
            else:
                low = mid
            if mid == (high + low) // 2:  # если среднее не изменится
                break

        key_ptr = len(self.keys)
        self.keys.append(k)
        leaf.pointers = numpy.insert(leaf.pointers, high, key_ptr)
        return (key_ptr, None)

    def insert(self, key: KeyType) -> Tuple[int | None, int | None]:
        """
            Вставка ключа: возвращаем индекс в массиве ключей в случае удачи либо None
            - поиск листового узла, куда вставим новый объект
        """
        path: list[int] = []  # здесь запоминаем индексы узлов, по которым будем проходить
        node: BPTree.Node = self.tree[self.root_node]
        while not node.is_leaf():  # Если внутренний узел, то ищем листовой

            path.append(cast(int, node.node_index))

            low, high = 0, len(node.pointers) - 1

            if key == self.keys[node.pointers[low]]:
                return (None, node.pointers[low])  # найден дубликат

            if key < self.keys[node.pointers[low]]:  # меньше самого маленького
                node: BPTree.Node = self.tree[cast(BPTree.INode, node).descendants[0]]
                continue

            if key == self.keys[node.pointers[high]]:
                return (None, node.pointers[high])  # найден дубликат

            if key > self.keys[node.pointers[high]]:  # больше самого большого
                # переход на самую последнюю страницу в ссылках
                node: BPTree.Node = self.tree[cast(BPTree.INode, node).descendants[-1]]
                continue

            while low < high:  # ключ где-то в середине
                mid = (low + high) // 2
                if key == self.keys[node.pointers[mid]]:
                    return (None, node.pointers[mid])  # найден дубликат
                if key < self.keys[node.pointers[mid]]:
                    high = mid
                else:   # т.е.  key > self.keys[node.pointers[mid]]:
                    low = mid
                if mid == (low + high) // 2:
                    # если средний элемент не изменится
                    # получилось, что ключ находится между
                    # идем в сторону меньшего
                    node = self.tree[cast(BPTree.INode, node).descendants[high]]
                    break

        # листовой узел найден, можем вставлять
        return self.insert_into_leaf(cast(BPTree.LeafNode, node), key, path)

    def min(self) -> Tuple[LeafNode, KeyType]:
        node = cast(BPTree.Node, self.tree[self.root_node])
        while True:
            if node.is_leaf():
                return (cast(BPTree.LeafNode, node), self.keys[node.pointers[0]])
            else:
                node = cast(BPTree.INode, node)
                node = self.tree[node.descendants[0]]

    def max(self) -> Tuple[LeafNode, KeyType]:
        node = cast(BPTree.Node, self.tree[self.root_node])
        while True:
            if node.is_leaf():
                return (cast(BPTree.LeafNode, node), self.keys[node.pointers[-1]])
            else:
                node = cast(BPTree.INode, node)
                node = self.tree[node.descendants[-1]]

    def find(self, key: KeyType) -> Tuple[int | None, LeafNode | None, int | None]:
        """
            возвращает индекс ключа
            - несмотря на то, что нам не обязательно в некоторых случаях опускаться до листов дерева
              все равно делаем это - для реализации удаления
        """
        node = cast(BPTree.Node, self.tree[self.root_node])
        while True:
            if node.is_leaf():
                node = cast(BPTree.LeafNode, node)
                low, high = 0, len(node.pointers) - 1
                if self.keys[node.pointers[low]] == key:
                    return (node.pointers[low], node, low)
                if self.keys[node.pointers[high]] == key:
                    return (node.pointers[high], node, high)
                if key > self.keys[node.pointers[high]]:
                    return (None, None, None)
                if key < self.keys[node.pointers[low]]:
                    return (None, None, None)
                while low < high:
                    mid = (low + high) // 2
                    mid_key = self.keys[node.pointers[mid]]
                    if mid_key == key:
                        return (node.pointers[mid], node, mid)
                    if mid_key > key:
                        high = mid
                    else:
                        low = mid
                    if mid == (low + high) // 2:
                        return (None, None, None)
            else:
                low, high = 0, len(node.pointers) - 1
                node = cast(BPTree.INode, node)
                if key >= self.keys[node.pointers[high]]:
                    node = self.tree[node.descendants[high + 1]]
                    continue
                if key == self.keys[node.pointers[low]]:
                    node = self.tree[node.descendants[low + 1]]
                    continue
                if key < self.keys[node.pointers[low]]:
                    node = self.tree[node.descendants[low]]
                    continue

                while low < high:
                    node = cast(BPTree.INode, node)
                    mid = (low + high) // 2
                    if key == self.keys[node.pointers[mid]]:
                        node = self.tree[node.descendants[mid + 1]]
                        break
                    if key > self.keys[node.pointers[mid]]:
                        low = mid
                    else:
                        high = mid
                    if (low + high) // 2 == mid:  # если ключ больше не поменяется
                        node = self.tree[node.descendants[high]]  # то же что и self.tree[node.descendants[low + 1]]
                        break

    def validate_node(self, node_index: int) -> int:
        err_cnt = 0
        node = self.tree[node_index]
        for i, ptr in enumerate(node.pointers):
            if i == 0:  # пропускаем самый первый указатель
                continue
            prev_ptr = cast(int, node.pointers[i - 1])
            if (kn := self.keys[ptr]) <= (kp := self.keys[prev_ptr]):
                err_cnt += 1
                print(f"ERROR: wrong order: node_index: {node.node_index}/{i - 1}/{i}: {kp} >= {kn}")

        if node.prev_node is not None:
            prev_node = cast(BPTree.Node, self.tree[node.prev_node])
            if (kp := self.keys[prev_node.pointers[-1]]) >= (kn := self.keys[node.pointers[0]]):
                err_cnt += 1
                print(f"ERROR: wrong node linking: {node.node_index}/{prev_node.node_index}: {kp}, {kn}")
        return err_cnt

    def validate(self) -> dict[str, str]:
        """в случае успеха - возвращается пустой словарь (т.е. без ошибок)"""
        result: dict[str, str] = {}

        prev_references: set[int] = set()
        next_references: set[int] = set()

        num_first_leaves = 0  # must became 1 at the end of checking
        num_last_leaves = 0   # must became 1 at the end of checking

        if len(self.tree) == 1:
            if not self.tree[0].is_leaf():
                result["The only leaf must be the root of index"] = "Failure"
            if self.tree[0].node_index != 0:
                result["The only leaf must have node_index = 0"] = "Failure"
        else:
            if type(self.tree[self.root_node]) is not BPTree.INode:
                result["Expected root node type must be INode"] = "Failure"

        for i, n in enumerate(self.tree):
            if i != cast(int, n.node_index):
                result["Wrong node index:"] = f"{i} -> {n.node_index}"

            if (err_cnt := self.validate_node(i)) > 0:
                result["Node currupted:"] = f"node:{i}, errors count:{err_cnt}"

            if n.is_leaf():
                if n.next_node is None:
                    num_last_leaves += 1
                if n.prev_node is None:
                    num_first_leaves += 1

            if n.prev_node and (n.prev_node not in prev_references):
                prev_references.add(n.prev_node)
            elif n.prev_node in prev_references:
                result["Too many prev_references for node:"] = f"{i} -> BAD: {n.prev_node}"
            elif n.prev_node:
                if not self.tree[n.prev_node].next_node == n.node_index:
                    result["Wrong reference to a previous node:"] = f"{i} -> {n.prev_node}"

            if n.next_node and (n.next_node not in next_references):
                next_references.add(cast(int, n.next_node))
            elif n.next_node in next_references:
                result["Too many next_references for node:"] = f"{i} -> BAD: {n.next_node}"
            elif n.next_node:
                if not self.tree[n.next_node].prev_node == n.node_index:
                    result["Wrong reference to a next node:"] = f"{i} -> {n.next_node}"

        return result

    def print(self, msg: Any = "") -> None:
        """
            печатает все дерево - лучше не использовать
        """
        print(self.validate())
        for i, n in enumerate(self.tree):
            self.print_node(n)
        print("\u2500" * 120)
        node, k = self.min()
        print("first leaf:", node)
        print("min key:", k)
        print(f"-> {msg}")


if __name__ == "__main__":
    import faker
    BPTree.HALF_INODE_SIZE = 5
    BPTree.MAX_INODE_SIZE = BPTree.HALF_INODE_SIZE * 2
    BPTree.HALF_LEAF_SIZE = 5
    BPTree.MAX_LEAF_SIZE = BPTree.HALF_LEAF_SIZE * 2

    class MyKey(msgspec.Struct, frozen=True, order=True):
        full_name: str

    class Data(msgspec.Struct):
        data: Any

    index = BPTree[MyKey, Data]()
    faker.Faker.seed(1)
    fake = faker.Faker("ru_RU")
    numpy.random.seed(1)

    data = []

    for i in tqdm(range(20000)):
        val = numpy.random.randint(0, 999999)
        k = MyKey(f"{val:06}")
        index.insert(k)
        print(data.append(k), k)

    print(index.min()[1])
    print(index.max()[1])

    index.print_node(index.tree[0])

    index.find(MyKey("7524"))

    print("\u2500" * 50)
    for k in data:
        id = index.find(k)[0]
        if id is not None:
            print(f"found: {id} -> {index.keys[id]}")
        else:
            index.find(k)
            print(f"Key: {k} not found")
