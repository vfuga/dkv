# pyright: reportOperatorIssue=false
import msgspec
import numpy
# import heapq
# import pickle
from typing import Any, TypeVar, Optional, List, Generic, Type, cast, Tuple   # noqa
from abc import ABC                                                           # noqa


KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


class BPTree(Generic[KeyType, ValueType]):

    MAX_LEAF_SIZE = 4   # максимальное количество ключей на странице
    MAX_INODE_SIZE = 4
    HALF_LEAF_SIZE = MAX_LEAF_SIZE // 2  # при разбивании страницы попалам
    HALF_INODE_SIZE = MAX_INODE_SIZE // 2
    LEAF_NODE: bool = True
    INODE: bool = not LEAF_NODE

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
        print(f"{space}node-idx:{node.node_index}, prev:{node.prev_node}, next:{node.next_node}, level: {node.level}, ")
        if node.is_leaf():
            for i in node.pointers:
                print(f"{space}|-> {self.keys[i]}")
        else:
            n = cast(BPTree.INode, node)
            for i, p in enumerate(n.pointers):
                print(f"{space}-> {self.keys[p]}: {n.descendants[i]}, {n.descendants[i + 1]}")

    def insert_into_parent(self, inode: INode, k_ind: int, descendant: int, path) -> int | None:
        """
            Вставка ключа в родительский узел (в случае когда выполняется split)
            - если родительский узел существует, значит он всегда не пустой
            - когда родительский узел создан - в нем всегда есть хотя бы один ключ
        """
        low, high = 0, len(inode.pointers) - 1
        k = cast(KeyType, self.keys[k_ind])

        if len(inode.pointers) < self.MAX_INODE_SIZE:

            # # этот код никогда не будет выполняться, потому, что уникальность проверяется в процессе
            # # поиска места вставки
            # if k_ind == inode.pointers[high]:
            #     raise RuntimeError("Non unique index value")
            # if k_ind == inode.pointers[low]:
            #     raise RuntimeError("Non unique index value")

            if k < self.keys[inode.pointers[low]]:   # меньше самого маленького
                inode.pointers = numpy.insert(inode.pointers, 0, k_ind)
                inode.descendants = numpy.insert(inode.descendants, 1, descendant)
                return k_ind

            if k > self.keys[inode.pointers[high]]:  # больше самого большого
                inode.pointers = numpy.insert(inode.pointers, len(inode.pointers), k_ind)
                inode.descendants = numpy.insert(inode.descendants, len(inode.pointers), k_ind)
                return k_ind

            while low < high:
                # после этого цикла у нас только два варианта - два ключа,
                # между которыми мы должны вставить еще один ключ (вместо high)

                mid = (high + low) // 2
                if k > self.keys[mid]:
                    high = mid
                if k < self.keys[mid]:
                    low = mid
                if mid == (high + low) // 2:  # если mid не изменится
                    break

            inode.pointers = numpy.insert(inode.pointers, high, k_ind)  # +
            inode.descendants = numpy.insert(inode.descendants, high + 1, descendant)
            return k_ind

        else:  # len(inode.pointers) >= self.MAX_INODE_SIZE:
            """
               рекурсивно расщепляем родительский узел
            """
            new_inode = cast(BPTree.INode, self.node_factory(leaf_node=False))
            new_inode.level = inode.level

            # горизонтальное связываение
            inode.next_node = new_inode.node_index
            new_inode.prev_node = inode.node_index

            new_inode.pointers = inode.pointers[:self.HALF_INODE_SIZE]
            inode.pointers = inode.pointers[self.HALF_INODE_SIZE + 1:]
            new_inode.descendants = inode.descendants[:self.HALF_INODE_SIZE + 1]
            inode.descendants = inode.descendants[self.HALF_INODE_SIZE:]
            if len(path) == 0:
                new_root = cast(BPTree.INode, self.node_factory(leaf_node=False))
                new_root.level = inode.level + 1
                new_root.pointers = inode.pointers[:1]
                new_root.descendants = numpy.array([new_inode.node_index, inode.node_index], numpy.int32)
                self.root_node = cast(int, new_root.node_index)

            raise NotImplementedError()

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
            n_leaf = cast(BPTree.LeafNode, self.node_factory(leaf_node=True))
            # копируем указатели
            n_leaf.pointers = leaf.pointers[self.HALF_LEAF_SIZE:]
            leaf.pointers = leaf.pointers[:self.HALF_LEAF_SIZE]

            parent_node = cast(BPTree.INode, self.tree[path.pop()])

            # рекурсивное обновление верхних узлов /узел/ключ/узел-потомок/стэк-узлов)
            self.insert_into_parent(parent_node, split_key_ind, cast(int, n_leaf.node_index), path)

            # горизонтальное связывание
            if leaf.next_node is not None:
                p_leaf = self.tree[leaf.next_node]
                p_leaf.prev_node = n_leaf.node_index
                n_leaf.next_node = p_leaf.node_index
            n_leaf.prev_node = leaf.node_index
            leaf.next_node = n_leaf.node_index

            return (leaf, n_leaf)

    def get_leaf(self, idx: int) -> LeafNode | None:   # noqa
        if self.tree[idx].level == 0:
            return cast(BPTree.LeafNode, self.tree[idx])
        return None

    def get_inode(self, idx: int) -> INode | None:   # noqa
        if self.tree[idx].level >= 1:
            return cast(BPTree.INode, self.tree[idx])
        return None

    def insert_into_leaf(self, leaf: LeafNode, k: KeyType, path: list[int]) -> int | None:
        """
            вставка в листовой узел.
            keys - список, куда будет добавлен новый ключ
            k - сам ключ
            возвращаем индекс добавленного ключа или None в случае неудачи
        """
        # сначала добавляем ключ к списку всех ключей индекса, т.е. он становится последним элементом списка
        # это нужно, чтобы получить указатель на новый ключ
        # если вставка не получится сделаем self.keys.pop()

        # узнаем, сколько элементов в узле уже есть
        page_len: int = len(leaf.pointers)

        if page_len >= self.MAX_LEAF_SIZE:  # если место в узле закончилось
            # расщепить страницы и вставить в ту страницу, куда полагается
            low_node, high_node = self.split_leaf(leaf, path)
            if k == self.keys[high_node.pointers[0]]:
                raise Exception("Этот код не должен выполняться")
            if k < self.keys[high_node.pointers[0]]:
                return self.insert_into_leaf(low_node, k, path)
            else:
                return self.insert_into_leaf(high_node, k, path)

        # если узел пустой
        if page_len == 0:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, 0, key_ptr)
            return key_ptr

        # Двоичный поиск места, куда можно вставить
        # самый большой и самый маленький элементы - в виде tuple
        # где первое поле - это номер элемента в массиве указателей
        low, high = 0, page_len - 1

        # если искомый элемент меньше самого маленького
        if k < self.keys[leaf.pointers[low]]:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, 0, key_ptr)
            return key_ptr
        if k == self.keys[leaf.pointers[high]]:  # уникальность
            return None

        # если искомый элемент больше самого большого
        if k > self.keys[leaf.pointers[high]]:
            key_ptr = len(self.keys)
            self.keys.append(k)
            leaf.pointers = numpy.insert(leaf.pointers, page_len, key_ptr)
            return key_ptr
        if k == self.keys[leaf.pointers[high]]:  # уникальность
            return None

        while low < high:
            mid = (high + low) // 2
            if self.keys[leaf.pointers[mid]] == k:  # уникальность
                return None
            if self.keys[leaf.pointers[mid]] > k:
                high = mid
            else:
                low = mid
            if mid == (high + low) // 2:  # если среднее не изменится
                break

        key_ptr = len(self.keys)
        self.keys.append(k)
        leaf.pointers = numpy.insert(leaf.pointers, high, key_ptr)
        return key_ptr

    def insert(self, key: KeyType) -> int | None:
        """
            Вставка ключа: возвращаем индекс в массиве ключей в случае удачи либо None
            - поиск листового узла, куда вставим новый объект
        """
        path: list[int] = []  # здесь запоминаем индексы узлов, по которым будем проходить
        node: BPTree.Node = self.tree[self.root_node]
        while not node.is_leaf():  # Если внутренний узел, то ищем листовой

            path.append(cast(int, node.node_index))

            low, high = 0, len(node.pointers) - 1

            if key < self.keys[node.pointers[low]]:  # меньше самого маленького
                node: BPTree.Node = self.tree[cast(BPTree.INode, node).descendants[0]]
                continue

            if key > self.keys[node.pointers[high]]:  # больше самого большого
                # переход на самую последнюю страницу в ссылках
                node: BPTree.Node = self.tree[cast(BPTree.INode, node).descendants[-1]]
                continue

            while low < high:  # ключ где-то в середине
                mid = (low + high) // 2
                if key == self.keys[node.pointers[mid]]:
                    # ключи точно совпадают (полагаем, что все ключи уникальные)
                    # т.е. такой ключ уже есть
                    return None
                if key < self.keys[node.pointers[mid]]:
                    high = mid
                else:   # т.е.  key > self.keys[node.pointers[mid]]:
                    low = mid
                if mid == (low + high) // 2:
                    # если средний элемент не изменится
                    # получилось, что ключ находится между
                    # идем в сторону меньшего
                    node = self.tree[cast(BPTree.INode, node).descendants[low + 1]]
                    break

        # листовой узел найден, можем вставлять
        return self.insert_into_leaf(cast(BPTree.LeafNode, node), key, path)

    def validate(self) -> dict[str, str]:
        """
            в случае успеха - возвращается пустой словарь
        """
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

            if n.is_leaf():
                if n.next_node is None:
                    num_last_leaves += 1
                if n.prev_node is None:
                    num_first_leaves += 1

            if n.prev_node and (n.prev_node not in prev_references):
                prev_references.add(n.prev_node)
            elif n.prev_node:
                result["Wrong reference to a previous node:"] = f"{i} -> {n.prev_node}"

            if n.next_node and (n.next_node not in next_references):
                next_references.add(cast(int, n.next_node))
            elif n.next_node:
                result["Wrong reference to a next node:"] = f"{i} -> {n.next_node}"

        return result


class PKey(msgspec.Struct, frozen=True, order=True):
    name: str
    pass


class Data(msgspec.Struct, frozen=False):
    name: str
    birth: str


# создаем индекс
index = BPTree[PKey, Data]()
# print(index.root, index.keys, index.tree)

if __name__ == "__main__":

    def print_index():
        print("\u2500" * 80)
        print(index.validate())
        for i, n in enumerate(index.tree):
            index.print_node(n)

    lnode = index.get_leaf(index.root_node)
    if lnode:
        index.insert(PKey("50"))
        index.insert(PKey("40"))
        index.insert(PKey("60"))
        index.insert(PKey("0"))

        print(index.insert(PKey("10")))

        print(index.insert(PKey("10")))
        print_index()
        print(index.insert(PKey("1")))
        print_index()
        print(index.insert(PKey("30")))
        print_index()

        print(index.insert(PKey("20")))
        print_index()
        print(index.insert(PKey("70")))
        print_index()
        print(index.insert(PKey("35")))   # неправильно меняет родительский узел
        print_index()
        print("\n\n" + str(index.insert(PKey("35"))))
        print_index()

        print(index.insert(PKey("10")))
        print(index.insert(PKey("10")))
        print(index.insert(PKey("30")))
        print_index()
        print(index.insert(PKey("45")))
        print_index()
        print(index.insert(PKey("48")))
        print_index()
        print(index.insert(PKey("46")))
        print_index()
        print(index.insert(PKey("41")))
        print_index()
        # print(index.insert(PKey("70")))
        # print(index.insert(PKey("70")))
        # print(index.insert(PKey("77")))
        # print(index.insert(PKey("79")))
        # print(index.insert(PKey("80")))
        # print(index.insert(PKey("80")))

        # lnode.print(index.keys)

        # index._leaf_insert(lnode, PKey("2"))
        # index._leaf_insert(lnode, PKey("A"))
        # # index._leaf_insert(lnode, PKey("B"))
        # # index._leaf_insert(lnode, PKey("A"))
        # # index._leaf_insert(lnode, PKey("C"))
        # # index._leaf_insert(lnode, PKey("D"))
        # # index._leaf_insert(lnode, PKey("E"))
