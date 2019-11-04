class Queue:
    def __init__(self, iter = []):
        self.first = None
        self.last = None
        self.backwards = False
        self._in_iter = False
        self.N = 0

        for item in iter:
            self.enqueue(item)

    def __iter__(self):
        self._in_iter = True
        node = self.first if not self.backwards else self.last
        while node != None:
            yield node.value
            node = node.succ if not self.backwards else node.pred
        self._in_iter = False

    def __len__(self):
        return self.N

    def reverse_iteration(self):
        if self._in_iter:
            raise "Cannot reverse when enumerating"
        self.backwards = not self.backwards

    def enqueue(self, value):
        if self._in_iter:
            raise "Cannot enqueue when enumerating"
        if self.last == None:
            self.first = self.last = Node(value)
        else:
            node = Node(value)
            node.pred = self.last
            self.last.succ = node
            self.last = node
        self.N += 1

    def dequeue(self):
        if self._in_iter:
            raise "Cannot dequeue when enumerating"
        node = self.first
        self.first = node.succ
        if self.first:
            self.first.pred = None

        self.N -= 1
        
        return node.value

class Node:
    def __init__(self, value):
        self.value = value
        self.succ = None
        self.pred = None

