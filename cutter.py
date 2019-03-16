# Original Code by Chengyao Zhang
# Modified by Zixiang Liu
# Mar 15, 2019
# Python 3.7

from queue import Queue

class cutter:
    def __init__(self, file_name = 'cora.cites'):
        '''
        initailize variables and read graph from file
        '''

        self.file_name = file_name
        # a dictionary with key = id of paper, value = ids of papers cited by key
        self.original_graph = {}
        # a dictionary with key = id of paper, value = ids of papers citing key
        self.reverse_graph = {}
        # remove nodes if not used
        self.nodes = set()
        # how many nodes citing key
        self.in_degree = {}
        # how many nodes citied by key
        self.out_degree = {}
        self.read_file()

    # this method stores graph and reverse graph from edges file
    def read_file(self):
        '''
        read from file
        Called in init
        '''
        with open(self.file_name, 'r') as file:
            for line in file:
                src = int(line.split()[1])
                dest = int(line.split()[0])
                if src not in self.original_graph:
                    self.original_graph[src] = []
                self.original_graph[src].append(dest)
                if dest not in self.reverse_graph:
                    self.reverse_graph[dest] = []
                self.reverse_graph[dest].append(src)
                self.nodes.add(src)
                self.nodes.add(dest)

                if dest not in self.in_degree:
                    self.in_degree[dest] = 1
                else:
                    self.in_degree[dest] += 1
                if src not in self.out_degree:
                    self.out_degree[src] = 1
                else:
                    self.out_degree[src] += + 1

        self.nodes = sorted(self.nodes)

    def cut(self, start = 56112, num_test = 270, verbose = True):
        '''
        cut the graph
        expand the sub_graph from a startin point
        '''
        count = 1
        sub_graph = {}

        # current node
        cur = start
        # queue
        q = Queue()
        # visited as a set
        visited = set()
        # add current to q and visited
        q.put(cur)
        visited.add(cur)

        while not q.empty():
            # pop q head
            cur = q.get()
            # if there are some papers citing current
            if self.reverse_graph.get(cur):
                parents = self.reverse_graph[cur]
                for parent in parents:
                    if parent not in visited:
                        sub_graph[parent] = []
                        # for all nodes cited by parent
                        for node in self.original_graph[parent]:
                            if node in visited:
                                sub_graph[parent].append(node)
                        # mark parent as visited and add count
                        visited.add(parent)
                        q.put(parent)
                        count += 1

            # if there are some papers cited by current
            if self.original_graph.get(cur):
                children = self.original_graph[cur]
                sub_graph[cur] = self.original_graph[cur]
                for child in children:
                    if child not in visited:
                        visited.add(child)
                        q.put(child)
                        count += 1

            if count >= num_test:
                break

        '''
        Explaination by 章承尧
        上面那个循环就是类似bfs，
        当队列里面的节点数超过了设置的测试节点数之后，就跳出了上面那个循环，
        这个时候所有在队列里的节点也是要被加到测试节点里的，
        所以这个循环是把所有这些节点连着的边都删掉，
        删掉之后可能会出现某些孤立的点（也就是入度和出度都是0的），
        所以就要把那些点也加到测试的子图里，
        这样才能保证挖出来的子图和剩下的那个子图都是联通的
        '''
        # after enough nodes in sub_graph
        while not q.empty():
            cur = q.get()
            # if there are some papers citing current
            if self.reverse_graph.get(cur):
                parents = self.reverse_graph[cur]
                for parent in parents:
                    if parent not in visited:
                        self.out_degree[parent] -=  1
                        if self.out_degree[parent] == 0 and ((parent not in self.in_degree) or self.in_degree[parent] == 0):
                            sub_graph[parent] = self.original_graph[parent]
                            visited.add(parent)
                            count = count + 1

            # if there are some papers cited by current
            if self.original_graph.get(cur):
                children = self.original_graph[cur]
                for child in children:
                    if child not in visited:
                        self.in_degree[child] = self.in_degree[child] - 1
                        if self.in_degree[child] == 0 and ((child not in self.out_degree) or self.out_degree[child] == 0):
                            if sub_graph.get(cur) == None:
                                sub_graph[cur] = []
                            sub_graph[cur].append(child)
                            if self.original_graph.get(child) != None:
                                sub_graph[child] = self.original_graph[child]
                            visited.add(child)
                            count = count + 1
        if verbose:
            print(count)
            print(len(visited))
            print(sub_graph)
        return sub_graph.keys()

if __name__ == "__main__":
    c = cutter(file_name = 'cora.cites')
    c.cut()
