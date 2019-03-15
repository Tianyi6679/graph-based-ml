class node:
    def __init__(self, paperID):
        self.id = paperID
        self.citing = set()
        self.children = set()
    def cite(self, paperID):
        self.citing.add(paperID)
    def print_node(self):
        print("entering paperID", self.id, "citing:", end=' ')
        for n in self.citing:
            print(n.id, end = ' ')
        print()
    def print_children(self):
        print("entering paperID", self.id, "children:", end=' ')
        for n in self.children:
            print(n.id, end=' ')
        print()

def trim_overcitation(paper, visited, count):
    '''
    paper: current node
    visited: the set keeps track of visited node
    count: IGNORE IT
    '''
    paper.print_node()
    if len(paper.citing) > 0:
        for n in paper.citing:
            if n not in visited:
                visited.add(n)
                paper.children = paper.children.union(trim_overcitation(n, visited, count))
            else:
                paper.children = paper.children.union(n.children)
    else:
        return paper.children

    #NOTE: At this point, the children set of this node have not included its own direct children yet,
    #      therefore the intersection indicates over-citations
    count[0] += len(paper.citing.intersection(paper.children))
    paper.citing = paper.citing.difference(paper.children)

    paper.children = paper.children.union(paper.citing)
    return paper.children

def trim_loops(paper, root_paper, visited, parent, count):
    '''
    paper: current node
    root_paper: originally equals to whole graph,
                but we will remove cited node from it as we visit each node.
                Finally, only uncited node are left.
    visited: the set keeps track of visted node
    parent: the set keeps track of current node's parent(s)
    count: [number of overcitations, number of loops], you CAN IGNORE IT
    '''
    for cited in paper.citing:
        #Intersection between the parent and its citing node indicates loop 
        count[1] += len(cited.citing.intersection(parent))
        cited.citing = cited.citing.difference(parent)
        visited.add(cited)
        if cited in root_paper:
            root_paper.remove(cited)
        #Depth-first-search to capture and remove any loops along the path
        parent.add(cited)
        trim_loops(cited, root_paper, visited, parent, count)
        parent.remove(cited)
    return

def trim(prune_overcitation = False):
    #Generate the graph based on the citation links provided in cora.cites
    graph = {}
    cited = set()
    with open('cora.cites') as file:
        for line in file:
            a, b = line.split()
            if a not in graph:
                graph[a] = node(a)
            if b not in graph:
                graph[b] = node(b)
            graph[b].cite(graph[a])
            cited.add(graph[a])
    print(len(graph))
    #Root_papers are papers which is not cited by any other paper
    root_paper = set(graph.values())
    count = [0, 0]
    
    #Starting from each unvisited paper to remove loops in the graph 
    visited = set()
    parent = set()
    for paper in graph.values():
        if paper not in visited:
            parent.add(paper)
            trim_loops(paper, root_paper, visited, parent, count)
            parent.remove(paper)
    print(len(root_paper))
    
    #Overcitations refer to the situation where a parent paper and its child paper cite a same paper
    #and in that case, only child's citation link is kept.
    #NOTE: papers' children don't have to be direct. The children of a paper's direct child are still counted as its children
    if prune_overcitation:
        visited = set()
        for paper in root_paper:
            trim_overcitation(paper, visited, count)


    print("delete over-citations:", count[0], "delete loops:", count[1])
    with open("cora_trimmed.cites","w+") as file:
        for paper in graph.values():
            for cited in paper.citing:
                file.write('{}\t{}\n'.format(cited.id, paper.id))

if __name__ == "__main__":
    trim(prune_overcitation = True)
