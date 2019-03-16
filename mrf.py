# Zixiang liu
# CS 249 Final Project
# Python 3.7
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB

class MRF:
    def __init__(self, cite_file = 'cora.cites', word_file = 'cora.content'):
        self.cite_file = cite_file
        self.word_file = word_file
        self.class_names = sorted(['Case_Based', 'Genetic_Algorithms',
               'Neural_Networks', 'Probabilistic_Methods',
               'Reinforcement_Learning', 'Rule_Learning', 'Theory'])
        self.num_class = len(self.class_names)
        self.load_data()
        #potential function
        self.theta = np.random.uniform(size=(14,7))
    def load_data(self):
        '''
        This method load the data into predefined data structures
        edges, is a list of tuples
        and then it is converted into
        edge, which is a matrix
        papers, a list of tuples to store all paper informations

        each paper in this class is assigned an index so that the matrix is easier to query
        '''
        # edges are stored as a list of tuples, first element cites the second one
        # (A, B) A cites B, both A and B are integers
        self.edges = []
        with open(self.cite_file, 'r') as f:
            for line in f:
                ll = line.split('\t')
                self.edges.append((int(ll[0]), int(ll[1])))
        # There are 5429 edges

        # papers are stored as a list of tuples,
        # first element is index,
        # second element is name(also in int)
        # third element is list of word count
        # fourth element is label
        # (A, B, C, D) where A is int, B is int, C is np.array of int, D is string
        self.papers = []

        # name_dict is a dictionary of name to index
        # name_dict[name(int)] = index
        self.name_dict = {}
        with open(self.word_file, 'r') as f:
            for i, line in enumerate(f):
                ll = line.split('\t')
                self.papers.append((i, int(ll[0]), np.array(ll[1:-1]).astype(int), ll[-1].rstrip('\n')))
                self.name_dict[int(ll[0])] = i
        # There are 2708 papers

        self.num_paper = len(self.papers)

        # edge matrix
        # for [i, j] if edge[i, j] == 1 then i cites j, else i does not
        # i, j are index of papers
        self.edge = np.zeros(shape=(self.num_paper, self.num_paper))
        for e in self.edges:
            self.edge[self.name_dict[e[0]], self.name_dict[e[1]]] = 1

    def pos_neighbor(self, index):
        '''
        return a np.array of indexs of papers cited by the node
        '''
        return np.where(self.edge[index]==1)[0]

    def neg_neighbor(self, index):
        '''
        return a np.array of indexs of papers which cite this node
        '''
        return np.where(self.edge[:, index]==1)[0]

    def neighbors(self, index):
        '''
        return a np.array of indexs of papers cited or citing this node
        '''
        return np.unique(np.concatenate((self.pos_neighbor(index), self.neg_neighbor(index))))

    def query_class(self, node, cls, p, asymmetric = False, lamb = 0.5, c = 7):
        '''
        query the probability of the index node of being a specific class
        args:
            node:        the node of interest, in tuple form
            cls:         class name of interest
            p:           probability of index node being this class given by classifier
            asymmetric:  True to use asymmetric potential function
            lamb:        smoothing factor lambda
            c:           Num of classes
        returns:
            float: score of this class
        '''
        index = node[0]
        phi = 0
        num_neighbor = self.neighbors(index).shape[0]
        label = self.class_names.index(cls)
        if asymmetric:
            posn = self.pos_neighbor(index)
            negn = self.neg_neighbor(index)
            for i in posn:
                n_label = self.class_names.index(self.papers[i][3])
                phi += self.theta[n_label*2][label]
            for i in negn:
                n_label = self.class_names.index(self.papers[i][3])
                phi += self.theta[n_label*2+1][label]
        else:
            n = self.neighbors(index)
            for i in n:
                #n_label = self.class_names.index(self.papers[i][3])
                #phi += (self.theta[n_label * 2 + 1][label] + self.theta[n_label * 2][label]) / 2
                phi += 1 if self.papers[i][3] == cls else 0

        phi = (phi + lamb/c)/(lamb + num_neighbor)
        return p*phi

    def train(self, iter= 1000, lr = 0.005, threshold = 1e-10):
        X = np.zeros((len(self.papers),14))
        y = np.zeros((len(self.papers), 7))

        for node in self.papers:
            index, _, _, cls = node
            posn = self.pos_neighbor(index)
            negn = self.neg_neighbor(index)
            for i in posn:
                X[index][self.class_names.index(self.papers[i][3])*2] += 1
            for i in negn:
                X[index][self.class_names.index(self.papers[i][3])*2+1] += 1
            y[index][self.class_names.index(cls)] = 1
        import util
        prev_loss = None
        print("start training, max iter:", iter, "learning rate:", lr)
        for epoch in range(iter):
            a = X.dot(np.power(self.theta,2))
            z = a.sum(axis=1)
            yhat = util.normalize(a)

            loss = np.power(yhat-y,2).mean()
            for i in range(7):
                grad = ((X*((z - a.T[i]) / np.power(z,2)).reshape(len(X),1)*self.theta.T[i])).T.dot((yhat-y).T[i])
                self.theta.T[i] -= lr*grad

            #self.theta -= lr*grad
            if epoch % 100 == 0:
                print("iter:",epoch,"loss:",loss)
            if prev_loss and (prev_loss-loss) <= threshold:
                print("converge at iter", epoch, "loss:", loss)
                break

            prev_loss = loss
        self.theta = np.power(self.theta,2)
        print(self.theta.T)

    def test_node(self, index, verbose = True, use_classifier = False):
        '''
        Test the prediction by Asymmetric potential function and symmetric potential function
        use verbose to print output
        the prediciton is based on the rest of the network
        args:
            index: index of paper of interest
            verbose: print or not
            use_classifier: True to use classifier, false to ignore it
        returns:
            tuple of result by two potential functions
        '''
        if verbose:
            print(self.papers[index])
            print(self.pos_neighbor(self.papers[index][0]))
            print(self.neg_neighbor(self.papers[index][0]))

        asyl = ''
        asym = 0
        syml = ''
        symm = 0
        for i, cls in enumerate(self.class_names):
            # if we do not use classifier, simply use a normalize factor as probability
            proba = 1/self.num_class
            # if we use classifier, the probability is given by classifier
            if use_classifier:
                proba = self.y_pred[index][i]
            asy = self.query_class(self.papers[index], cls, proba, asymmetric = True)
            sym = self.query_class(self.papers[index], cls, proba, asymmetric = False)
            if verbose:
                print('{}\nsymmetric: {:.5f} symmetric: {:.5f}'.format(cls, asy, sym))
            if asy > asym:
                asyl = cls
                asym = asy
            if sym > symm:
                syml = cls
                symm = sym

        rtn = (asyl == self.papers[index][3], syml ==  self.papers[index][3])
        if verbose:
            print('Asymmetric correct: {}, symmetric correct: {}'.format(rtn[0], rtn[1]))
        return rtn

    def test_all(self, use_classifier = True):
        '''
        test all nodes
        args:
            use_classifier: True to use classifier, false to ignore it
        '''
        self.classifier(verbose = True)
        asyout = 0
        symout = 0
        pct = 10
        interval = int(self.num_paper/10)
        start_time = time.time()
        for i in range(self.num_paper):
            out = self.test_node(i, False, use_classifier = use_classifier)
            asyout += 1 if out[0] else 0
            symout += 1 if out[1] else 0
            if i % interval == interval-1:
                print('{}% Done, used {:.3f}s'.format(pct, time.time()-start_time))
                pct += 10
                start_time = time.time()
        print('Asymmetric prodiction correct for {}/{} papers, {:.4f}%\nsymmetric perdiction correct for {}/{} papers, {:.4f}%'.format(
            asyout, self.num_paper,
            asyout/self.num_paper * 100,
            symout, self.num_paper,
            symout/self.num_paper * 100
        ))

    def classifier(self, verbose = False):
        '''
        Every basic Multinomial Naive Bayes classifier
        Use whole data as train
        And predict probability for each node under each class
        '''
        # generate train set(word counts and labels)
        X = np.array([x[2] for x in self.papers])
        y = [x[3] for x in self.papers]

        # use the very basic classifier
        nb = MultinomialNB()

        # train using all data
        nb.fit(X, y)

        # predict the probability for each node
        self.y_pred = nb.predict_proba(X)
        if verbose:
            print('Accuracy of classifier: {:.5f}'.format(nb.score(X, y)))

    def analyze_data(self, verbose = True):
        '''
        find papers which cites and is cited by the same paper
        store the indexs into a list, circle_paper
        For the cora we have, there are 276 papers has this issue

        also find the average ratio of bias in citing, where
        bias: the absolute value of difference in pos_neighbor and neg_neighbor, divide by all neighbors
        The closer bias to 0, means the number of pos_neighbor and neg_neighbor is balanced
        the closer to 1, means they are unbalanced.

        and the third concern is on:
        pos_paper: pos_neighbor > neg_neighbor
        neg_paper: neg_neighbor > pos_neighbor
        '''
        self.circle_paper = []
        bias = 0
        pos_paper = 0
        pos_rate = 0
        neg_paper = 0
        neg_rate = 0
        eql_paper = 0
        for i in range(self.num_paper):
            pl = self.pos_neighbor(self.papers[i][0]).shape[0]
            nl = self.neg_neighbor(self.papers[i][0]).shape[0]
            al = self.neighbors(self.papers[i][0]).shape[0]
            bias += abs(pl - nl)/al
            if pl+nl != al:
                self.circle_paper.append(i)
            if pl > nl:
                pos_paper += 1
                pos_rate += pl - nl
            elif pl < nl:
                neg_paper += 1
                neg_rate += nl - pl
            else:
                eql_paper += 1
        self.num_circle = len(self.circle_paper)
        self.bias = bias/self.num_paper

        if verbose:
            print('Circle papers: {}, bias = {}'.format(
                self.num_circle, self.bias
            ))
            print('{} papers has average {:.2f} more pos_neighbors\n{} has average {:.2f} more neg_neighbors\n{} papers are balanced'.format(
                pos_paper, pos_rate/pos_paper, neg_paper, neg_rate/neg_paper, eql_paper
            ))

if __name__ == "__main__":
    '''
    Just test all nodes, see if any potential function is better
    '''
    mrf = MRF(cite_file='cora.cites')
    mrf.analyze_data()
    mrf.train()
    mrf.test_all()
