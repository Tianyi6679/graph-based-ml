# Zixiang liu
# CS 249 Final Project
# Python 3.7
import numpy as np
import time
import cutter
from sklearn.naive_bayes import MultinomialNB


'''
UPDATE LOG:

IMPORTANT: now papers are stored in dictionary
'''
class MRF:
    '''
    Model of MRF
    '''
    def __init__(self, cite_file = 'cora.cites', word_file = 'cora.content', use_cutter = False):
        self.cite_file = cite_file
        self.word_file = word_file
        self.class_names = sorted(['Case_Based', 'Genetic_Algorithms',
               'Neural_Networks', 'Probabilistic_Methods',
               'Reinforcement_Learning', 'Rule_Learning', 'Theory'])
        self.num_class = len(self.class_names)
        self.load_data(use_cutter = use_cutter)
        #potential function
        self.theta = np.random.uniform(size=(self.num_class*2,self.num_class))

    def load_data(self, use_cutter = False):
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

        # papers are stored as dictionary of (key, list)
        # key is index
        # for the list
        #   first element is index,
        #   second element is name(also in int)
        #   third element is list of word count
        #   fourth element is label
        # [A, B, C, D] where A is int, B is int, C is np.array of int, D is string
        self.papers = {}

        # name_dict is a dictionary of name to index
        # name_dict[name(int)] = index
        self.name_dict = {}
        with open(self.word_file, 'r') as f:
            for i, line in enumerate(f):
                ll = line.split('\t')
                self.papers[i] = [i, int(ll[0]), np.array(ll[1:-1]).astype(int), ll[-1].rstrip('\n')]
                self.name_dict[int(ll[0])] = i
        # There are 2708 papers

        self.num_paper = len(self.papers)

        # edge matrix
        # for [i, j] if edge[i, j] == 1 then i cites j, else i does not
        # i, j are index of papers
        self.edge = np.zeros(shape=(self.num_paper, self.num_paper))
        for e in self.edges:
            self.edge[self.name_dict[e[0]], self.name_dict[e[1]]] = 1

        # if we use cutter to create train and test
        if use_cutter:
            self.cutter = cutter.cutter(file_name = self.cite_file)
            # change here for the amount of nodes to be cut as test
            test_nodes = self.cutter.cut(start = 56112, num_test = 200, verbose = False)
            # divide all papers into two
            self.train_papers = {}
            self.test_papers = {}
            for p in self.papers.values():
                if p[1] in test_nodes:
                    self.test_papers[p[0]] = p
                else:
                    self.train_papers[p[0]] = p
            self.train_edges = []
            self.test_edges = []
            self.train_edge = np.zeros(shape=(self.num_paper, self.num_paper))
            self.test_edge = np.zeros(shape=(self.num_paper, self.num_paper))
            for e in self.edges:
                if (e[0] in test_nodes) or (e[1] in test_nodes):
                    self.test_edges.append(e)
                    self.test_edge[self.name_dict[e[0]], self.name_dict[e[1]]] = 1
                else:
                    self.train_edges.append(e)
                    self.train_edge[self.name_dict[e[0]], self.name_dict[e[1]]] = 1
        else:
            self.train_papers = self.papers
            self.train_edges = self.edges
            self.train_edge = self.edge
            self.test_papers = None
            self.test_edges = None
            self.test_edge = None
        self.num_train_paper = len(self.train_papers)

    def pos_neighbor(self, index, edge):
        '''
        return a np.array of indexs of papers cited by the node
        '''
        return np.where(edge[index]==1)[0]

    def neg_neighbor(self, index, edge):
        '''
        return a np.array of indexs of papers which cite this node
        '''
        return np.where(edge[:, index]==1)[0]

    def neighbors(self, index, edge):
        '''
        return a np.array of indexs of papers cited or citing this node
        '''
        return np.unique(np.concatenate((self.pos_neighbor(index, edge), self.neg_neighbor(index, edge))))

    def query_class(self, node, cls, proba, papers, edge, asymmetric = False, lamb = 0.5):
        '''
        query the probability of the index node of being a specific class
        args:
            node:        the node of interest, in tuple form
            cls:         class name of interest
            proba:       probability of index node being this class given by classifier
            asymmetric:  True to use asymmetric potential function
            lamb:        smoothing factor lambda
            c:           Num of classes
        returns:
            float: score of this class
        '''
        index = node[0]
        phi = 0
        num_neighbor = self.neighbors(index, edge).shape[0]
        label = self.class_names.index(cls)
        c = self.num_class
        if asymmetric:
            posn = self.pos_neighbor(index, edge)
            negn = self.neg_neighbor(index, edge)
            for i in posn:
                n_label = self.class_names.index(papers[i][3])
                phi += self.theta[n_label*2][label]
                # comment above and uncomment below to not use learned weights
                # phi += 1 if papers[i][3] == cls else 0
            for i in negn:
                n_label = self.class_names.index(papers[i][3])
                phi += self.theta[n_label*2+1][label]
                # comment above and uncomment below to not use learned weights
                # phi += 0.75 if papers[i][3] == cls else 0
        else:
            n = self.neighbors(index, edge)
            for i in n:
                #n_label = self.class_names.index(self.train_papers[i][3])
                #phi += (self.theta[n_label * 2 + 1][label] + self.theta[n_label * 2][label]) / 2
                phi += 1 if papers[i][3] == cls else 0

        #phi = (phi + lamb/c)/(lamb + num_neighbor)
        return proba*np.exp(phi)

    def train(self, iter= 1000, lr = 0.001, threshold = 1e-8):
        '''
        learn theta
        '''
        # store train papers into a list
        # reassign train index that is only used in this method
        # also have dictionaries to convert between class index (index used in other part of the class) and train index
        train_papers_list = []
        ci2ti = {} # class index to train index
        ti2ci = {} # train index to class index

        for i, paper in enumerate(self.train_papers.values()):
            ti = i # train index
            ci = paper[0] # class index
            ci2ti[ci] = ti
            ti2ci[ti] = ci
            # reassign train index
            temp = paper[:]
            temp[0] = ti
            train_papers_list.append(temp)

        # positive neighbor and negative neighbor count
        X = np.zeros((len(train_papers_list),14))
        # all neighbor counts
        y = np.zeros((len(train_papers_list), 7))
        neighbors_num_for_each_class = np.zeros((7,14))
        report_interval = iter/10

        for node in train_papers_list:
            ti, _, _, cls = node
            posn = self.pos_neighbor(ti2ci[ti], self.train_edge)
            negn = self.neg_neighbor(ti2ci[ti], self.train_edge)
            for ci in posn:
                X[ti][self.class_names.index(train_papers_list[ci2ti[ci]][3])*2] += 1
                neighbors_num_for_each_class[self.class_names.index(train_papers_list[ti][3])][self.class_names.index(train_papers_list[ci2ti[ci]][3])*2]+=1
            for ci in negn:
                X[ti][self.class_names.index(train_papers_list[ci2ti[ci]][3])*2+1] += 1
                neighbors_num_for_each_class[self.class_names.index(train_papers_list[ti][3])][self.class_names.index(train_papers_list[ci2ti[ci]][3])*2+1]+=1
            y[ti][self.class_names.index(cls)] = 1

            if np.sum(X[ti]) == 0:
                print('isolated node, index:', ti )
                X[ti][self.class_names.index(cls)*2] = 1
                X[ti][self.class_names.index(cls)*2+1] = 1

        import util
        prev_loss = None
        print("start training, max iter:", iter, "learning rate:", lr)
        for epoch in range(iter):
            # gradient ascent using pseudo likelihood
            # calculate the probability for each node to be every possible class
            product = X.dot(self.theta)
            phi = np.exp(product)
            proba_for_each_node = util.normalize(phi)
            grad = neighbors_num_for_each_class-proba_for_each_node.T.dot(X)
            self.theta += lr*grad.T         
            a = X.dot(self.theta)
            z = a.sum(axis=1)
            yhat = util.normalize(a)

            loss = np.power(yhat-y,2).mean()
            '''
            for i in range(7):
                #grad =(X*(((z - a.T[i]) / np.power(z,2)).reshape(len(X),1))).T.dot((yhat-y).T[i])
                grad = ((X * ((z - a.T[i]) / np.power(z, 2)).reshape(len(X), 1) * self.theta.T[i])).T.dot(
                    (yhat - y).T[i])
                self.theta.T[i] -= lr*grad
            '''
            #self.theta -= lr*grad
            if epoch % report_interval == report_interval - 1:
                print("iter:",epoch+1,"loss:",loss)
            if prev_loss and (prev_loss-loss) <= threshold:
                print("converge at iter", epoch, "loss:", loss)
                break

            prev_loss = loss
        #self.theta = np.power(self.theta, 2)

    def evaluate(self, max_iter = 100, method = 'gibbs', asymmetric = True, verbose = False, use_classifier = True):
        '''
        predict the labels for test set and evaluate result
        args:
            max_iter: max number of iterations
            method: only implemented gibbs sampling
            asymmetric: whether or not use asymmetric potential, true to use asymmetric potential functions
            verbose: true to print more informations
            use_classifier: true to use classifier
        return:
            the accuracy of prediction on test set
        '''
        if not self.test_edges:
            print('No cutter program, cannot create train/test split.')
            return

        print('Predicting test set with {} potential function.'.format('asymmetric' if asymmetric else 'symmetric'))
        if method == 'gibbs':
            # use gibbs sampling

            # initialize known nodes and edge to train set
            all_papers = self.train_papers
            all_edge = self.train_edge
            # test if label has changed for two iterations
            changed = False

            # iteratively assign labels to node
            for i in range(max_iter):
                for p in self.test_papers.values():
                    # initialize with node without label
                    paper = p[:]
                    paper[3] = ''
                    if p[0] not in all_papers.keys():
                        # update papers
                        all_papers[p[0]] = paper
                        # update edge matrix
                        for e in self.test_edges:
                            if (self.name_dict[e[0]] in all_papers.keys()) and (self.name_dict[e[1]] in all_papers.keys()):
                                all_edge[self.name_dict[e[0]], self.name_dict[e[1]]] = 1
                        # this update only happens in first iteration, so it must have changed
                        changed = True

                    label = self.predict_node(
                        paper, all_papers, all_edge,
                        asymmetric = asymmetric,
                        verbose = verbose,
                        use_classifier = use_classifier
                    )
                    old_label = all_papers[paper[0]][3]
                    all_papers[paper[0]][3] = label
                    if not changed and label != old_label:
                        changed = True
                    if verbose:
                        print('Iter. {}'.format(i))
                if not changed:
                    print('Assignment of Labels remain the same. End iteration at {}.'.format(i))
                    break

            self.num_test = len(self.test_papers)
            correct_cnt = 0
            for p in self.test_papers.values():
                correct_cnt += 1 if p[3] == all_papers[p[0]][3] else 0
            acc = float(correct_cnt)/self.num_test
            print('Test Accuracy {}/{}: {:.2f}%'.format(correct_cnt, self.num_test, acc*100))
            return acc

    def predict_node(self, paper, papers, edge, asymmetric = True, verbose = True, use_classifier = True):
        '''
        predict the label of one node
        args:
            p: the node of interest
            papers: all papers available for calculation
            edge: all edges available for calculation
            asymmetric: True to use asymmetric potential function
            verbose: to print more output
        '''
        label = None
        max_p = 0
        proba_list = []
        if use_classifier:
            proba_list = self.get_classifier().predict_proba([paper[2]])[0]

        for i, cls in enumerate(self.class_names):
            # if we do not use classifier, simply use a normalize factor as probability
            proba = 1/self.num_class
            # if we use classifier, the probability is given by classifier
            if use_classifier:
                proba = proba_list[i]
            # query on the class
            pred_p = self.query_class(
                paper,
                cls, proba, papers, edge,
                asymmetric = asymmetric
            )
            if pred_p > max_p:
                label = cls
                max_p = pred_p
        if not label:
            if verbose:
                print('Node index: {}, ID: {} has no query result > 0. Use classifier result.'.format(paper[0], paper[1]))
            label = self.class_names[np.argmax(proba_list)]
        return label

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
            print(self.train_papers[index])
            print(self.pos_neighbor(self.train_papers[index][0]), self.train_edge)
            print(self.neg_neighbor(self.train_papers[index][0]), self.train_edge)

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
            asy = self.query_class(self.train_papers[index], cls, proba, self.train_papers, self.train_edge, asymmetric = True)
            sym = self.query_class(self.train_papers[index], cls, proba, self.train_papers, self.train_edge, asymmetric = False)
            if verbose:
                print('{}\nsymmetric: {:.5f} symmetric: {:.5f}'.format(cls, asy, sym))
            if asy > asym:
                asyl = cls
                asym = asy
            if sym > symm:
                syml = cls
                symm = sym

        rtn = (asyl == self.train_papers[index][3], syml ==  self.train_papers[index][3])
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
        interval = int(len(self.train_papers)/10)
        start_time = time.time()
        for i in self.train_papers.keys():
            out = self.test_node(i, False, use_classifier = use_classifier)
            asyout += 1 if out[0] else 0
            symout += 1 if out[1] else 0
            if i % interval == interval-1:
                print('Test nodes {}% Done, used {:.3f}s'.format(pct, time.time()-start_time))
                pct += 10
                start_time = time.time()

        print('Asymmetric prodiction correct for {}/{} papers, {:.4f}%\nsymmetric perdiction correct for {}/{} papers, {:.4f}%'.format(
            asyout, self.num_train_paper,
            asyout/self.num_train_paper * 100,
            symout, self.num_train_paper,
            symout/self.num_train_paper * 100
        ))

    def classifier(self, verbose = False):
        '''
        Every basic Multinomial Naive Bayes classifier
        Use whole data as train
        And predict probability for each node under each class
        '''
        # generate train set(word counts and labels)
        X = []
        y = []
        index_dict = {}

        for i, p in enumerate(self.train_papers.values()):
            X.append(p[2])
            y.append(p[3])
            index_dict[i] = p[0]

        # use the very basic classifier
        self.nb = MultinomialNB()

        # train using all data
        self.nb.fit(X, y)

        # predict the probability for each node
        self.y_pred = {}
        pred = self.nb.predict_proba(X)
        for i, label in enumerate(pred):
            self.y_pred[index_dict[i]] = label
        if verbose:
            print('Accuracy of classifier: {:.5f}'.format(self.nb.score(X, y)))

    def get_classifier(self):
        '''
        return the classifier object
        '''
        if not self.nb:
            self.classifier()
        return self.nb

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
        if verbose:
            print('Training set has {} papers'.format(self.num_train_paper))
        for i in self.train_papers.keys():
            pl = self.pos_neighbor(self.train_papers[i][0], self.train_edge).shape[0]
            nl = self.neg_neighbor(self.train_papers[i][0], self.train_edge).shape[0]
            al = self.neighbors(self.train_papers[i][0], self.train_edge).shape[0]
            if al != 0:
                bias += abs(pl - nl)/al
            else:
                if verbose:
                    print('Paper index: {}, ID: {} has no neighbor'.format(i, self.train_papers[i][1]))
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
    mrf = MRF(cite_file = 'cora.cites', word_file = 'cora.content', use_cutter = True)
    mrf.analyze_data()
    mrf.train()
    print(mrf.theta)
    mrf.test_all()
    mrf.evaluate(asymmetric = True)
    mrf.evaluate(asymmetric = False)
