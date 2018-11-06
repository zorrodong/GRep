import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import scipy

def RandomWalkRestart(A, rst_prob, delta = 1e-4, reset=None, max_iter=50):
    nnode = A.shape[0]
    #print nnode
    if reset is None:
        reset = np.eye(nnode)
    nsample,nnode = reset.shape
    #print nsample,nnode
    P = renorm(A)
    P = P.T
    norm_reset = renorm(reset.T)
    norm_reset = norm_reset.T
    Q = norm_reset
    for i in range(1,max_iter):
        #Q = gnp.garray(Q)
        #P = gnp.garray(P)
        Q_new = rst_prob*norm_reset + (1-rst_prob) * np.dot(Q, P)#.as_numpy_array()
        delta = np.linalg.norm(Q-Q_new, 'fro')
        Q = Q_new
        print 'random walk iter',i
        sys.stdout.flush()
        if delta < 1e-4:
            break
    return Q
    
def evaluate_vec(pred,truth):
	pred = np.array(pred)
	truth = np.array(truth)
	pear = scipy.stats.pearsonr(pred, truth)[0]
	spear = scipy.stats.spearmanr(pred, truth)[0]

	if set(np.unique(truth))==set([0,1]):
		fpr, tpr, thresholds = metrics.roc_curve(truth, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		auprc = metrics.average_precision_score(truth, pred)
	else:
		auc = 0.5
		auprc = 0
	return auc,pear,spear,auprc



def evalute_path_emb(path2label, p2g, p2i, nselect_path=1000000,up=100000,low = -1,path2gene=[]):
    auc = {}
    auprc = {}
    best_rank = []
    npath,ngene = np.shape(p2g)
    for path in path2label:
        if p2i[path] >= nselect_path:
            continue
        label_l = path2label[path]
        if len(path2gene) >0 and ( len(path2gene[path])<low or len(path2gene[path])>up):
            continue
        score = np.zeros(ngene)
        label = np.zeros(ngene)
        for g in label_l:
            label[g] = 1
        if np.sum(label) == 0:
            continue
        for i in range(ngene):
            score[i] = p2g[p2i[path], i]
        score_rank = np.argsort(score*-1)
        for g in label_l:
            best_rank.append(np.where(score_rank==g)[0][0])
        auc[path], tmp, tmp, auprc[path] = evaluate_vec(score,label)
    best_rank = np.array(best_rank)
    return np.mean(auc.values()), np.std(auc.values()),auc,np.mean(auprc.values()),best_rank
