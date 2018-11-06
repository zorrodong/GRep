import numpy as np
import tensorflow as tf
from util import *
import sys
from scipy import stats

class GRep:
	def __init__(self, log_Path_RWR, log_node_RWR, path2gene, auc_d, p2i, path2label, L, alpha, optimize_gene_vec=True, lr = 1e-2, n_hidden=None,
				 max_iter=2000, tolerance=1000, seed=0):
		"""
		Parameters
		----------
		log_Path_RWR: logrithm of gene set diffusion matrix
		log_node_RWR : logirthm of gene diffusion matrix
		path2gene : gene membership of a gene set. Only used in evaluation
		auc_d : auc of comparison approaches. Only used in evaluation
        L: dimension
        optimize_gene_vec: optimize gene vectors or not
		"""

		tf.reset_default_graph()
		tf.set_random_seed(seed)
		np.random.seed(seed)
		self.auc_d = auc_d
		self.alpha = alpha
		self.lr = lr
		self.path2gene = path2gene
		self.optimize_gene_vec = optimize_gene_vec
		self.nselect_path = np.shape(log_Path_RWR)[0]
		self.p2i = p2i
		self.path2label = path2label
		self.log_Path_RWR = log_Path_RWR.astype(np.float32)
		self.log_node_RWR = log_node_RWR.astype(np.float32)
		self.feed_dict = None
		self.npath, self.nnode = self.log_Path_RWR.shape
		self.L = L
		self.max_iter = max_iter

		if n_hidden is None:
			n_hidden = [512]
		self.n_hidden = n_hidden

		self.__build()
		print 'build finished'
		sys.stdout.flush()
		self.__build_loss()
		print 'build loss finished'
		sys.stdout.flush()

	def __build(self):
		w_init = tf.contrib.layers.xavier_initializer
		self.node_mu = tf.get_variable(name='node_mu', dtype=tf.float32, shape=[self.nnode, self.L],initializer= w_init())
		self.node_context = tf.get_variable(name='node_context', shape=[self.nnode, self.L], dtype=tf.float32, initializer = w_init())
		self.path_cov_w = tf.get_variable(name='Path_sigma_w', dtype=tf.float32, shape=[self.npath, self.L,  self.n_hidden[0]],initializer= w_init())
		self.path_cov_x = tf.get_variable(name='Path_sigma_x', dtype=tf.float32, shape=[self.npath, self.L,  self.n_hidden[0]],initializer= w_init())
		self.path_mu = tf.get_variable(name='Path_mu',dtype=tf.float32, shape=[self.npath, self.L], initializer= w_init())


	def fast_mf(self):
		pmat_fast = []
		for p in range(self.npath):
			pmat_fast.append(self.fast_full_cov_mat(p))
		x_cov = tf.stack(pmat_fast)
		return x_cov

	def fast_full_cov_mat(self, p):
		path_bar = tf.gather(self.path_mu, p)
		path_bar = tf.expand_dims(path_bar,0) # change mu to 1 * ndim
		multiply = tf.constant([self.nnode, 1])
		node_bar = tf.tile(path_bar, multiply) # path mu is npath*ndim

		self.x_diff = tf.square(tf.subtract(self.node_mu, node_bar))

		inv_scale_w = tf.gather(self.path_cov_w,p)
		inv_scale_x = tf.gather(self.path_cov_x,p)
		inv_scale = tf.matmul(inv_scale_w, tf.transpose(inv_scale_x))
		cov =tf.matmul(inv_scale, tf.transpose(inv_scale))
		inv_cov = cov
		x_cov = tf.matmul(self.x_diff, inv_cov)
		x_cov_x_sum = tf.reduce_sum(x_cov, axis =1) * -0.5
		return x_cov_x_sum

	def node_vec_estimation(self):
		gmat = tf.matmul(self.node_mu, tf.transpose(self.node_context))
		return gmat


	def __build_loss(self):

		self.Path_our = self.fast_mf()
		if self.optimize_gene_vec:
			self.gmat = self.node_vec_estimation()
			self.loss = tf.losses.mean_squared_error(self.Path_our, tf.convert_to_tensor(self.log_Path_RWR)) +tf.losses.mean_squared_error(self.gmat, tf.convert_to_tensor(self.log_node_RWR))

		else:
			self.loss = tf.losses.mean_squared_error(self.Path_our, tf.convert_to_tensor(self.log_Path_RWR))
			print 'here'




	def train(self, gpu_list='0'):

		train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=gpu_list,
																		  allow_growth=True)))
		sess.run(tf.global_variables_initializer())


		for epoch in range(self.max_iter):

			Path_our = self.fast_mf()
			Path_our = sess.run(Path_our)

			auc, auc_std, auc_d, auprc,best_rank = evalute_path_emb(self.path2label, Path_our, self.p2i, self.nselect_path)

			print epoch,'auc',auc
			loss, _ = sess.run([self.loss, train_op], self.feed_dict)

			print('epoch: {:3d}, loss: {:.10f}'.format(epoch, loss))

		return Path_our
