import numpy as np
import tensorflow as tf
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

me0=0.00
std0=0.10
np.random.seed(1)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return(tf.Variable(initial))

def bias_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05) 
	return(tf.Variable(initial))
	
def conv2d(x, W, s=[1,1,1,1], padding='SAME'):
	if (padding.upper() == 'VALID'):
		return (tf.nn.conv2d(x,W,strides=s,padding='VALID'))
	# SAME
	return (tf.nn.conv2d(x,W,strides=s,padding='SAME'))

def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.nn.tanh(x)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def my_conv2d(x,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,pad='SAME',activation_L='None'):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,in_ch,out_ch],mean = me,stddev = std))
    b=tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L is None:
        L = tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b
    else:  
        L = activation_L(tf.nn.conv2d(x,kernel,strides=[1,str1,str2,1],padding=pad)+b)
    return L

def my_conv2d_transpose(x,out_wid,out_hei,in_ch,out_ch,me,std,ker_sh1,ker_sh2,str1,str2,activation_L1='None'):
    kernel=tf.Variable(tf.random_normal(shape=[ker_sh1,ker_sh2,out_ch,in_ch],mean = me,stddev = std))
    output_shape=[tf.shape(x)[0],out_wid,out_hei,out_ch]
    #print(output_shape)
    b=tf.Variable(tf.zeros([out_ch]))
    tf.add_to_collection("p_var",kernel)
    tf.add_to_collection("p_var",b)
    if activation_L1 is None:
        L = tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b
    else:
        L =tf.add(activation_L1(tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME")+b),tf.nn.conv2d_transpose(x,kernel,output_shape,strides=[1,str1,str2,1],padding="SAME"))
    return L


def my_fc(x, in_cn, out_cn, name_scope,activation_L1='None'):
	with tf.variable_scope(name_scope):
		w = weight_variable([in_cn, out_cn])
		b = bias_variable([out_cn])
		tf.add_to_collection("p_var",w)
		tf.add_to_collection("p_var",b)
	if activation_L1 is None:
		h = tf.matmul(x, w) + b
	else:
		h = activation_L1(tf.matmul(x, w) + b)
	return h


def model(x,outputN):
  #x  None*36*36*1
	x1 = my_conv2d(x, 1,8,me0,std0,4,4,2,2,"SAME",tf.nn.relu)   #None*32*32*1-->None*16*16*8
	x2 = my_conv2d(x1, 8,16,me0,std0,3,3,2,2,"SAME",tf.nn.relu) #            -->None*8*8*16
	x3 = my_conv2d(x2, 16,32,me0,std0,3,3,2,2,'SAME',tf.nn.relu) #            -->None*4*4*32
	x4=tf.reshape(x3,[-1,4*4*32])
	x5 = my_fc(x4,4*4*32,128,'fc1',tf.nn.tanh) # None*200*200*1
	x6 = my_fc(x5,128,outputN,'fc2',None) # None*200*200*1
	return x6


def first(argv = None):

	f10_name = './dens_quar_expand_1.npy'
	f20_name = './stiffness_directional_expand_1.npy'

	batch_size =500 #256
	decay_steps = 4000
	decay_rate = 0.99
	starter_learning_rate = 1e-5

	n_epochs = 100 #
  
	#size of input
	nelx=32
	nely=32
	resolution= nelx*nely
	outputN=6
	dens = np.load(f10_name)#np.array(np.load(f1_name)).astype(np.float32)

	stiffness = np.load(f20_name)#np.array(np.load(f2_name)).astype(np.float32)

	data_size=np.size(dens,0)

	test_size=100 
	train_size=4500   
    
	stiffness_mean = np.mean(stiffness)
	#stiffness_min = np.min(stiffness)
	#stiffness_max = np.max(stiffness)
	
	mi=0.0e8
	ma=3.97e8# exact maximum 50e6
	dens_train0=dens[0:train_size,:]-0.5
	stiffness_train0=(stiffness[0:train_size]-mi)/(ma-mi)#(stiffness[0:train_size,:]-stiffness_min)/(stiffness_max-stiffness_min)
 
	dens_test0=dens[data_size-test_size:data_size,:]-0.5
	stiffness_test0=(stiffness[data_size-test_size:data_size]-mi)/(ma-mi)#(stiffness[train_size:train_size+test_size,:]-stiffness_min)/(stiffness_max-stiffness_min)


	del dens
	del stiffness

	xs = tf.placeholder(tf.float32, shape=[None, resolution],name='xs_node')
	xs_reshape = tf.reshape(xs, shape=[-1, nelx, nely, 1])
	ys = tf.placeholder(tf.float32, shape=[None,outputN])
	prediction = tf.reshape(model(xs_reshape,outputN),[-1,outputN])

	#Grd=tf.gradient(prediction[:,0],xs)	
    
	#metrics
	mse = tf.losses.mean_squared_error(ys,prediction)
	mae = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction)))
	#loss
	loss = mse

	#train rate    
	global_step = tf.Variable(0, trainable=False)
	add_global = global_step.assign_add(1)
	learning_rate = tf.train.exponential_decay(starter_learning_rate,
		global_step=global_step,
		decay_steps=decay_steps,
		decay_rate=decay_rate,
		staircase=False)
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	
	cgt_var=tf.global_variables()
	saver = tf.train.Saver([var for var in cgt_var if  var in tf.get_collection_ref("p_var")])
 
    
	start_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		if True:
			saver.restore(sess,'./savemodel/model')
		batch_test = 100 #256
		mse_train = np.zeros(n_epochs)
		mse_test = np.zeros(n_epochs)
		mae_train = np.zeros(n_epochs)
		mae_test = np.zeros(n_epochs)
		prediction_num = 100 # < batch_test
		iter_num = train_size // batch_size
		test_iter_num = test_size//batch_test
		print('test_iter=',test_iter_num)
		order = np.arange(train_size)
		print('Training...')
		for epoch in range(n_epochs):
			total_mse = 0
			total_mae = 0
			test_total_mse = 0
			test_total_mae = 0
			np.random.shuffle(order)
			dens_train = dens_train0[order,:]
			stiffness_train = stiffness_train0[order]
			for iter_train in range(iter_num):
				x_batch = dens_train[iter_train*batch_size:(iter_train+1)*batch_size,:] #
				y_batch = stiffness_train[iter_train*batch_size:(iter_train+1)*batch_size] #
				_, l_rate= sess.run([add_global, learning_rate,], feed_dict={xs: x_batch,ys:y_batch})
				_, batch_loss, batch_mae= sess.run([train_step, mse, mae], feed_dict={xs:x_batch,ys:y_batch})
				total_mse += batch_loss
				total_mae += batch_mae
			print('Epoch:',epoch,', Learning rate:',l_rate)

			mse_train[epoch] = total_mse/iter_num
			mae_train[epoch] = total_mae/iter_num
			print('MSE_train:', mse_train[epoch], end = ' ')
			print('MAE_train:', mae_train[epoch])

			for iter_test in range(test_iter_num):
				x_test = dens_test0[iter_test*batch_test:iter_test*batch_test+batch_test,:]
				y_test = stiffness_test0[iter_test*batch_test:iter_test*batch_test+batch_test] 
				test_mse,test_mae,test_prediction = sess.run([mse, mae, prediction], feed_dict={xs:x_test,ys:y_test})
				test_total_mse += test_mse
				test_total_mae += test_mae
        
			plt.figure()
			flag2=x_test[0,:].reshape(nelx,nely).T
			flag3=np.zeros((2*nely,2*nelx))
			flag3[0:nely,0:nelx]=flag2[:,nelx-1::-1]
			flag3[0:nely,nelx:2*nelx]=flag2
			flag3[nely:,:]=flag3[nely-1::-1,:]
			plt.imshow(1-flag3,interpolation='None')#,cmap='gray') 
			plt.savefig('Test_result/draw_dens000.png')
			plt.close()
			      
			plt.figure()
			flag2=x_test[25,:].reshape(nelx,nely).T
			flag3=np.zeros((2*nely,2*nelx))
			flag3[0:nely,0:nelx]=flag2[:,nelx-1::-1]
			flag3[0:nely,nelx:2*nelx]=flag2
			flag3[nely:,:]=flag3[nely-1::-1,:]
			plt.imshow(1-flag3,interpolation='None')#,cmap='gray') 
			plt.savefig('Test_result/draw_dens25.png')      
			plt.close() 
			    		
			plt.figure()
			flag2=x_test[45,:].reshape(nelx,nely).T
			flag3=np.zeros((2*nely,2*nelx))
			flag3[0:nely,0:nelx]=flag2[:,nelx-1::-1]
			flag3[0:nely,nelx:2*nelx]=flag2
			flag3[nely:,:]=flag3[nely-1::-1,:]
			plt.imshow(1-flag3,interpolation='None')#,cmap='gray') 
			plt.savefig('Test_result/draw_dens45.png')      
			plt.close()
		
			plt.figure()
			plt.scatter(y_test[0:prediction_num,:].reshape(prediction_num*outputN),test_prediction[0:prediction_num,:].reshape(prediction_num*outputN))
			plt.plot(y_test[0:prediction_num:].reshape(prediction_num*outputN),y_test[0:prediction_num,:].reshape(prediction_num*outputN),'r-')
			#plt.show()
			plt.savefig("Test_result/test_accuracy_all.png")
			plt.close()
      
			
			mse_test[epoch] = test_total_mse/test_iter_num
			mae_test[epoch] = test_total_mae/test_iter_num
			print('MSE_test:', mse_test[epoch], end = '   ')
			print('MAE_test:', mae_test[epoch])


   
			if (epoch+1)%10 == 0: #1000
				saver.save(sess, "./savemodel/model")
			current_time = time.localtime()
			print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))
		saver.save(sess, "./savemodel/model")
		print('Training is finished!')


	print('Mean stiffness is:',stiffness_mean)
	end_time = time.localtime()
	print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
	print('Time end: ', time.strftime('%Y-%m-%d %H:%M:%S', end_time))


if __name__ == '__main__':
	first()
