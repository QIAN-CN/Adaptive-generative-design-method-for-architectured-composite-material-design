import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_train0 =2*np.load('./dens_quar_expand.npy')[0:5000]-1
minor=0
flag=np.random.permutation(data_train0.shape[0])
data_train=data_train0[flag,:]
nelx=32
nely=32

def get_inputs(noise_dim, image_height, image_width, image_depth):

    inputs_real = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth], name='inputs_real')
    inputs_noise = tf.placeholder(tf.float32, [None, noise_dim], name='inputs_noise')
    return inputs_real, inputs_noise

def get_generator(noise_img, output_dim, is_train=True, alpha=0.1):
    with tf.variable_scope("generator", reuse=(not is_train)):
        # 100 x 1 to 4 x 4 x 512

        layer1 = tf.layers.dense(noise_img, 4*4*16)          #(N,16)-->(N,4*4*16)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 16])          #(N,4*4*16)-->(N,4,4,16)
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1 = tf.nn.dropout(layer1, keep_prob=0.9)                  
        
        # 4 x 4 x 512 to 7 x 7 x 256
        layer2 = tf.layers.conv2d_transpose(layer1, 16, 5, strides=2, padding='same')     #(N,4,4,16)--> (N,8,8,32)
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.9)
        
        # 7 x 7 256 to 14 x 14 x 128
        layer3 = tf.layers.conv2d_transpose(layer2, 16, 4, strides=2, padding='same')         #(N,8,8,32)--> (N,16,16,16)
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.9)

        # 7 x 7 256 to 14 x 14 x 128
        layer4 = tf.layers.conv2d_transpose(layer3, 8, 4, strides=2, padding='same')         #(N,16,16,16)--> (N,32,32,8)
        layer4 = tf.layers.batch_normalization(layer4, training=is_train)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.nn.dropout(layer4, keep_prob=0.9)
        
        logits0 = tf.layers.conv2d(layer4, output_dim, 3, strides=1, padding='same')


        temp=tf.reshape(logits0,[tf.shape(logits0)[0],nelx*nely])
        temp1=tf.contrib.framework.sort(temp,axis=1)
        temp2=(temp1[:,357]+temp1[:,358])/2
        logits=logits0-temp2[:,None,None,None]

        outputs = tf.tanh(logits)
        return outputs        
        
        
    
def get_discriminator(inputs_img, reuse=False, alpha=0.1):
    with tf.variable_scope("discriminator", reuse=reuse):
        # 28 x 28 x 1 to 14 x 14 x 128
        layer1 = tf.layers.conv2d(inputs_img, 4, 4, strides=2, padding='same') #(N,32,32,1)--> (N,16,16,4)
        layer1 = tf.layers.batch_normalization(layer1, training=True)
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.9)
        
        # 14 x 14 x 128 to 7 x 7 x 256
        layer2 = tf.layers.conv2d(layer1,8, 4, strides=2, padding='same')   #(N,16,16,4)--> (N,8,8,8)
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.9)
        
        # 7 x 7 x 256 to 4 x 4 x 512
        layer3 = tf.layers.conv2d(layer2, 16, 3, strides=2, padding='same')   #(N,8,8,8)--> (N,4,4,16)
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.9)

        # 7 x 7 x 256 to 4 x 4 x 512
        layer4 = tf.layers.conv2d(layer3, 32, 3, strides=2, padding='same')   #(N,4,4,16)--> (N,2,2,32)
        layer4 = tf.layers.batch_normalization(layer4, training=True)
        layer4 = tf.maximum(alpha * layer4, layer4)
        layer4 = tf.nn.dropout(layer4, keep_prob=0.9)
        
        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer4, (-1, 2*2*32))

        logits = tf.layers.dense(flatten, 1,activation=None)
      
        outputs = tf.sigmoid(logits)          
        
        return logits, outputs


def get_loss(inputs_real, inputs_noise, image_depth, smooth=0.0):
 
    g_outputs = get_generator(inputs_noise, image_depth, is_train=True)
    d_logits_real, d_outputs_real = get_discriminator(inputs_real)
    d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
    
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                    labels=tf.ones_like(d_outputs_fake)*(1-smooth)))    
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real)*(1-smooth)))                                                              
                                                                            
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))                                                                       
    d_loss = tf.add(d_loss_real, d_loss_fake)    
    return g_loss, d_loss



def get_optimizer(g_loss, d_loss, beta1=0.5, learning_rate=0.001):

    train_vars = tf.trainable_variables()
    
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
    
    # Optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    
    return g_opt, d_opt


def plot_images(samples,steps):
    nelx=32;nely=32;
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(9,2))
    for img, ax in zip(samples, axes):
    
        flag2=0.5+0.5*img.reshape((nelx, nely)).T
        flag3=np.zeros((2*nely,2*nelx))
        flag3[0:nely,0:nelx]=flag2[:,nelx-1::-1]
        flag3[0:nely,nelx:2*nelx]=flag2
        flag3[nely:,:]=flag3[nely-1::-1,:]
    
        ax.imshow(1-flag3)#, cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    name='images/GANgenerated%d.png' % steps
    plt.savefig(name)
    
def show_generator_output(sess, n_images, inputs_noise, output_dim):

    cmap = 'Greys_r'
    noise_shape = inputs_noise.get_shape().as_list()[-1]
    examples_noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])

    samples = sess.run(get_generator(inputs_noise, output_dim, False),
                       feed_dict={inputs_noise: examples_noise})

    
    result = np.squeeze(samples, -1)
    return result


batch_size = 500
noise_size = 16
epochs = 50
n_samples = 4
learning_rate =1e-4
beta1 = 0.5


def train(noise_size, data_shape, batch_size, n_samples):

    losses = []
    steps = 0
    
    inputs_real, inputs_noise = get_inputs(noise_size, data_shape[1], data_shape[2], data_shape[3])
    g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])
    g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        GANg_vars = tf.global_variables()
        g_vars = [var for var in GANg_vars if var.name.startswith("generator")]
        d_vars = [var for var in GANg_vars if var.name.startswith("discriminator")]
        saver = tf.train.Saver([var for var in list(set(g_vars).union(set(d_vars)))])
        ckpt_path = './ckpt/checkpoint_model.ckpt'
        if True:
            saver.restore(sess, ckpt_path + '-'+ str(1))
       
        for e in range(epochs):
            if np.mod(e+1,10)==0 and e>0:
               saver.save(sess, ckpt_path, global_step=1)                     
            for batch_i in range(data_train.shape[0]//batch_size):
                steps += 1
                batch = data_train[batch_i*batch_size:(batch_i+1)*batch_size,:]

                batch_images = batch.reshape((batch_size, data_shape[1], data_shape[2], data_shape[3]))
                # scale to -1, 1
                #batch_images = batch_images

                # noise
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

                # run optimizer
                _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
                
                if steps % 100 == 0: #and e>=epochs/2:
                    train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})     
                    train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                    losses.append((train_loss_d, train_loss_g))
                    samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                    plot_images(samples,steps)
 

                    print("Epoch {}/{}....".format(e+1, epochs), 
                          "Discriminator Loss: {:.4f}....".format(train_loss_d),
                          "Generator Loss: {:.4f}....". format(train_loss_g))

with tf.Graph().as_default():
    train(noise_size, [-1, 32, 32, 1], batch_size, n_samples)
