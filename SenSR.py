import numpy as np
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf
from collections import OrderedDict

def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl

def fair_dist(proj, w=0.):
    tf_proj = tf.constant(proj, dtype=tf.float32)
    if w>0:
        return lambda x, y: tf.reduce_sum(tf.square(tf.matmul(x-y,tf_proj)) + w*tf.square(tf.matmul(x-y,tf.eye(proj.shape[0]) - tf_proj)), axis=1)
    else:
        return lambda x, y: tf.reduce_sum(tf.square(tf.matmul(x-y,tf_proj)), axis=1)

def weight_variable(shape, name):
    if len(shape)>1:
        init_range = np.sqrt(6.0/(shape[-1]+shape[-2]))
    else:
        init_range = np.sqrt(6.0/(shape[0]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32) # seed=1000
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def sample_batch_idx(y, n_per_class):
    batch_idx = []
    for i in range(y.shape[1]):
        batch_idx += np.random.choice(np.where(y[:,i]==1)[0], size=n_per_class, replace=False).tolist()

    np.random.shuffle(batch_idx)
    return batch_idx

def fc_network(variables, layer_in, n_layers, l=0, activ_f = tf.nn.relu, units = []):
    if l==n_layers-1:
        layer_out = tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)]
        units.append(layer_out)
        return layer_out, units
    else:
        layer_out = activ_f(tf.matmul(layer_in, variables['weight_'+str(l)]) + variables['bias_' + str(l)])
        l += 1
        units.append(layer_out)
        return fc_network(variables, layer_out, n_layers, l=l, activ_f=activ_f, units=units)

def forward(tf_X, tf_y, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=0.):

    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]

    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred))

    correct_prediction = tf.equal(tf.argmax(l_pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if l2_reg > 0:
        loss = cross_entropy + l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
    else:
        loss = cross_entropy

    return variables, l_pred, loss, accuracy

def train_nn(X_train, y_train, X_test=None, y_test=None, weights=None, n_units = [], lr=0.001, batch_size=1000, epoch=2000, verbose=False, activ_f = tf.nn.relu, l2_reg=0.):
    N, D = X_train.shape

    try:
        K = y_train.shape[1]
    except:
        K = len(weights[-1])

    tf_X = tf.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

    variables, l_pred, loss, accuracy = forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)

    if epoch > 0:
        train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        n_per_class = int(batch_size/K)
        n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
        batch_size = int(K*n_per_class)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for it in range(epoch):
            batch_idx = sample_batch_idx(y_train, n_per_class)

            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            train_step.run(feed_dict={
                  tf_X: batch_x, tf_y: batch_y})

            if it % 10 == 0 and verbose:
                print('\nEpoch %d train accuracy %f' % (it, accuracy.eval(feed_dict={
                      tf_X: X_train, tf_y: y_train})))
                if y_test is not None:
                    print('Epoch %d test accuracy %g' % (it, accuracy.eval(feed_dict={
                          tf_X: X_test, tf_y: y_test})))
        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))

        weights = [x.eval() for x in variables.values()]
        train_logits = l_pred.eval(feed_dict={tf_X: X_train})
        if X_test is not None:
            test_logits = l_pred.eval(feed_dict={tf_X: X_test})
        else:
            test_logits = None

    return weights, train_logits, test_logits

def forward_fair(tf_X, tf_y, tf_fair_X, weights=None, n_units = None, activ_f = tf.nn.relu, l2_reg=0.):

    if weights is not None:
        n_layers = int(len(weights)/2)
        n_units = [weights[i].shape[0] for i in range(0,len(weights),2)]
    else:
        n_features = int(tf_X.shape[1])
        n_class = int(tf_y.shape[1])
        n_layers = len(n_units) + 1
        n_units = [n_features] + n_units + [n_class]

    variables = OrderedDict()
    if weights is None:
        for l in range(n_layers):
            variables['weight_' + str(l)] = weight_variable([n_units[l],n_units[l+1]], name='weight_' + str(l))
            variables['bias_' + str(l)] = bias_variable([n_units[l+1]], name='bias_' + str(l))
    else:
        weight_ind = 0
        for l in range(n_layers):
            variables['weight_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1
            variables['bias_' + str(l)] = tf.constant(weights[weight_ind], dtype=tf.float32)
            weight_ind += 1


    ## Defining NN architecture
    l_pred, units = fc_network(variables, tf_X, n_layers, activ_f = activ_f)
    l_pred_fair, units_fair = fc_network(variables, tf_fair_X, n_layers, activ_f = activ_f)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred))
    cross_entropy_fair = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_y, logits=l_pred_fair))

    correct_prediction = tf.equal(tf.argmax(l_pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if l2_reg > 0:
        cross_entropy += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])
        cross_entropy_fair += l2_reg*sum([tf.nn.l2_loss(variables['weight_' + str(l)]) for l in range(n_layers)])

    return variables, l_pred, cross_entropy, accuracy, cross_entropy_fair

def train_fair_nn(X_train, y_train, sensitive_directions, X_test=None, y_test=None, weights=None, n_units = [], lr=0.001, batch_size=1000, epoch=2000, verbose=False, activ_f = tf.nn.relu, l2_reg=0., lamb_init=2., subspace_epoch=10, subspace_step=0.1, eps=None, full_step=-1, full_epoch=10, fair_start = True):

    if fair_start:
        fair_start = epoch/2
    else:
        fair_start = 0

    ## Fair distance
    proj_compl = compl_svd_projector(sensitive_directions, svd=-1)
    dist_f = fair_dist(proj_compl, 0.)
    V_sensitive = sensitive_directions.shape[0]

    global_step = tf.contrib.framework.get_or_create_global_step()

    N, D = X_train.shape
    lamb = lamb_init

    try:
        K = y_train.shape[1]
    except:
        K = len(weights[-1])

    n_per_class = int(batch_size/K)
    n_per_class = int(min(n_per_class, min(y_train.sum(axis=0))))
    batch_size = int(K*n_per_class)

    tf_X = tf.placeholder(tf.float32, shape=[None,D])
    tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

    ## Fair variables
    tf_directions = tf.constant(sensitive_directions, dtype=tf.float32)
    adv_weights = tf.Variable(tf.zeros([batch_size,V_sensitive]))
    full_adv_weights = tf.Variable(tf.zeros([batch_size,D]))
    tf_fair_X = tf_X + tf.matmul(adv_weights, tf_directions) + full_adv_weights

    variables, l_pred, _, accuracy, loss = forward_fair(tf_X, tf_y, tf_fair_X, weights=weights, n_units = n_units, activ_f = activ_f, l2_reg=l2_reg)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(loss, var_list=list(variables.values()), global_step=global_step)
    reset_optimizer = tf.variables_initializer(optimizer.variables())
    reset_main_step = True

    ## Attack is subspace
    fair_optimizer = tf.train.AdamOptimizer(learning_rate=subspace_step)
    fair_step = fair_optimizer.minimize(-loss, var_list=[adv_weights], global_step=global_step)
    reset_fair_optimizer = tf.variables_initializer(fair_optimizer.variables())
    reset_adv_weights = adv_weights.assign(tf.zeros([batch_size,V_sensitive]))

    ## Attack out of subspace
    distance = dist_f(tf_X, tf_fair_X)
    tf_lamb = tf.placeholder(tf.float32, shape=())
    dist_loss = tf.reduce_mean(distance)
    fair_loss = loss - tf_lamb*dist_loss

    if full_step > 0:
        full_fair_optimizer = tf.train.AdamOptimizer(learning_rate=full_step)
        full_fair_step = full_fair_optimizer.minimize(-fair_loss, var_list=[full_adv_weights], global_step=global_step)
        reset_full_fair_optimizer = tf.variables_initializer(full_fair_optimizer.variables())
        reset_full_adv_weights = full_adv_weights.assign(tf.zeros([batch_size,D]))

    ######################

    failed_attack_count = 0
    failed_full_attack = 0
    failed_subspace_attack = 0

    out_freq = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for it in range(epoch):

            batch_idx = sample_batch_idx(y_train, n_per_class)
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]

            if it > fair_start:
                if reset_main_step:
                    sess.run(reset_optimizer)
                    reset_main_step = False

                loss_before_subspace_attack = loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})

                ## Do subspace attack
                for adv_it in range(subspace_epoch):
                    fair_step.run(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})
                ## Check result
                loss_after_subspace_attack = loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y})
                if loss_after_subspace_attack < loss_before_subspace_attack:
                        print('WARNING: subspace attack failed: objective decreased from %f to %f; resetting the attack' % (loss_before_subspace_attack, loss_after_subspace_attack))
                        sess.run(reset_adv_weights)
                        failed_subspace_attack += 1

                if full_step > 0:
                    fair_loss_before_l2_attack = fair_loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})

                    ## Do full attack
                    for full_adv_it in range(full_epoch):
                        full_fair_step.run(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})

                    ## Check result
                    fair_loss_after_l2_attack = fair_loss.eval(feed_dict={
                            tf_X: batch_x, tf_y: batch_y, tf_lamb: lamb})
                    if fair_loss_after_l2_attack < fair_loss_before_l2_attack:
                        print('WARNING: full attack failed: objective decreased from %f to %f; resetting the attack' % (fair_loss_before_l2_attack, fair_loss_after_l2_attack))
                        sess.run(reset_full_adv_weights)
                        failed_full_attack += 1

                adv_batch = tf_fair_X.eval(feed_dict={tf_X: batch_x})

                if np.isnan(adv_batch.sum()):
                    print('Nans in adv_batch; making no change')
                    sess.run(reset_adv_weights)
                    if full_step > 0:
                        sess.run(reset_full_adv_weights)
                    failed_attack_count += 1

                elif eps is not None:
                    mean_dist = dist_loss.eval(feed_dict={tf_X: batch_x})
                    lamb = max(0.00001,lamb + (max(mean_dist,eps)/min(mean_dist,eps))*(mean_dist - eps))
            else:
                adv_batch = batch_x

            _, loss_at_update = sess.run([train_step,loss], feed_dict={
                  tf_X: batch_x, tf_y: batch_y})

            if it > fair_start:
                sess.run(reset_adv_weights)
                sess.run(reset_fair_optimizer)
                if full_step > 0:
                    sess.run(reset_full_fair_optimizer)
                    sess.run(reset_full_adv_weights)

            if it % out_freq == 0 and verbose:
                train_acc, train_logits = sess.run([accuracy,l_pred], feed_dict={
                      tf_X: X_train, tf_y: y_train})
                print('Epoch %d train accuracy %f; lambda is %f' % (it, train_acc, lamb))
                if y_test is not None:
                    test_acc, test_logits = sess.run([accuracy,l_pred], feed_dict={
                            tf_X: X_test, tf_y: y_test})
                    print('Epoch %d test accuracy %g' % (it, test_acc))

                ## Attack summary
                if it > fair_start:
                    print('FAILED attacks: subspace %d; full %d; Nans after attack %d' % (failed_subspace_attack, failed_full_attack, failed_attack_count))
                    print('Loss clean %f; subspace %f; full %f' % (loss_before_subspace_attack, loss_after_subspace_attack, loss_at_update))

        if y_train is not None:
            print('\nFinal train accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_train, tf_y: y_train})))
        if y_test is not None:
            print('Final test accuracy %g' % (accuracy.eval(feed_dict={
                  tf_X: X_test, tf_y: y_test})))
        if eps is not None:
            print('Final lambda %f' % lamb)

        fair_weights = [x.eval() for x in variables.values()]
        train_logits = l_pred.eval(feed_dict={tf_X: X_train})
        if X_test is not None:
            test_logits = l_pred.eval(feed_dict={tf_X: X_test})
        else:
            test_logits = None

    return fair_weights, train_logits, test_logits
