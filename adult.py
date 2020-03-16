import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import sem
import SenSR
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
import AdvDebCustom

def get_adult_data():
    '''
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']
    train = pd.read_csv('adult.data', header = None)
    test = pd.read_csv('adult.test', header = None)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']

    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_ Male', 'race_ White'])

def preprocess_adult_data(seed = 0):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)

    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_adult_data()

    # we will standardize continous features
    continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]

    # get a 80%/20% train/test split
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed = seed)
    SS = StandardScaler().fit(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_train.features[:, continous_features_indices] = SS.transform(dataset_orig_train.features[:, continous_features_indices])
    dataset_orig_test.features[:, continous_features_indices] = SS.transform(dataset_orig_test.features[:, continous_features_indices])

    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y_train.reshape(-1,1))
    names_income = one_hot.categories_
    y_train = one_hot.transform(y_train.reshape(-1,1))
    y_test = one_hot.transform(y_test.reshape(-1,1))

    # Also create a train/test set where the predictive features (X) do not include gender and gender is what you want to predict (y). This is used when learnng the sensitive direction for SenSR
    X_gender_train = np.delete(X_train, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)
    X_gender_test = np.delete(X_test, [dataset_orig_test.feature_names.index(feat) for feat in ['sex_ Male']], axis = 1)

    y_gender_train = dataset_orig_train.features[:, dataset_orig_train.feature_names.index('sex_ Male')]
    y_gender_test = dataset_orig_test.features[:, dataset_orig_test.feature_names.index('sex_ Male')]

    one_hot.fit(y_gender_train.reshape(-1,1))
    names_gender = one_hot.categories_
    y_gender_train = one_hot.transform(y_gender_train.reshape(-1,1))
    y_gender_test = one_hot.transform(y_gender_test.reshape(-1,1))

    return X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test

def save_to_file(directory, variable, name):
    timestamp = str(int(time.time()))
    with open(directory + name + '_' + timestamp + '.txt', "w") as f:
        f.write(str(np.mean(variable))+"\n")
        f.write(str(sem(variable))+"\n")
        for s in variable:
            f.write(str(s) +"\n")

def compute_gap_RMS_and_gap_max(data_set):
    '''
    Description: computes the gap RMS and max gap
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = -1*data_set.false_negative_rate_difference()
    TNR = -1*data_set.false_positive_rate_difference()

    return np.sqrt(1/2*(TPR**2 + TNR**2)), max(np.abs(TPR), np.abs(TNR))

def compute_balanced_accuracy(data_set):
    '''
    Description: computes the balanced accuracy, i.e. the average of TPR and TNR
    Input: data_set: a BinaryLabelDataset (from the aif360 module)
    '''
    TPR = data_set.true_positive_rate()
    TNR = data_set.true_negative_rate()
    return 0.5*(TPR+TNR)

def get_consistency(X, weights=0, proj = 0, gender_idx = 39, race_idx = 40, relationship_idx = [33, 34, 35, 36, 37, 38], husband_idx = 33, wife_idx = 38, adv = 0, dataset_orig_test = 0):
    '''
    Description: Ths function computes spouse consistency and gender and race consistency.
    Input:
        X: numpy matrix of predictive features
        weights: learned weights for project, baseline, and sensr
        proj: if using the project first baseline, this is the projection matrix
        gender_idx: column corresponding to the binary gender variable
        race_idx: column corresponding to the binary race variable
        relationship)_idx: list of column for the following features: relationship_ Husband, relationship_ Not-in-family, relationship_ Other-relative, relationship_ Own-child, relationship_ Unmarried, relationship_ Wife
        husband_idx: column corresponding to the husband variable
        wife_idx: column corresponding to the wife variable
        adv: the adversarial debiasing object if using adversarial Adversarial Debiasing
        dataset_orig_test: this is the data in a BinaryLabelDataset format when using adversarial debiasing
    '''
    gender_race_idx = [gender_idx, race_idx]

    if adv == 0:
        N, D = X.shape
        K = 1

        tf_X = tf.placeholder(tf.float32, shape=[None,D])
        tf_y = tf.placeholder(tf.float32, shape=[None,K], name='response')

        n_units = weights[1].shape
        n_units = n_units[0]

        _, l_pred, _, _ = SenSR.forward(tf_X, tf_y, weights=weights, n_units = n_units, activ_f = tf.nn.relu)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n, _ = X.shape

        # make 4 versions of the original data by changing binary gender and gender, then count how many classifications change
        #copy 1
        X00 = np.copy(X)
        X00[:, gender_race_idx] = 0

        if np.ndim(proj) != 0:
            X00 = X00@proj

        if adv == 0:
            X00_logits = l_pred.eval(feed_dict={tf_X: X00})
            X00_preds = np.argmax(X00_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X00
            dataset_mod, _ = adv.predict(dataset_mod)
            X00_preds = [i[0] for i in dataset_mod.labels]

        #### copy 2
        X01 = np.copy(X)
        X01[:, gender_race_idx] = 0
        X01[:, gender_idx] = 1

        if np.ndim(proj) != 0:
            X01 = X01@proj

        if adv == 0:
            X01_logits = l_pred.eval(feed_dict={tf_X: X01})
            X01_preds = np.argmax(X01_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X01
            dataset_mod, _ = adv.predict(dataset_mod)
            X01_preds = [i[0] for i in dataset_mod.labels]

        #### copy 3
        X10 = np.copy(X)
        X10[:, gender_race_idx] = 0
        X10[:, race_idx] = 1

        if np.ndim(proj) != 0:
            X10 = X10@proj

        if adv == 0:
            X10_logits = l_pred.eval(feed_dict={tf_X: X10})
            X10_preds = np.argmax(X10_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X10
            dataset_mod, _ = adv.predict(dataset_mod)
            X10_preds = [i[0] for i in dataset_mod.labels]

        #### copy 4
        X11 = np.copy(X)
        X11[:, race_idx] = 1
        X11[:, gender_idx] = 1

        if np.ndim(proj) != 0:
            X11 = X11@proj

        if adv == 0:
            X11_logits = l_pred.eval(feed_dict={tf_X: X11})
            X11_preds = np.argmax(X11_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X11
            dataset_mod, _ = adv.predict(dataset_mod)
            X11_preds = [i[0] for i in dataset_mod.labels]

        gender_and_race_consistency = np.mean([1 if X00_preds[i] == X01_preds[i] and X00_preds[i] == X10_preds[i] and X00_preds[i] == X11_preds[i] else 0 for i in range(len(X00_preds))])

        # make two copies of every datapoint: one which is a husband and one which is a wife. Then count how many classifications change
        X_husbands = np.copy(X)
        X_husbands[:,relationship_idx] = 0
        X_husbands[:,husband_idx] = 1

        if np.ndim(proj) != 0:
            X_husbands = X_husbands@proj

        if adv == 0:
            husband_logits = l_pred.eval(feed_dict={tf_X: X_husbands})
            husband_preds = np.argmax(husband_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X_husbands
            dataset_mod, _ = adv.predict(dataset_mod)
            husband_preds = [i[0] for i in dataset_mod.labels]

        X_wives = np.copy(X)
        X_wives[:,relationship_idx] = 0
        X_wives[:,wife_idx] = 1

        if np.ndim(proj) != 0:
            X_wives = X_wives@proj

        if adv == 0:
            wife_logits = l_pred.eval(feed_dict={tf_X: X_wives})
            wife_preds = np.argmax(wife_logits, axis = 1)
        else:
            dataset_mod = dataset_orig_test.copy(deepcopy=True)
            dataset_mod.features = X_wives
            dataset_mod, _ = adv.predict(dataset_mod)
            wife_preds = [i[0] for i in dataset_mod.labels]

        spouse_consistency = np.mean([1 if husband_preds[i] == wife_preds[i] else 0 for i in range(len(husband_preds))])

        return gender_and_race_consistency, spouse_consistency

def get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test, gender_race_features_idx = [39, 40], gender_idx = 39 ):
    '''
    Description: Get the sensitive directions and projection matrix. he sensitive directions include the race and gender direction as well as the learned hyperplane that predicts gender (without using gender as a predictive feature of course).
    '''
    weights, train_logits, test_logits = SenSR.train_nn(X_gender_train, y_gender_train, X_test = X_gender_test, y_test = y_gender_test, n_units=[], l2_reg=.1, batch_size=5000, epoch=5000, verbose=False)

    n, d = weights[0].shape
    sensitive_directions = []

    # transform the n-dimensional weights back into the full n+1 dimensional space where the gender coordinate is zeroed out
    full_weights = np.zeros((n+1,d))

    #before the gender coordinate, the coordinates of the full_weights and learned weights correspond to the same features
    for i in range(gender_idx):
        full_weights[i,:] = weights[0][i,:]

    #after the gender coordinate, the i-th coordinate of the full_weights correspond to the (i-1)-st coordinate of the learned weights
    for i in range(gender_idx+1, n+1):
        full_weights[i, :] = weights[0][i-1,:]

    sensitive_directions.append(full_weights.T)

    for idx in gender_race_features_idx:
        temp_direction = np.zeros((n+1,1)).reshape(1,-1)
        temp_direction[0, idx] = 1
        sensitive_directions.append(np.copy(temp_direction))

    sensitive_directions = np.vstack(sensitive_directions)
    tSVD = TruncatedSVD(n_components= 2 + len(gender_race_features_idx))
    tSVD.fit(sensitive_directions)
    sensitive_directions = tSVD.components_

    return sensitive_directions, SenSR.compl_svd_projector(sensitive_directions)

def get_metrics(dataset_orig, preds):
    '''
    Description: This code computes accuracy, balanced accuracy, max gap and gap rms for race and gender
    Input: dataset_orig: a BinaryLabelDataset (from the aif360 module)
            preds: predictions
    '''
    dataset_learned_model = dataset_orig.copy()
    dataset_learned_model.labels = preds

    # wrt gender
    privileged_groups = [{'sex_ Male': 1}]
    unprivileged_groups = [{'sex_ Male': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc = compute_balanced_accuracy(classified_metric)

    gender_gap_rms, gender_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
    print("Test set: gender gap rms = %f" % gender_gap_rms)
    print("Test set: gender max gap rms = %f" % gender_max_gap)
    print("Test set: Balanced TPR = %f" % bal_acc)

    # wrt race
    privileged_groups = [{'race_ White': 1}]
    unprivileged_groups = [{'race_ White': 0}]

    classified_metric = ClassificationMetric(dataset_orig,
                                                     dataset_learned_model,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    race_gap_rms, race_max_gap = compute_gap_RMS_and_gap_max(classified_metric)
    print("Test set: race gap rms = %f" % race_gap_rms)
    print("Test set: race max gap rms = %f" % race_max_gap)


    return classified_metric.accuracy(), bal_acc, race_gap_rms, race_max_gap, gender_gap_rms, gender_max_gap

def run_baseline_experiment(X_train, y_train, X_test, y_test):
    return SenSR.train_nn(X_train, y_train, X_test = X_test, y_test = y_test, n_units=[100], l2_reg=0., lr = .00001, batch_size=1000, epoch=60000, verbose=False)

def run_SenSR_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i):
    sensitive_directions, _ = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)
    return SenSR.train_fair_nn(
        X_train,
        y_train,
        sensitive_directions,
        X_test=X_test,
        y_test=y_test,
        n_units = [100],
        lr=.0001,
        batch_size=1000,
        epoch=12000,
        verbose=False,
        l2_reg=0.,
        lamb_init=2.,
        subspace_epoch=50,
        subspace_step=10,
        eps=.001,
        full_step=.0001,
        full_epoch=40,
        fair_start = False)

def run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i):
    _, proj_compl = get_sensitive_directions_and_projection_matrix(X_gender_train, y_gender_train, X_gender_test, y_gender_test)

    np.save(directory+'proj_compl_'+str(i), proj_compl)

    X_train_proj = X_train@proj_compl
    X_test_proj = X_test@proj_compl
    weights, train_logits, test_logits = SenSR.train_nn(
        X_train_proj,
        y_train,
        X_test = X_test_proj,
        y_test = y_test,
        n_units=[100],
        l2_reg=0.,
        lr = .00001,
        batch_size=1000,
        epoch=60000,
        verbose=False)
    return weights, train_logits, test_logits, proj_compl

def run_adv_deb_experiment(dataset_orig_train, dataset_orig_test):
    sess = tf.Session()
    tf.name_scope("my_scope")
    privileged_groups = [{'sex_ Male': 1, 'race_ White':1}]
    unprivileged_groups = [{'sex_ Male': 0, 'race_ White':0}]
    adv = AdvDebCustom.AdversarialDebiasing(unprivileged_groups, privileged_groups, "my_scope", sess, seed=None, adversary_loss_weight=0.001, num_epochs=500, batch_size=1000, classifier_num_hidden_units=100, debias=True)

    _ = adv.fit(dataset_orig_train)
    test_data, _ = adv.predict(dataset_orig_test)

    return adv, test_data.labels

def run_experiments(name, num_exp, directory):
    '''
    Description: Run each experiment num_exp times where a new train/test split is generated. Save results in the path specified by directory

    Inputs: name: name of the experiment. Valid choices are baseline, project, SenSR, adv_deb
    '''

    if name not in ['baseline', 'project', 'SenSR', 'adv_deb']:
        raise ValueError('You did not specify a valid experiment to run.')

    gender_race_consistencies = []
    spouse_consistencies = []

    accuracy = []
    balanced_accuracy = []

    gender_gap_rms = []
    gender_max_gap = []

    race_gap_rms = []
    race_max_gap = []

    for i in range(num_exp):
        print('On experiment', i)

        # get train/test data
        X_train, X_test, y_train, y_test, X_gender_train, X_gender_test, y_gender_train, y_gender_test, dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = i)

        tf.reset_default_graph()

        # run experiments
        if name == 'baseline':
            weights, train_logits, test_logits  = run_baseline_experiment(X_train, y_train, X_test, y_test)
        elif name == 'SenSR':
            weights, train_logits, test_logits = run_SenSR_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i)
        elif name == 'project':
            weights, train_logits, test_logits, proj_compl = run_project_experiment(X_train, X_test, X_gender_train, X_gender_test, y_train, y_test, y_gender_train, y_gender_test, directory, i)
        elif name == 'adv_deb':
            adv, preds = run_adv_deb_experiment(dataset_orig_train, dataset_orig_test)

        # get race/gender and spouse consistency

        if name == 'project':
            gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights, proj = proj_compl)
        elif name == 'adv_deb':
            gender_race_consistency, spouse_consistency = get_consistency(X_test, adv = adv, dataset_orig_test = dataset_orig_test)
        else:
            gender_race_consistency, spouse_consistency = get_consistency(X_test, weights = weights)
        gender_race_consistencies.append(gender_race_consistency)
        print('gender/race combined consistency', gender_race_consistency)

        spouse_consistencies.append(spouse_consistency)
        print('spouse consistency', spouse_consistency)

        # get accuracy, balanced accuracy, gender/race gap rms, gender/race max gap

        if name != 'adv_deb':
            np.save(directory+'weights_'+str(i), weights)
            preds = np.argmax(test_logits, axis = 1)
        acc_temp, bal_acc_temp, race_gap_rms_temp, race_max_gap_temp, gender_gap_rms_temp, gender_max_gap_temp = get_metrics(dataset_orig_test, preds)

        gender_gap_rms.append(gender_gap_rms_temp)
        gender_max_gap.append(gender_max_gap_temp)
        accuracy.append(acc_temp)
        balanced_accuracy.append(bal_acc_temp)
        race_gap_rms.append(race_gap_rms_temp)
        race_max_gap.append(race_max_gap_temp)

    # save results to file
    save_info = [
        (gender_race_consistencies, 'gender_race_consistencies'),
        (accuracy, 'accuracy'),
        (balanced_accuracy, 'balanced_accuracy'),
        (gender_gap_rms, 'gender_gap_rms'),
        (gender_max_gap, 'gender_max_gap'),
        (race_gap_rms, 'race_gap_rms'),
        (race_max_gap, 'race_max_gap'),
        (spouse_consistencies, 'spouse_consistencies')]

    for (variable, name) in save_info:
        save_to_file(directory, variable, name)

num_exp = 10

# change path_to_save_experiments to the directoy where you want to save the results of the experiments
# and create a folder called experiments with 4 subfolders named baseline, project, sensr, adv_deb
path_to_save_experiments = './experiments/'

run_experiments('baseline', num_exp, path_to_save_experiments +'baseline/')
run_experiments('project', num_exp, path_to_save_experiments + 'project/')
run_experiments('SenSR', num_exp, path_to_save_experiments + 'sensr/')
run_experiments('adv_deb', num_exp, path_to_save_experiments + 'adv_deb/')
