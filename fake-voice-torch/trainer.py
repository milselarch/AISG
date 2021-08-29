import os

class DiscriminatorTrainer(object):
    def __init__(
        self, load_pretrained=False, saved_model_name=None,
        real_test_mode=False
    ):
        if not os.path.exists(model_params['model_save_dir']):
            os.makedirs(model_params['model_save_dir'])

        if not load_pretrained:
            self.model = self.build_custom_convnet(model_params)
            self.model.summary()
        else:
            self.model = load_model(
                os.path.join(
                    f"./{model_params['model_save_dir']}",
                    saved_model_name
                ), custom_objects={'customPooling': customPooling}
            )

        self.model_name = f"saved_model_{'_'.join(str(v) for k,v in model_params.items())}.h5"​【13 cm】
        self.real_test_model_name = f"real_test_saved_model_{'_'.join(str(v) for k,v in model_params.items())}.h5"​【13 cm】
        self.model_save_filename = os.path.join(f"./{model_params['model_save_dir']}", self.model_name)
        self.real_test_model_save_filename = os.path.join(f"./{model_params['model_save_dir']}", self.real_test_model_name)
        if real_test_mode:
            if run_on_foundations:
                self.real_test_data_dir = "/data/inference_data/"
            else:
                self.real_test_data_dir = "../data/inference_data/"

            # preprocess the files
            self.real_test_processed_data_real = preprocess_from_ray_parallel_inference(self.real_test_data_dir, "real", use_parallel=True)
            self.real_test_processed_data_fake = preprocess_from_ray_parallel_inference(self.real_test_data_dir,
                                                                                        "fake",
                                                                                        use_parallel=True)

            self.real_test_processed_data = self.real_test_processed_data_real + self.real_test_processed_data_fake
            self.real_test_processed_data = sorted(self.real_test_processed_data, key=lambda x: len(x[0]))
            self.real_test_features = [x[0] for x in self.real_test_processed_data]
            self.real_test_labels = [x[1] for x in self.real_test_processed_data]
            print(f"Length of real_test_processed_data: {len(self.real_test_processed_data)}")

    def build_custom_convnet(self, model_params):


    def forward(self):


    def train(self, xtrain, ytrain, xval, yval):
        callbacks = []
        tb = TensorBoard(log_dir='tflogs', write_graph=True, write_grads=False)
        callbacks.append(tb)

        try:
            foundations.set_tensorboard_logdir('tflogs')
        except:
            print("foundations command not found")

        es = EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0001,
                           verbose=1)
        callbacks.append(tb)
        callbacks.append(es)

        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2,
                               verbose=1)
        callbacks.append(rp)

        f1_callback = f1_score_callback(xval, yval, model_save_filename=self.model_save_filename)
        callbacks.append(f1_callback)

        class_weights = {1: 5, 0: 1}

        train_generator = DataGenerator(xtrain, ytrain)
        validation_generator = DataGenerator(xval, yval)
        self.model.fit_generator(train_generator,
                                 steps_per_epoch = len(train_generator),
                                 epochs = model_params['epochs'],
                                 validation_data=validation_generator,
                                 callbacks = callbacks,
                                 shuffle = False,
                                  use_multiprocessing = True,
                                  verbose = 1,
                                 class_weight =class_weights)

        self.model = load_model(self.model_save_filename, custom_objects={'customPooling': customPooling})

        try:
            foundations.save_artifact(self.model_save_filename, key='trained_model.h5'​【1,5 m】)
        except:
            print("foundations command not found")


    def inference_on_real_data(self, threshold=0.5):
        datagen_val = DataGenerator(self.real_test_features, mode='test', batch_size=1)
        y_pred = self.model.predict_generator(datagen_val, use_multiprocessing=False, max_queue_size=50)
        y_pred_labels = np.zeros((len(y_pred)))
        y_pred_labels[y_pred.flatten() > threshold] = 1
        acc_score = accuracy_score(self.real_test_labels, y_pred_labels)
        f1_score_val = f1_score(self.real_test_labels, y_pred_labels)
        return acc_score, f1_score_val


    def get_labels_from_prob(self, y, threshold=0.5):
        y_pred_labels = np.zeros((len(y)))
        y = np.array(y)
        if isinstance(threshold, list):
            y_pred_labels[y.flatten() > threshold[0]] = 1
        else:
            y_pred_labels[y.flatten() > threshold] = 1
        return y_pred_labels

    def get_f1score_for_optimization(self, threshold, y_true, y_pred, ismin=False):
        y_pred_labels = self.get_labels_from_prob(y_pred, threshold=threshold)
        if ismin:
            return - f1_score(y_true, y_pred_labels)
        else:
            return f1_score(y_true, y_pred_labels)

    def predict_labels(self, x, threshold=0.5, raw_prob=False, batch_size=model_params['batch_size']):
        test_generator = DataGenerator(x, mode='test', batch_size=batch_size)
        y_pred = self.model.predict_generator(test_generator, steps=len(test_generator), max_queue_size=10)
        print(y_pred)
        if raw_prob:
            return y_pred
        else:
            y_pred_labels = self.get_labels_from_prob(y_pred, threshold=threshold)
            return y_pred_labels

    def optimize_threshold(self, xtrain, ytrain, xval, yval):
        ytrain_pred = self.predict_labels(xtrain, raw_prob=True)
        yval_pred = self.predict_labels(xval, raw_prob=True)
        self.opt_threshold = 0.5
        ytrain_pred_labels = self.get_labels_from_prob(ytrain_pred, threshold=self.opt_threshold)
        yval_pred_labels = self.get_labels_from_prob(yval_pred, threshold=self.opt_threshold)
        train_f1_score = f1_score(ytrain_pred_labels, ytrain)
        val_f1_score = f1_score(yval_pred_labels, yval)
        print(f"train f1 score: {train_f1_score}, val f1 score: {val_f1_score}")

        f1_train_partial = partial(self.get_f1score_for_optimization, y_true=ytrain.copy(), y_pred=ytrain_pred.copy(), ismin=True)
        n_searches = 50
        dim_0 = Real(low=0.2, high=0.8, name='dim_0')
        dimensions = [dim_0]
        search_result = gp_minimize(func=f1_train_partial,
                                    dimensions=dimensions,
                                    acq_func='gp_hedge',  # Expected Improvement.
                                    n_calls=n_searches,
                                    # n_jobs=n_cpu,
                                    verbose=False)

        self.opt_threshold = search_result.x
        if isinstance(self.opt_threshold,list):
            self.opt_threshold = self.opt_threshold[0]
        self.optimum_threshold_filename = f"model_threshold_{'_'.join(str(v) for k, v in model_params.items())}.npy"
        np.save(os.path.join(f"{model_params['model_save_dir']}",self.optimum_threshold_filename), self.opt_threshold)
        train_f1_score = self.get_f1score_for_optimization(self.opt_threshold, y_true=ytrain, y_pred=ytrain_pred)
        val_f1_score = self.get_f1score_for_optimization(self.opt_threshold, y_true=yval, y_pred=yval_pred )
        print(f"optimized train f1 score: {train_f1_score}, optimized val f1 score: {val_f1_score}")



    def evaluate(self, xtrain, ytrain, xval, yval, num_examples=1):
        ytrain_pred = self.predict_labels(xtrain, raw_prob=True)
        yval_pred = self.predict_labels(xval, raw_prob=True)
        try:
            self.optimum_threshold_filename = f"model_threshold_{'_'.join(str(v) for k, v in model_params.items())}.npy"
            self.opt_threshold = np.load(os.path.join(f"{model_params['model_save_dir']}",self.optimum_threshold_filename)).item()
            print(f"loaded optimum threshold: {self.opt_threshold}")
        except:
            self.opt_threshold = 0.5


        ytrain_pred_labels = self.get_labels_from_prob(ytrain_pred, threshold=self.opt_threshold)
        yval_pred_labels = self.get_labels_from_prob(yval_pred, threshold=self.opt_threshold)

        train_accuracy = accuracy_score(ytrain, ytrain_pred_labels)
        val_accuracy = accuracy_score(yval, yval_pred_labels)

        train_f1_score = f1_score(ytrain, ytrain_pred_labels)
        val_f1_score = f1_score(yval, yval_pred_labels)
        print (f"train accuracy: {train_accuracy}, train_f1_score: {train_f1_score},"
               f"val accuracy: {val_accuracy}, val_f1_score: {val_f1_score} ")

        try:
            foundations.log_metric('train_accuracy',np.round(train_accuracy,2))
            foundations.log_metric('val_accuracy', np.round(val_accuracy,2))
            foundations.log_metric('train_f1_score', np.round(train_f1_score,2))
            foundations.log_metric('val_f1_score', np.round(val_f1_score,2))
            foundations.log_metric('optimum_threshold', np.round(self.opt_threshold,2))
        except Exception as e:
            print(e)

        # True Positive Example
        ind_tp = np.argwhere(np.equal((yval_pred_labels + yval).astype(int), 2)).reshape(-1, )

        # True Negative Example
        ind_tn = np.argwhere(np.equal((yval_pred_labels + yval).astype(int), 0)).reshape(-1, )

        # False Positive Example
        ind_fp =np.argwhere( np.greater(yval_pred_labels, yval)).reshape(-1, )

        # False Negative Example
        ind_fn = np.argwhere(np.greater(yval, yval_pred_labels)).reshape(-1, )


        path_to_save_spetrograms = './spectrograms'
        if not os.path.isdir(path_to_save_spetrograms):
            os.makedirs(path_to_save_spetrograms)
        specs_saved = os.listdir(path_to_save_spetrograms)
        if len(specs_saved)>0:
            for file_ in specs_saved:
                os.remove(os.path.join(path_to_save_spetrograms,file_))

        ind_random_tp = np.random.choice(ind_tp, num_examples).reshape(-1,)
        tp_x = [xtrain[i] for i in ind_random_tp]

        ind_random_tn = np.random.choice(ind_tn, num_examples).reshape(-1,)
        tn_x = [xtrain[i] for i in ind_random_tn]

        ind_random_fp = np.random.choice(ind_fp, num_examples).reshape(-1,)
        fp_x = [xtrain[i] for i in ind_random_fp]

        ind_random_fn = np.random.choice(ind_fn, num_examples).reshape(-1,)
        fn_x = [xtrain[i] for i in ind_random_fn]

        print("Plotting spectrograms to show what the hell the model has learned")
        for i in range(num_examples):
            plot_spectrogram(tp_x[i], path=os.path.join(path_to_save_spetrograms, f'true_positive_{i}.png'))
            plot_spectrogram(tn_x[i], path=os.path.join(path_to_save_spetrograms,f'true_negative_{i}.png'))
            plot_spectrogram(fp_x[i], path=os.path.join(path_to_save_spetrograms,f'false_positive_{i}.png'))
            plot_spectrogram(fn_x[i], path=os.path.join(path_to_save_spetrograms,f'fale_negative_{i}.png'))

        try:
            foundations.save_artifact(os.path.join(path_to_save_spetrograms, f'true_positive_{i}.png'), key='true_positive_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'true_negative_{i}.png'), key='true_negative_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'false_positive_{i}.png'), key='false_positive_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'fale_negative_{i}.png'), key='false_negative_example')

        except Exception as e:
            print(e)