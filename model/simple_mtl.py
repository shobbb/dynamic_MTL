import tensorflow as tf
tfk = tf.keras

class SimpleMTL:
    def __init__(self, config):
        self.config = config
        self._build()

    def _feature_set_build(self, name='feature_function'):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            fc = tfk.Sequential(name=f'{name}/dense')

            for i, h in enumerate(self.config['feature_func']['linear_hids']):
                fc.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))

            outsize = self.config['feature_func']['outsize']
            num_feature_nodes = self.config['num_feature_nodes']
            fc.add(tfk.layers.Dense(
                outsize * num_feature_nodes, activation=tf.nn.tanh, name=f'final{i}'
            ))

            return tf.reshape(fc(self.input), [-1, num_feature_nodes, outsize])

    def _predict_task_feat_relationships(self, name='predict_relationships'):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            fc = tfk.Sequential(name=f'{name}/dense')

            for i, h in enumerate(self.config['predict_relationships']['linear_hids']):
                    fc.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            
            num_tasks = self.config['num_tasks']
            num_feature_nodes = self.config['num_feature_nodes']
            fc.add(tfk.layers.Dense(
                num_tasks * num_feature_nodes, activation=tf.nn.tanh, name=f'final{i}'
            ))

            return tf.nn.softmax(
                tf.reshape(fc(self.input), [-1, num_tasks, num_feature_nodes])
            )

    def _compute_ith_task(self, task_num, input, name="task_network"):
        name += f"_{task_num}"
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            fc = tfk.Sequential(name=f'{name}/dense')

            for i, h in enumerate(self.config['task_network']['linear_hids']):
                    fc.add(tfk.layers.Dense(h, activation=tf.nn.tanh, name=f'l{i}'))
            fc.add(tfk.layers.Dense(
                1, activation=tf.identity, name=f'final{i}'
            ))

            return fc(input)

    
    def _build(self):
        num_tasks = self.config['num_tasks']
        dim = self.config['dim']

        self.input = tf.compat.v1.placeholder(tf.float32, (None, dim), 'inputs')
        self.ys = tf.compat.v1.placeholder(tf.float32, (None, num_tasks), 'ys')
        print(f'inputs shape: {self.input.get_shape()}')

        # get feature matrix
        self._feature_mat = self._feature_set_build()
        print(f'feature matrix shape: {self._feature_mat.get_shape()}')

        # output task to feature relationships
        self._t_to_feats_mat = self._predict_task_feat_relationships()
        print(f'task to feats mat shape: {self._t_to_feats_mat.get_shape()}')
        self.relationships = tf.matmul(
            self._t_to_feats_mat, tf.transpose(self._t_to_feats_mat, perm=[0, 2, 1])
        )
        print(f'relationships shape {self.relationships.get_shape()}')

        self._weighted_feat_sum = tf.matmul(self._t_to_feats_mat, self._feature_mat)
        print(f'weighted feat sum shape: {self._weighted_feat_sum.get_shape()}')
        self._task_input = tf.concat([
            tf.tile(
                tf.expand_dims(self.input, 1), [1, num_tasks, 1]
            ), self._weighted_feat_sum
        ], -1)

        print(f'task input shape: {self._task_input.get_shape()}')
        self._split_task_input = tf.unstack(self._task_input, axis=1)
        print(f'split task input shape: {self._split_task_input[0].get_shape()}')
        self.output = tf.squeeze(tf.stack([
            self._compute_ith_task(i, self._split_task_input[i]) for i in range(num_tasks)
        ], axis=1))
        self.loss = tf.reduce_mean(tf.math.square(self.output - self.ys))

