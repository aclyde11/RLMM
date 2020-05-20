# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import numpy as np
#
# from ray.rllib.models.modelv2 import ModelV2
# from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils import try_import_tf
#
# tf = try_import_tf()
#
#
#
# class MyKerasRNN(RecurrentTFModelV2):
#     """Example of using the Keras functional API to define a RNN model."""
#
#     def __init__(self,
#                  obs_space,
#                  action_space,
#                  num_outputs,
#                  model_config,
#                  name,
#                  hiddens_size=384,
#                  cell_size=384, envconf):
#         super(MyKerasRNN, self).__init__(obs_space, action_space, num_outputs,
#                                          model_config, name)
#         self.cell_size = cell_size
#         # Define input layers
#         input_layer = tf.keras.layers.Input(
#             shape=(None, envconf['output_size'][0], envconf['output_size'][1], envconf['output_size'][2], envconf['output_size'][3]), name="inputs")
#         state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
#         state_in_c = tf.keras.layers.Input(shape=(cell_size,), name="c")
#         seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)
#
#         # Preprocess observation with a hidden layer and send to LSTM cell
#         h = tf.keras.layers.Reshape([-1] + list(envconf['output_size']))(input_layer)
#
#         h = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.Conv3D(filters=64, kernel_size=8, padding='valid', name='notconv1'))(h)
#         h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(64, 8, padding='valid', name='conv3d_2'))(h)
#         h = tf.keras.layers.LeakyReLU(alpha=0.1)(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
#                                                                          strides=None,
#                                                                          padding='valid'))(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding='valid', name='notconv12'))(h)
#         h = tf.keras.layers.ReLU()(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22'))(h)
#         h = tf.keras.layers.ReLU()(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(32, 3, padding='valid', name='conv3d_22'))(h)
#         h = tf.keras.layers.ReLU()(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
#                                                                          strides=None,
#                                                                          padding='valid'))(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(32, 2, padding='valid', name='conv3d_22'))(h)
#         h = tf.keras.layers.ReLU()(h)
#         h = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv3D(24, 2, padding='valid', name='conv3d_22'))(h)
#         h = tf.keras.layers.ReLU()(h)
#
#         h = tf.keras.layers.Reshape([-1, 3 * 3 * 3 * 24])(h)
#
#         state_vec = tf.keras.layers.Input(shape=(None,2), name='state_vec_input')
#         h2 = tf.keras.layers.Dense(16, activation=tf.nn.relu, name='st1')(state_vec)
#         h = tf.keras.layers.Concatenate()([h, h2])
#         dense1 = tf.keras.layers.Dense(
#             hiddens_size, activation=tf.nn.relu, name="dense1")(h)
#         lstm_out, state_h, state_c = tf.keras.layers.LSTM(
#             cell_size, return_sequences=True, return_state=True, name="lstm")(
#             inputs=dense1,
#             mask=tf.sequence_mask(seq_in),
#             initial_state=[state_in_h, state_in_c])
#
#         # Postprocess LSTM output with another hidden layer and compute values
#         logits = tf.keras.layers.Dense(
#             self.num_outputs,
#             activation=tf.keras.activations.linear,
#             name="logits")(lstm_out)
#         values = tf.keras.layers.Dense(
#             1, activation=None, name="values")(lstm_out)
#
#         # Create the RNN model
#         self.rnn_model = tf.keras.Model(
#             inputs=[input_layer, seq_in, state_in_h, state_in_c, state_vec],
#             outputs=[logits, values, state_h, state_c])
#         self.register_variables(self.rnn_model.variables)
#         # self.rnn_model.summary()
#
#     @override(RecurrentTFModelV2)
#     def forward_rnn(self, inputs, state, seq_lens):
#         # print("Forward rnn", inputs.shape, inputs[0].shape, inputs[1].shape)
#         model_out, self._value_out, h, c = self.rnn_model([inputs[:,:,:-2], seq_lens] +
#                                                           state + [inputs[:,:,-2:]])
#         return model_out, [h, c]
#
#     @override(ModelV2)
#     def get_initial_state(self):
#         return [
#             np.zeros(self.cell_size, np.float32),
#             np.zeros(self.cell_size, np.float32),
#         ]
#
#     @override(ModelV2)
#     def value_function(self):
#         return tf.reshape(self._value_out, [-1])
