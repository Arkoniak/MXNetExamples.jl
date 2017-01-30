module Lenet
using MXNet

"""
LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE (1998)
"""
function get_symbol(num_classes = 10)
  # input
  data = mx.Variable(:data)

  # first conv
  conv1 = @mx.chain mx.Convolution(data, kernel=(5,5), num_filter=20)  =>
                    mx.Activation(act_type=:tanh) =>
                    mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

  # second conv
  conv2 = @mx.chain mx.Convolution(conv1, kernel=(5,5), num_filter=50) =>
                    mx.Activation(act_type=:tanh) =>
                    mx.Pooling(pool_type=:max, kernel=(2,2), stride=(2,2))

  # first fully-connected
  fc1   = @mx.chain mx.Flatten(conv2) =>
                    mx.FullyConnected(num_hidden=500) =>
                    mx.Activation(act_type=:tanh)

  # second fully-connected
  fc2   = mx.FullyConnected(fc1, num_hidden=num_classes)

  # softmax loss
  lenet = mx.SoftmaxOutput(fc2, name=:softmax)

  return lenet
end
end
