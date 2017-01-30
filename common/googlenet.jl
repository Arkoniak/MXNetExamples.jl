module Googlenet
using MXNet

function ConvFactory(data, num_filter, kernel; stride=(1, 1), pad=(0, 0), name="", suffix="")
  conv = @mx.chain mx.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                        name=Symbol("conv_"*name*suffix)) =>
         mx.Activation(act_type=:relu, name=Symbol("relu_"*name*suffix))

  return conv
end

function InceptionFactory(data, num_1x1m num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name)
  # 1x1
  c1x1 = ConvFactory(data, num_1x1, (1, 1), name=name*"_1x1")

  # 3x3 reduce + 3x3
  c3x3r = ConvFactory(data, num_3x3red, (1, 1), name=name*"_3x3", suffix="_reduce")
  c3x3 = ConvFactory(c3x3r, num_3x3, (3, 3), pad=(1, 1), name=name*"_3x3")

  # double 3x3 reduce + double 3x3
  cd5x5r = ConvFactory(data, num_d5x5red, (1, 1), name=name*"_3x3", suffix="_reduce")
  cd5x5 = ConvFactory(data, num_d5x5, (5, 5), pad=(2, 2), name=name*"_5x5")

  # pool + proj
  pooling = mx.Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=pool*"_pool_"*name*"_pool")
  cproj = ConvFactory(pooling, proj, (1, 1), name=name*"_proj")

  # concat
  concat = mx.Concat([c1x1, c3x3, cd5x5, cproj], name="ch_concat_"*name*"_chconcat")

  return concat
end

function get_symbol(num_classes=10)
  googlenet = @mx.chain mx.Variable(:data) =>
    ConvFactory(64, (7, 7), stride=(2, 2), pad=(3, 3), name="conv1") =>
    mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max) =>
    ConvFactory(64, (1, 1), stride=(1, 1), name="conv2") =>
    ConvFactory(192, (3, 3), stride=(1, 1), pad=(1, 1), name="conv3") =>
    mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max) =>

    InceptionFactory(64, 96, 128, 16, 32, :max, 32, "in3a") =>
    InceptionFactory(128, 128, 192, 32, 96, :max, 64, "in3b") =>
    mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max) =>

    InceptionFactory(192, 96, 208, 16, 48, :max, 64, "in4a") =>
    InceptionFactory(160, 112, 224, 24, 64, :max, 64, "in4b") =>
    InceptionFactory(128, 128, 256, 24, 64, :max, 64, "in4c") =>
    InceptionFactory(112, 144, 288, 32, 64, :max, 64, "in4d") =>
    InceptionFactory(256, 160, 320, 32, 128, :max, 128, "in4e") =>
    mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max) =>

    InceptionFactory(256, 160, 320, 32, 128, :max, 128, "in5a") =>
    InceptionFactory(384, 192, 384, 48, 128, :max, 128, "in5b") =>
    mx.Pooling(kernel=(7, 7), stride=(1, 1), pool_type=:avg) =>

    mx.Flatten() =>
    mx.FullyConnected(num_hidden=num_classes) =>
    mx.SoftmaxOutput(name=:softmax)

  return googlenet
end

end
