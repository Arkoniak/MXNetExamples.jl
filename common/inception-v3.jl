module InceptionV3
using MXNet

"""
Inception V3, suitable for images with around 299 x 299

Reference:

Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision." arXiv preprint arXiv:1512.00567 (2015).
"""

function ConvFactory(data, num_filter; kernel=(1, 1), stride=(1, 1), pad=(0, 0), name="", suffix="")
  conv = @mx.chain mx.Convolution(data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                        name=Symbol("conv_"*name*suffix), no_bias=true) =>
         mx.BatchNorm(name="batchorm_"*name*suffix, fix_gamma=true) =>
         mx.Activation(act_type=:relu, name=Symbol("relu_"*name*suffix))

  return conv
end

function Inception7A(data, num_1x1
                     num_3x3_red, num_3x3_1, num_3x3_2,
                     num_5x5_red, num_5x5,
                     pool, proj,
                     name)
  tower_1x1 = ConvFactory(data, num_1x1, name=name*"_conv")
  tower_5x5 = @mx.chain ConvFactory(data, num_5x5_red, name=name*"_tower", suffix="_conv") =>
              ConvFactory(num_5x5, kernel=(5, 5), pad=(2, 2), name=name*"_tower", suffix="_conv_1")

  tower_3x3 = @mx.chain ConvFactory(data, num_3x3_red, name=name*"_tower_1", suffix="_conv") =>
              ConvFactory(num_3x3_1, kernel=(3, 3), pad=(1, 1), name=name*"_tower_1", suffix="_conv_1") =>
              ConvFactory(num_3x3_2, kernel=(3, 3), pad=(1, 1), name=name*"_tower_1", suffix="_conv_2")
  
  cproj = @mx.chain mx.Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=pool*"_pool_"*name*"_pool") =>
          ConvFactory(proj, name=name*"_tower_2", suffix="_conv")

  concat = mx.Concat(tower_1x1, tower_5x5, tower_3x3, cproj, name="ch_concat_"*name*"_chconcat")

  return concat
end

# First Downsample
function Inception7B(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                pool,
                name)
  tower_3x3 = ConvFactory(num_3x3, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=name*"_conv")
  tower_d3x3 = @mx.chain ConvFactory(num_d3x3_red, name=name*"_tower", suffix="_conv") =>
               ConvFactory(num_d3x3_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=name*"_tower", suffix="_conv_1") =>
               ConvFactory(num_d3x3_2, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=name*"_tower", suffix="_conv_2")

  pooling = mx.Pooling(data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type=:max, name="max_pool_"*name*"_pool")

  concat = mx.Concat(tower_3x3, tower_d3x3, pooling, name="ch_concat_"*name*"_chconcat")

  return concat
end

function Inception7C(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                pool, proj,
                name)
  tower_1x1 = ConvFactory(data, num_1x1, kernel=(1, 1), name=name*"_conv")

  tower_d7 = @mx.chain ConvFactory(data, num_d7_red, name=name*"_tower", suffix="_conv") =>
             ConvFactory(num_d7_1, kernel=(1, 7), pad=(0, 3), name=name*"_tower", suffix="_conv_1") =>
             ConvFactory(num_d7_2, kernel=(7, 1), pad=(3, 0), name=name*"_tower", suffix="_conv_2")

  tower_q7 = @mx.chain ConvFactory(data, num_q7_red, name=name*"tower_1", suffix="_conv") =>
             ConvFactory(num_q7_1, kernel=(7, 1), pad=(3, 0), name=name*"tower_1", suffix="_conv_1") =>
             ConvFactory(num_q7_2, kernel=(1, 7), pad=(0, 3), name=name*"tower_1", suffix="_conv_2") =>
             ConvFactory(num_q7_3, kernel=(7, 1), pad=(3, 0), name=name*"tower_1", suffix="_conv_3") =>
             ConvFactory(num_q7_4, kernel=(1, 7), pad=(0, 3), name=name*"tower_1", suffix="_conv_4")

  cproj = @mx.chain mx.Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=pool*"_pool_"*name*"_pool") =>
          ConvFactory(proj, kernel=(1, 1), name=name*"_tower_2", suffix="_conv")

  concat = mx.Concat(tower_1x1, tower_d7, tower_q7, cproj, name="ch_concat_"*name*"_chconcat")

  return concat
end

function Inception7D(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                pool,
                name)
  tower_3x3 = @mx.chain ConvFactory(data, num_3x3_red, name=name*"_tower", suffix="_conv") =>
              ConvFactory(num_3x3, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=name*"_tower", suffix="_conv_1")

  tower_d7_3x3 = @mx.chain ConvFactory(data, num_d7_3x3_red, name=name*"_tower_1", suffix="_conv") =>
                 ConvFactory(num_d7_1, kernel=(1, 7), pad=(0, 3), name=name*"_tower_1", suffix="_conv_1") =>
                 ConvFactory(num_d7_2, kernel=(7, 1), pad=(3, 0), name=name*"_tower_1", suffix="_conv_2") =>
                 ConvFactory(num_d7_3x3, kernel=(3, 3), stride=(2, 2), name=name*"_tower_1", suffix="_conv_3")

  pooling = mx.Pooling(data, kernel=(3, 3), stride=(2, 2), pool_type=pool, name=pool*"_pool_"*name*"_pool")

  concat = mx.Concat(tower_3x3, tower_d7_3x3, pooling, name="ch_concat_"*name*"_chconcat")

  return concat
end

function Inception7E(data,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                pool, proj,
                name)
  tower_1x1 = ConvFactory(data, num_1x1, kernel=(1, 1), name=name*"_conv")
  tower_d3 = ConvFactory(data, num_d3_red, name=name*"_tower", suffix="_conv")
  tower_d3_a = ConvFactory(tower_d3, num_d3_1, kernel=(1, 3), pad=(0, 1), name=name*"_tower", suffix="_mixed_conv")
  tower_d3_b = ConvFactory(tower_d3, num_d3_2, kernel=(3, 1), pad=(1, 0), name=name*"_tower", suffix="_mixed_conv_1")

  tower_3x3_d3 = @mx.chain ConvFactory(data, num_3x3_d3_red, name=name*"_tower_1", suffix="_conv") =>
                 ConvFactory(num_3x3, kernel=(3, 3), pad=(1, 1), name=name*"_tower_1", suffix="_conv_1")

  tower_3x3_d3a = ConvFactory(tower_3x3_d3, num_3x3_d3_1, kernel=(1, 3), pad=(0, 1), name=name*"_tower_1", suffix="_mixed_conv")
  tower_3x3_d3b = ConvFactory(tower_3x3_d3, num_3x3_d3_2, kernel=(3, 1), pad=(1, 0), name=name*"_tower_1", suffix="_mixed_conv_1")

  cproj = @mx.chain mx.Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=pool*"_pool_"*name*"_pool") =>
          ConvFactory(proj, kernel=(1, 1), name=name*"_tower_2", suffix="conv")

  concat = mx.Concat(tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj, name="ch_concat_"*name*"_chconcat")

  return concat
end

function get_symbol(num_classes=1000)
  arch = @mx.chain mx.Variable(:data) =>

         # stage 1
         ConvFactory(32, kernel=(3, 3), stride=(2, 2), name="conv") =>
         ConvFactory(32, kernel=(3, 3), name="conv_1") =>
         ConvFactory(64, kernel=(3, 3), pad=(1, 1), name="conv_2") =>
         mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max, name=:pool) =>

         # stage 2
         ConvFactory(80, kernel=(1, 1), name="conv_3") =>
         ConvFactory(192, kernel=(3, 3), name="conv_4") => 
         mx.Pooling(kernel=(3, 3), stride=(2, 2), pool_type=:max, name=:pool1) =>

         # stage 3
         Inception7A(64,
                     64, 96, 96,
                     48, 64,
                     :avg, 32, "mixed") =>
         Inception7A(64,
                     64, 96, 96,
                     48, 64,
                     :avg, 64, "mixed_1") =>
         Inception7A(64,
                     64, 96, 96,
                     48, 64,
                     :avg, 64, "mixed_2") =>
         Inception7B(384,
                     64, 96, 96,
                     :max, "mixed_3") =>

         # stage 4
         Inception7C(192,
                     128, 128, 192,
                     128, 128, 128, 128, 192,
                     :avg, 192, "mixed_4") =>
         Inception7C(192,
                     160, 160, 192,
                     160, 160, 160, 160, 192,
                     :avg, 192, "mixed_5") =>
         Inception7C(192,
                     160, 160, 192,
                     160, 160, 160, 160, 192,
                     :avg, 192, "mixed_6") =>
         Inception7C(192,
                     192, 192, 192,
                     192, 192, 192, 192, 192,
                     :avg, 192, "mixed_7") =>
         Inception7D(192, 320,
                     192, 192, 192, 192,
                     :max, "mixed_8") =>

         # stage 5
         Inception7E(320,
                     384, 384, 384,
                     448, 384, 384, 384,
                     :avg, 192, "mixed_9") =>
         Inception7E(320,
                     384, 384, 384,
                     448, 384, 384, 384,
                     :max, 192, "mixed_10") =>

         # pool
         mx.Pooling(kernel=(8, 8), stride=(1, 1), pool_type=:avg, name=:global_pool) =>
         mx.Flatten(name=:flatten) =>
         mx.FullyConnected(num_hidden=num_classes, name=:fc1) =>
         mx.SoftmaxOutput(name=:softmax)
   return arch
end

end
