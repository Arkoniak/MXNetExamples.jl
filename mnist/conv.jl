using MXNet
using ArgParse

include(joinpath(dirname(@__FILE__), "..", "common", "mnist-data.jl"))

function make_conv_arch(ndf=20, fix_gamma=true, no_bias=true, eps=1e-5+1e-12)
  dloss = @mx.chain mx.Variable(:data) =>
          mx.Convolution(name=:d1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf, no_bias=no_bias) =>
          mx.LeakyReLU(name=:dact1, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d2, kernel=(4, 4), stride=(2, 2), num_filter=ndf*2, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn2, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact2, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d3, kernel=(4, 4), stride=(2, 2), num_filter=ndf*4, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn3, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact3, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d4, kernel=(2, 2), num_filter=1, no_bias=no_bias) =>
          mx.Flatten() =>

          mx.FullyConnected(name=:fc1, num_hidden=10) =>
          mx.SoftmaxOutput(name=:softmax)

  return dloss
end

function make_lenet_arch()
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
  fc2   = mx.FullyConnected(fc1, num_hidden=10)

  # softmax loss
  lenet = mx.SoftmaxOutput(fc2, name=:softmax)
end

########################################################
# ArgParse part
########################################################
function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--gpu", "-g"
      arg_type = Int
      help = "GPU device number"
    "--arch", "-a"
      default = "conv"
      help = "Architecture type: conv or lenet"
  end

  return parse_args(ARGS, s)
end

parsed_args = parse_commandline()
if parsed_args["gpu"] !== nothing
  ctx = mx.gpu(parsed_args["gpu"])
else
  ctx = mx.cpu()
end

train_data, test_data = get_mnist_providers(100, flat=false)
if parsed_args["arch"] == "conv"
  arch = make_conv_arch()
else
  arch = make_lenet_arch()
end

model = mx.FeedForward(arch, context=ctx)

mx.fit(model, mx.ADAM(), train_data, n_epoch = 10, eval_data = test_data, callbacks=[mx.speedometer(), mx.do_checkpoint("conv")])

#= mod = mx.Module.SymbolModule() =#
