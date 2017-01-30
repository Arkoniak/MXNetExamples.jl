using MXNet

function get_mnist_data(batch_size=100, label_name=:dloss_label)
  include(joinpath(dirname(@__FILE__), "..", "common", "mnist-data.jl"))
  return get_mnist_providers(batch_size, label_name=label_name)
end

function make_conv_arch(ndf=20, fix_gamma=true, no_bias=true, eps=1e-5+1e-12)
  dloss = @mx.chain mx.Variable(:data) =>
          mx.Reshape(shape=(28, 28, 1, -1)) =>
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
          mx.SoftmaxOutput(name=:dloss)

  return dloss
end

ctx = mx.cpu()

train_data, test_data = get_mnist_data(100)
arch = make_conv_arch()
model = mx.FeedForward(arch, context=ctx)

mx.fit(model, mx.ADAM(), train_data, n_epoch = 10, eval_data = test_data, callbacks=[mx.speedometer(), mx.do_checkpoint("conv")])

#= mod = mx.Module.SymbolModule() =#
