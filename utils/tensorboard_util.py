# Tensorboard visualization for model weights https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md

def weight_histograms_const(writer, global_step, model_name, model_network, weights, res, walltime):
    flattened_weights = weights.flatten()
    tag = f"{model_name}/{model_network}b{res}/const"
    writer.add_histogram(tag, flattened_weights, global_step=global_step, bins='tensorflow', walltime=walltime)

def weight_histograms_conv2d(writer, global_step, model_name, model_network, conv_name, weights, res, walltime):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    tag = f"{model_name}/{model_network}'b{res}'/{conv_name}/"
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        writer.add_histogram(tag+f"kernel_{k}", flattened_weights, global_step=global_step, bins='tensorflow', walltime=walltime)
    
def weight_histograms_linear(writer, global_step, model_name, model_network, weights, weight_name, walltime):
  flattened_weights = weights.flatten()
  tag = f"{model_name}/{model_network}"+weight_name
  writer.add_histogram(tag, flattened_weights, global_step=global_step, bins='tensorflow', walltime=walltime)
  
def weight_histograms_generator(writer, global_step, module, model_name, walltime):
    # Iterate over all synthesis blocks
    for res in module.synthesis.block_resolutions:
        block = getattr(module.synthesis, f'b{res}')
        if res==4:
            weight_histograms_const(writer, global_step, model_name, "synthesis/", block.const, res, walltime)
            conv = getattr(block, "conv1")
            weight_histograms_conv2d(writer, global_step, model_name, "synthesis/", "conv1", conv.weight, res, walltime)
            weight_histograms_linear(writer, global_step, model_name, "synthesis/", conv.affine.weight, f"b{res}/conv1/affine", walltime)
        else:
            conv = getattr(block, "conv0")
            weight_histograms_conv2d(writer, global_step, model_name, "synthesis/", "conv0", conv.weight, res, walltime)
            weight_histograms_linear(writer, global_step, model_name, "synthesis/", conv.affine.weight, f"b{res}/conv0/affine", walltime)
            conv = getattr(block, "conv1")
            weight_histograms_conv2d(writer, global_step, model_name, "synthesis/", "conv1", conv.weight, res, walltime)
            weight_histograms_linear(writer, global_step, model_name, "synthesis/", conv.affine.weight, f"b{res}/conv1/affine", walltime)
        torgb = getattr(block, "torgb")
        weight_histograms_conv2d(writer, global_step, model_name, "synthesis/", "torgb", torgb.weight, res, walltime)
        weight_histograms_linear(writer, global_step, model_name, "synthesis/", torgb.affine.weight, f"b{res}/torgb/affine", walltime)

    for idx in range(module.mapping.num_layers):
        layer = getattr(module.mapping, f"fc{idx}")
        weight_histograms_linear(writer, global_step, model_name, "mapping/", layer.weight, f"fc{idx}", walltime)
        
def weight_histograms_discriminator(writer, global_step, module, model_name, walltime):
    for res in module.block_resolutions:
        block = getattr(module, f'b{res}')
        if res==4:
            weight_histograms_conv2d(writer, global_step, model_name, "", "conv", block.conv.weight, res, walltime)
            weight_histograms_linear(writer, global_step, model_name, "", block.fc.weight, f"b{res}/fc", walltime)
            weight_histograms_linear(writer, global_step, model_name, "", block.out.weight, f"b{res}/out", walltime)
        
        else:
            if res==256:
                weight_histograms_conv2d(writer, global_step, model_name, "", "fromrgb", block.fromrgb.weight, res, walltime)
            weight_histograms_conv2d(writer, global_step, model_name, "", "conv0", block.conv0.weight, res, walltime)
            weight_histograms_conv2d(writer, global_step, model_name, "", "conv1", block.conv1.weight, res, walltime)
            weight_histograms_conv2d(writer, global_step, model_name, "", "skip", block.skip.weight, res, walltime)