# FastSpeech hyperparameters
encoder_layers = 6
encoder_input_size = 384

decoder_layers = 6
decoder_input_size = 384

dp_input_size = 384
dp_hidden_size = 256
dp_output_size = 384
dp_kernel_size = 3
dp_alpha = 1

vocab_size = 51

fft_emb = 384
fft_att_hidden = 384
fft_conv_hidden = 384
fft_conv_kernel = 3

att_heads = 2

conv_kernel = 3

conv1_input_size = 384
conv1_output_size = 1536
conv2_input_size = 1536
conv2_output_size = 384

linear_input = 384
linear_mel = 80

dropout = 0.1

# Noam Optimizer
warmup = 4000

# Training
epochs = 80000