from numpy import array, conj, sqrt, zeros
from numpy.random import uniform, normal, randint
from pickle import dump
import os
from PLSParameters import PLSParameters
from Node import Node

max_iter = 100000 # Number of entries in the dataset
max_SNR = 50
SNR_dB = range(0, max_SNR, 5)
fading = 1 # 0 - AWGN only, 1 - with fading channel
num_ant = 2 # Number of antennas
bit_codebook = 1 # Bits per codebook index
num_classes = 2**bit_codebook # Number of classes that the classifer needs to learn

pls_params = PLSParameters(num_ant, bit_codebook)
codebook = pls_params.codebook_gen() # DFT codebook


for s in range(len(SNR_dB)):
    SNR_lin = 10**(SNR_dB[s]/10)
    precoders = []
    labels = []
    for i in range(max_iter):
        if fading == 0:
            tx_PMI = randint(0, num_classes)  # generate random precoder index

            precoder = codebook[tx_PMI]

            prec_power = sum(sum(precoder * conj(precoder))) / (num_ant ** 2)

            noise_var = abs(prec_power) / SNR_lin

            noise = normal(0, sqrt(noise_var), (num_ant, num_ant)) + 1j * normal(0, sqrt(noise_var),
                                                                                 (num_ant, num_ant))

            # Add noise
            rx_precoder = precoder + noise
        else:
            HAB, HBA = pls_params.channel_gen()

            N = Node(pls_params)  # Wireless network node - could be Alice or Bob
            # 1. Alice to Bob
            GA = N.unitary_gen()
            rx_sigB0 = N.receive('Bob', SNR_dB[s], HAB, GA)

            # 1. At Bob
            UB0 = N.sv_decomp(rx_sigB0)[0]
            bits_subbandB = N.secret_key_gen()
            FB = N.precoder_select(bits_subbandB, codebook)

            # 2. Bob to Alice
            rx_sigA = N.receive('Alice', SNR_dB[s], HBA, UB0, FB)

            # 2. At Alice
            UA, _, VA = N.sv_decomp(rx_sigA)
            # VA is the recived precoder

            rx_precoder = VA[0]

            tx_PMI = bits_subbandB[0]

            weights = 2 ** array(range(len(tx_PMI) - 1, -1, -1))
            tx_PMI = sum(tx_PMI * weights)
        
        precoder_mat2array = rx_precoder.flatten()
        precoder_real = precoder_mat2array.real
        precoder_imag = precoder_mat2array.imag
        
        precoder_real_imag = zeros(2*len(precoder_real), dtype=float)
        precoder_real_imag[0::2] = precoder_real
        precoder_real_imag[1::2] = precoder_imag
        precoders.append(precoder_real_imag)
        labels.append(tx_PMI)

    dir_name = 'datasets'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_name = f'{dir_name}/{fading}_precoder_data_{pls_params.num_ant}_ant_SNR_{SNR_dB[s]}dB_{pls_params.bit_codebook}_bit_codebk'
    with open(f'{file_name}.pkl', 'wb') as f:
        dump([precoders, labels], f)


