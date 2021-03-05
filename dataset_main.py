from numpy import array
from pickle import dump
from PLSParameters import PLSParameters
from Node import Node


max_SNR = 50
SNR_dB = range(0, max_SNR, 10)
max_iter = 100000

pls_profiles = {
               0: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 1},
               1: {'bandwidth': 960e3, 'bin_spacing': 15e3, 'num_ant': 2, 'bit_codebook': 2},
               }

for prof in pls_profiles.values():
    pls_params = PLSParameters(prof)
    codebook = pls_params.codebook_gen()
    N = Node(pls_params)  # Wireless network node - could be Alice or Bob
    for s in range(len(SNR_dB)):
        precoders = []
        labels = []
        for i in range(max_iter):
            HAB, HBA = pls_params.channel_gen()

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
            precoders.append(rx_precoder.flatten())

            tx_PMI = bits_subbandB[0]

            weights = 2 ** array(range(len(tx_PMI) - 1, -1, -1))
            tx_PMI = sum(tx_PMI * weights)

            labels.append(tx_PMI)

        file_name = f'precoder_data_{pls_params.num_ant}_ant_SNR_{SNR_dB[s]}dB_{pls_params.bit_codebook}_bit_codebk'
        with open(f'{file_name}.pkl', 'wb') as f:
            dump([precoders, labels], f)


