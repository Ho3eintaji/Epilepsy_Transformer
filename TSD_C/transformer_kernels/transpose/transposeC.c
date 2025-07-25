//
// Created by alireza on 10/6/23.
//

#include "softmaxC.h"

// rearranges the data from a multi-head attention format, specifically
//  transposing the output from the multi-head attention mechanism. This function reorganizes the input data 
//  from a layout where the sequence length, number of heads, and head hidden size are interleaved, into another layout.
// input order:  head 1, token 1 | head 1, token 2 | ... | head 2, token 1 | head 2, token 2 | ...
// output order: token 1, head 1 | token 1, head 2 | ... | token 2, head 1 | token 2, head 2 | ...
void multihead_transpose(const quant_bit_width * input, quant_bit_width* output, size_t seq_len,
                         size_t head_hidden_size, size_t num_head) {
    const quant_bit_width * initial_input = input;
    for (int i=0; i < seq_len; i++){
        for (int n=0; n< num_head; n++){
            input = initial_input + i*head_hidden_size + n*seq_len*head_hidden_size;
            for (int j=0; j < head_hidden_size; j++){
                *output++ = *input++;
            }
        }
    }
}


void transpose_quant(const quant_bit_width * input, quant_bit_width* output,
                     size_t width, size_t height) {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            output[i * width + j] = input[j * height + i];
        }
    }
}


