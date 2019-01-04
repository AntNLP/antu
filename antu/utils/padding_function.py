

def shadow_padding(batch_input, vocabulary):
    maxlen = 0
    for ins in batch_input:
        for field_name in ins:
            for vocab_name in ins[field_name]:
                maxlen = max(maxlen, len(ins[field_name][vocab_name]))

    masks = dict()
    inputs = dict()
    for ins in batch_input:
        for field_name in ins:
            for vocab_name in ins[field_name]:
                if vocab_name in vocabulary.no_pad_namespace:
                    continue
                else:
                    print(field_name, vocab_name)
                    print(ins[field_name][vocab_name])
                    padding_length = maxlen - len(ins[field_name][vocab_name])
                    padding_index = vocabulary.get_padding_index(vocab_name)
                    padding_list = [padding_index] * padding_length
                    ins[field_name][vocab_name].extend(padding_list)
                    ins_mask = [1]*(maxlen-padding_length) + [0]*padding_length
                    if field_name not in masks:
                        masks[field_name] = dict()
                    if vocab_name not in masks[field_name]:
                        masks[field_name][vocab_name] = []
                    masks[field_name][vocab_name].append(ins_mask)
                    if field_name not in inputs:
                        inputs[field_name] = dict()
                    if vocab_name not in inputs[field_name]:
                        inputs[field_name][vocab_name] = []
                    inputs[field_name][vocab_name].append(ins[field_name][vocab_name])

    return masks, inputs